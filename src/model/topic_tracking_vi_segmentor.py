'''
Created on Nov 1, 2017

@author: pjdrm
'''
import numpy as np
import copy
from tqdm import trange
from dataset.synthetic_doc_cvb import CVBSynDoc
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.special import gammaln

class TopicTrackingVIModel(object):

    def __init__(self, beta, docs):
        self.beta = beta
        self.W = docs.W
        self.data = Data(docs)
        
        self.dp_matrices = self.init_dp_matrices()
        self.best_lm_word_counts = self.init_best_lm_word_counts() #best_lm_word_counts are the word count from OTHER documents that allowed the highest likelihood of a language model
        self.best_seg_tracker = self.init_seg_tracker()
        
        self.seg_ll_C = gammaln(self.beta.sum())-gammaln(self.beta).sum()
    
    def init_dp_matrices(self):
        dp_matrices = []
        for doc_i in range(self.data.n_docs):
            doc_len = self.data.doc_len(doc_i)
            doc_i_dp_matrix = np.zeros((doc_len, doc_len))
            dp_matrices.append(doc_i_dp_matrix)
        return dp_matrices
    
    def init_cum_sums(self):
        cum_sums = []
        for doc_i in range(self.data.n_docs):
            cum_sums.append(np.cumsum(self.data.doc_word_counts(doc_i), axis=0))
        return cum_sums
        
    def init_best_lm_word_counts(self):
        best_lm_word_counts = [[] for i in range(self.data.n_docs)]
        for doc_lm_word_counts in best_lm_word_counts:
            for lm in range(self.data.max_doc_len):
                word_counts_dict = {"ll": -np.inf, "wc": np.zeros(self.W)}
                doc_lm_word_counts.append(word_counts_dict)
        return best_lm_word_counts
    
    def init_seg_tracker(self):
        best_seg_tracker = []
        for doc_i in range(self.data.n_docs):
            best_seg_tracker.append(np.zeros(self.data.doc_len(doc_i), dtype=np.int32))
        
        return best_seg_tracker
        
    def slice_docs(self, u, doc_i):
        '''
        Computes the commulative counts of u sentence from all documents
        except doc_i.
        :param u: sentence index
        :param doc_i: document index
        '''
        word_counts = np.zeros(self.W)
        for d in range(self.data.n_docs):
            if d == doc_i:
                '''
                We just want to compute the counts from other documents,
                we need them later in the update part.
                '''
                continue
            
            if self.data.doc_len(doc_i) >= u:
                word_counts += self.data.doc_word_counts(d)[u]
        return word_counts
    
    def update_best_lm_word_counts(self, doc_i, lm, word_counts, seg_ll):
        '''
        Checks if the current lm estimate was higher than before (when using one
        less sentence). If yes, the tracker for the word counts is updated.
        :param doc_i: index of document
        :param lm: language model
        :param word_counts: word counts from other documents from the current lm estimate
        :param seg_ll: segment log likelihood from the current lm estimate
        '''
        if seg_ll > self.best_lm_word_counts[doc_i][lm]["ll"]:
            self.best_lm_word_counts[doc_i][lm]["wc"] = word_counts
            self.best_lm_word_counts[doc_i][lm]["ll"] = seg_ll
    
    def get_prev_seg_ll(self, u, lm, doc):
        '''
        Returns the log likelihood of the best segmentation
        of a document without sentence u.  #TODO: rebuild this behavior!!!!
        :param u: sentence
        :param lm: language model (column in the DP matrix)
        :param doc: document
        '''
        if u == 0 or lm == 0:
            #We are just in the first sentence or language model, no previous seg exists
            return 0.0
        
        best_seg_point = self.best_seg_tracker[doc][lm-1]
        prev_seg_ll = self.dp_matrices[doc][lm-1][best_seg_point]
        '''
        prev_seg_ll = 0.0
        for sent in range(u-1,-1,-1):
            best_seg_point = self.best_seg_tracker[doc][sent]
            prev_seg_ll += self.dp_matrices[doc][sent][best_seg_point]
        '''
        return prev_seg_ll
    
    def segment_ll(self, word_counts):
        '''
        Returns the likelihood if we considering all sentences (word_counts)
        as a single language model.
        :param seg_word_counts: vector with the size equal to the length of
        the vocabulary and values with the corresponding word counts.
        '''
        f1 = gammaln(word_counts+self.beta).sum()
        f2 = gammaln((word_counts+self.beta).sum())
        seg_ll = self.seg_ll_C+f1-f2
        return seg_ll
    
    def segment_u(self, u, lm):
        '''
        Estimates, for all documents, the best segmentation
        from u to lm (column index in the DP matrix).
        :param u: sentence index
        :param lm: language model index
        '''
        for doc_i in range(self.data.n_docs):
            word_counts = self.best_lm_word_counts[doc_i][lm]["wc"]
            word_counts_h1 = word_counts+self.slice_docs(u, doc_i) #Note that these are only counts from other documents
            cum_sum = np.sum(self.data.doc_word_counts(doc_i)[lm:u+1,:], axis=0)
            #This assumes a segment u to lm with all documents
            all_docs_u_lm_seg_ll = self.segment_ll(word_counts_h1+cum_sum)
            
            if u > 0 and u != lm:
                #u == lm is the diagonal, for which only one case exists
                word_counts_h2 = word_counts
                #This is the case where I had to the language model on u from doc_i
                u_lm_seg_ll = self.segment_ll(word_counts_h2+cum_sum)#By adding cum_sums[doc_i][u] counts I am just considering the sentence from this doc_i to the LM
            else:
                u_lm_seg_ll = -np.inf
                
            if all_docs_u_lm_seg_ll > u_lm_seg_ll:
                best_word_counts = word_counts_h1
                best_seg_ll = all_docs_u_lm_seg_ll
            else:
                best_word_counts = word_counts_h2
                best_seg_ll = u_lm_seg_ll
                
            prev_seg_ll = self.get_prev_seg_ll(u, lm, doc_i)#TODO: figure out if this is inconsistent. I might be adding sentence counts from other docs to more than one langauge model
            total_seg_ll = prev_seg_ll+best_seg_ll
            self.update_best_lm_word_counts(doc_i, lm, best_word_counts, total_seg_ll)
            self.dp_matrices[doc_i][u][lm] = total_seg_ll
    
    def dp_segmentation(self):
        for u in range(self.data.max_doc_len):
            for lm in range(u+1):
                self.segment_u(u, lm)
            for doc_i in range(self.data.n_docs):
                self.best_seg_tracker[doc_i][u] = np.argmax(self.dp_matrices[doc_i][u,0:u+1])
                
    def get_segmentation(self, doc_i):
        '''
        Returns the final segmentation for a document.
        This is done by backtracking the best segmentations
        in a bottom-up fashion.
        :param doc_i: document index
        '''
        u = len(self.best_seg_tracker[doc_i])-1
        seg_points = []
        while 1:
            seg_point = self.best_seg_tracker[doc_i][u]-1
            if seg_point == -1:
                break
            seg_points.append(seg_point)
            u = seg_point-1
        seg_points.reverse()
        hyp_seg = []
        seg_begin = 0
        for seg_point in seg_points:
            seg_len = seg_point-seg_begin
            hyp_seg += [0]*(seg_len)+[1]
            seg_begin = seg_point+1
            
        seg_len = self.data.doc_len(doc_i)-seg_begin
        hyp_seg += [0]*seg_len
        hyp_seg[-1] = 0
        return hyp_seg
            
class Data(object):
    '''
    Wrapper class for MultiDocument object. Represent the full collection of documents.
    In this segmentor implementation it is convenient to have
    individual word counts for each document. 
    '''
    def __init__(self, docs):
        self.n_docs = docs.n_docs
        self.doc_lens = []
        self.docs_word_counts = []
        self.multi_doc_slicer(docs)
        self.max_doc_len = np.max(self.doc_lens)
        
    def multi_doc_slicer(self, docs):
        doc_begin = 0
        for doc_end in docs.docs_index:
            doc = copy.deepcopy(docs)
            self.doc_lens.append(doc_end - doc_begin)
            U_W_counts = doc.U_W_counts[doc_begin:doc_end, :]
            self.docs_word_counts.append(U_W_counts)
            doc_begin = doc_end
        
    def doc_len(self, doc_i):
        '''
        Returns the length (number of sentences) of doc_i
        :param doc_i: document index
        '''
        return self.doc_lens[doc_i]
        
    def doc_word_counts(self, doc_i):
        '''
        Returns the word count matrix for doc_i
        :param doc_i: document index
        '''
        return self.docs_word_counts[doc_i]
    
W = 5
beta = np.array([0.6]*W)
n_docs = 2
doc_len = 40
pi = .08
sent_len = 15
doc_synth = CVBSynDoc(beta, pi, sent_len, doc_len, n_docs)

vi_tt_model = TopicTrackingVIModel(beta, doc_synth)
vi_tt_model.dp_segmentation()
print(vi_tt_model.get_segmentation(0))
print(doc_synth.rho.tolist())

'''
K = 3
W = 5
alpha = [15]*K
beta = [0.6]*W 
n_words = 1000
n_iters = 1000

doc_synth = CVBSynDoc(alpha, beta, n_words)
vi_tt_model = TopicTrackingVIModel(alpha, beta, doc_synth, n_cvb_iters=n_iters)
vi_tt_model.cvb_algorithm()
hyp_word_topics = vi_tt_model.get_word_topics()
ref_word_topics = doc_synth.Z
print(hyp_word_topics)
print("ARI %f", adjusted_rand_score(ref_word_topics, hyp_word_topics))
'''
        