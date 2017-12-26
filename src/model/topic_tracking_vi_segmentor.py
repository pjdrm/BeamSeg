'''
Created on Nov 1, 2017

@author: pjdrm
'''
import numpy as np
import copy
from tqdm import trange
from dataset.synthetic_doc_cvb import CVBSynDoc
from scipy.special import gammaln
import eval.eval_tools as eval_tools
from itertools import chain, combinations
import time

class TopicTrackingVIModel(object):

    def __init__(self, beta, data):
        self.beta = beta
        self.W = data.W
        self.data = data
        
        self.dp_matrices = self.init_dp_matrices()
        self.best_lm_docs = self.init_best_lm_docs()#keeps track which (other) documents' sentences contributed to the highest likelihood of a language model
        self.best_lm_word_counts = self.init_best_lm_word_counts()#best_lm_word_counts are the word count from OTHER documents that allowed the highest likelihood of a language model
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
        
    def init_best_lm_docs(self):
        best_lm_docs = []
        for doc_i in range(self.data.n_docs):
            best_lm_doc_i = []
            for lm in range(self.data.doc_len(doc_i)):
                lm_doc_i = []
                for doc_j in range(self.data.n_docs):
                    lm_doc_i.append(-1)
                best_lm_doc_i.append(lm_doc_i)
            best_lm_docs.append(best_lm_doc_i)
        return best_lm_docs
                
            
        
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
    
    def possible_other_docs(self, doc, u, lm):
        '''
        Returns the documents (different from doc) that are consistent
        with previous segmentation points. For example, if u=3 and lm=4
        we can only consider other documents for which their u=2 was used
        in lm=4.
        :param doc: document index
        :param u: sentence index
        :param lm: language model index
        '''
        if u == lm:
            '''
            This is the diagonal case for which there are no sentences from other docs
            Thus, there are nos restrictions of other docs.
            '''
            possible_docs = list(range(self.data.n_docs))
            possible_docs.pop(doc)
            return possible_docs
        
        possible_docs = []
        lm_docs = self.best_lm_docs[doc][lm]
        for doc_i in range(self.data.n_docs):
            if doc_i == doc:
                continue
            u_prev = lm_docs[doc_i]
            if u_prev == u-1:
                possible_docs.append(doc_i)
        return possible_docs
        
    def aggregate_u_counts(self, u, docs):
        '''
        Computes the commulative counts of u sentence from all documents.
        :param u: sentence index
        :param doc_i: list of documents
        '''
        word_counts = np.zeros(self.W)
        for doc_i in docs:
            if self.data.doc_len(doc_i) >= u:
                word_counts += self.data.doc_word_counts(doc_i)[u]
        return word_counts
    
    def update_best_lm_docs(self, doc, u, lm, best_doc_comb):
        for doc_i in best_doc_comb:
            self.best_lm_docs[doc][lm][doc_i] = u
    
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
            cum_sum = np.sum(self.data.doc_word_counts(doc_i)[lm:u+1,:], axis=0)
            best_seg_ll = -np.inf #self.segment_ll(word_counts+cum_sum)
            best_doc_comb = None
            other_docs = self.possible_other_docs(doc_i, u, lm)
            
            doc_combs = list(chain.from_iterable(combinations(other_docs, r) for r in range(1,len(other_docs)+1)))#all n combinations of docs
            for doc_comb in doc_combs:
                other_doc_word_counts = word_counts+self.aggregate_u_counts(u, doc_comb)
                seg_ll = self.segment_ll(other_doc_word_counts+cum_sum)
                if seg_ll > best_seg_ll:
                    best_word_counts = other_doc_word_counts
                    best_seg_ll = seg_ll
                    best_doc_comb = doc_comb
                    
            seg_ll_doc_i_u_only = self.segment_ll(word_counts+cum_sum) #case of only adding the u sentence from doc_i
            if seg_ll_doc_i_u_only > best_seg_ll:
                best_seg_ll = seg_ll_doc_i_u_only
                best_doc_comb = [doc_i] #Irrelevant though
                best_word_counts = word_counts
                
            if best_doc_comb is None:
                print()
            prev_seg_ll = self.get_prev_seg_ll(u, lm, doc_i) #TODO: figure out if this is inconsistent. I might be adding sentence counts from other docs to more than one langauge model
            total_seg_ll = prev_seg_ll+best_seg_ll
            self.update_best_lm_docs(doc_i, u, lm, best_doc_comb)
            self.update_best_lm_word_counts(doc_i, lm, best_word_counts, total_seg_ll)
            self.dp_matrices[doc_i][u][lm] = total_seg_ll
            
    
    def dp_segmentation(self):
        for u in range(self.data.max_doc_len):
            for lm in range(u+1):
                #print("u: %d lm: %d"%(u, lm))
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
            if u == -1:
                break
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
    
    def get_all_segmentations(self):
        '''
        Returns a single vector with the final
        segmentation for all documents.
        '''
        all_segs = []
        for doc_i in range(self.data.n_docs):
            all_segs += self.get_segmentation(doc_i)
        return all_segs
    
class Data(object):
    '''
    Wrapper class for MultiDocument object. Represent the full collection of documents.
    In this segmentor implementation it is convenient to have
    individual word counts for each document. 
    '''
    def __init__(self, docs):
        self.W = docs.W
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
    
def sigle_vs_md_eval(doc_synth, beta):
    '''
    Print the WD results when segmenting single documents
    and all of them simultaneously (multi-doc model)
    :param doc_synth: collection of synthetic documents
    :param beta: beta prior vector
    '''
    single_docs = doc_synth.get_single_docs()
    single_doc_wd = []
    start = time.time()
    for doc in single_docs:
        data = Data(doc)
        vi_tt_model = TopicTrackingVIModel(beta, data)
        vi_tt_model.dp_segmentation()
        single_doc_wd += eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc)
    end = time.time()
    sd_time = (end - start)
        
    single_doc_wd = ['%.3f' % wd for wd in single_doc_wd]
    data = Data(doc_synth)
    vi_tt_model = TopicTrackingVIModel(beta, data)
    start = time.time()
    vi_tt_model.dp_segmentation()
    end = time.time()
    md_time = (end - start)
    multi_doc_wd = eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth)
    multi_doc_wd = ['%.3f' % wd for wd in multi_doc_wd]
    print("Single:%s time: %f\nMulti: %s time: %f" % (str(single_doc_wd), sd_time, str(multi_doc_wd), md_time))
    
    
W = 10
beta = np.array([0.6]*W)
n_docs = 5
doc_len = 40 
pi = .2
sent_len = 5
doc_synth = CVBSynDoc(beta, pi, sent_len, doc_len, n_docs)
data = Data(doc_synth)

#sigle_vs_md_eval(doc_synth, beta)

vi_tt_model = TopicTrackingVIModel(beta, data)
vi_tt_model.dp_segmentation()
print(eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth))
