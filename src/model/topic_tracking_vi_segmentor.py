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
        
        self.doc_combs_list = self.init_doc_combs()#All n possible combinations (up to the number of documents). Its a list of pairs where the first element is the combination and second the remaining docs
        self.best_segmentation = [[] for i in range( self.data.max_doc_len)]
        
        self.seg_ll_C = gammaln(self.beta.sum())-gammaln(self.beta).sum()
    
    def init_doc_combs(self):
        all_docs = set(range(self.data.n_docs))
        doc_combs = chain.from_iterable(combinations(all_docs, r) for r in range(1,len(all_docs)+1))
        doc_combs_list = []
        for doc_comb in doc_combs:
            other_docs = all_docs - set(doc_comb)
            doc_combs_list.append([doc_comb, other_docs])
        return doc_combs_list
    
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
    
    def fit_sentence(self, seg_u_list, docs, u_clusters):
        '''
        Adds the u sentences (a segment) of the docs to the best u_cluster. That is,
        the cluster where u-1 (of the corresponding doc) is located. Note
        that the cluster is for the best likelihood, thus, we need to add
        u to the same cluster as u-1.
        :param seg_u_list: list of u indexes corresponding to a segment
        :param docs: list of documents
        :param u_clusters: list of SentenceCluster corresponding to the best segmentation up to u-1
        '''
        for doc in docs:
            for u_cluster in reversed(u_clusters):#We reverse because we are only checking the document list of the cluster
                if u_cluster.has_doc(doc):
                    for u in seg_u_list:
                        u_cluster.add_sent(u, doc)
                    break
        return u_clusters
    
    def print_seg(self, u_clusters, doc_comb):
        print("==========================")
        print("doc_comb: " + str(doc_comb))
        for i, u_cluster in enumerate(u_clusters):
            print("Seg " + str(i))
            print(str(u_cluster.u_list) + " " + str(u_cluster.doc_list))
    
    def segment_u(self, u, lm):
        '''
        Estimates, for all documents, the best segmentation
        from u to lm (column index in the DP matrix).
        :param u: sentence index
        :param lm: language model index
        '''
        if lm == 0:#The first column corresponds to having all sentences from all docs in a single segment (there is only one language model)
            u_cluster = SentenceCluster(u, lm, list(range(self.data.n_docs)), self.data)
            segmentation_ll = self.segment_ll(u_cluster.get_word_counts())
            return segmentation_ll, [u_cluster]
           
        #print("Matrix entry ("+str(u)+", "+str(lm)+")") 
        best_seg_ll = -np.inf
        best_seg_clusters = None
        seg_u_list = range(lm, u+1)
        for doc_comb, other_docs in self.doc_combs_list:
            best_prev_seg = copy.deepcopy(self.best_segmentation[lm-1])
            doc_comb_seg = SentenceCluster(u, lm, doc_comb, self.data)
            best_prev_seg = self.fit_sentence(seg_u_list, other_docs, best_prev_seg)
            segmentation_ll = 0.0
            for u_cluster in best_prev_seg:
                segmentation_ll += self.segment_ll(u_cluster.get_word_counts())
            segmentation_ll += self.segment_ll(doc_comb_seg.word_counts)
            best_prev_seg.append(doc_comb_seg)
            #self.print_seg(best_prev_seg, doc_comb)
            if segmentation_ll >= best_seg_ll:
                best_seg_ll = segmentation_ll
                #best_prev_seg.append(doc_comb_seg)
                best_seg_clusters = best_prev_seg
        #print("==========================")
        return best_seg_ll, best_seg_clusters
            
    def dp_segmentation(self):
        for u in range(self.data.max_doc_len):
            best_seg_ll = -np.inf
            best_seg_clusters = None
            for lm in range(u+1):
                #print("u: %d lm: %d"%(u, lm))
                seg_ll, seg_clusters = self.segment_u(u, lm)
                if seg_ll > best_seg_ll:
                    best_seg_ll = seg_ll
                    best_seg_clusters = seg_clusters
            self.best_segmentation[u] = best_seg_clusters
                
    def get_segmentation(self, doc_i):#TODO: needs to be redone, segmentation is now based on the SentenceCluster class
        '''
        Returns the final segmentation for a document.
        This is done by backtracking the best segmentations
        in a bottom-up fashion.
        :param doc_i: document index
        '''
        hyp_seg = []
        for u_cluster in self.best_segmentation[-1]:
            found_doc = False
            for u, doc_j in zip(u_cluster.u_list, u_cluster.doc_list):
                if doc_j == doc_i:
                    hyp_seg.append(0)
                    found_doc = True
            if found_doc:
                hyp_seg[-1] = 1
        #hyp_seg[0] = 0
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
    
class SentenceCluster(object):
    '''
    Class to keep track of a set of sentences (possibly from different documents)
    that belong to the same segment.
    '''
    def __init__(self, u, lm, docs, data):
        self.data = data
        self.u_list = []
        self.doc_list = []
        self.word_counts = np.zeros(self.data.W)
        seg_len = u-lm+1
        for doc_i in docs:
            if self.data.doc_len(doc_i) < u:
                u_end = self.data.doc_len(doc_i)
            else:
                u_end = u
            self.u_list += list(range(lm, u_end+1))
            self.doc_list += [doc_i]*seg_len
            self.word_counts += np.sum(self.data.doc_word_counts(doc_i)[lm:u_end+1], axis=0)
    
    def has_doc(self, doc_i):
        return doc_i in self.doc_list
    
    def add_sent(self, u, doc_i):
        self.u_list.append(u)
        self.doc_list.append(doc_i)
        self.word_counts += self.data.doc_word_counts(doc_i)[u]
        
    def get_word_counts(self):
        return self.word_counts
            
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
    sd_segs = []
    for doc in single_docs:
        data = Data(doc)
        vi_tt_model = TopicTrackingVIModel(beta, data)
        vi_tt_model.dp_segmentation()
        sd_segs.append(vi_tt_model.get_segmentation(0))
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
    
    md_segs = []
    for doc_i in range(vi_tt_model.data.n_docs):
        md_segs.append(vi_tt_model.get_segmentation(doc_i))
        
    gs_segs = []
    for gs_doc in doc_synth.get_single_docs():
        gs_segs.append(gs_doc.rho)
        
    for sd_seg, md_seg, gs_seg in zip(sd_segs, md_segs, gs_segs):
        print("GS: " + str(gs_seg.tolist()))
        print("SD: " + str(sd_seg))
        print("MD: " + str(md_seg)+"\n")
        
    print("Single:%s time: %f\nMulti: %s time: %f" % (str(single_doc_wd), sd_time, str(multi_doc_wd), md_time))
    
W = 300
beta = np.array([0.6]*W)
n_docs = 2
doc_len = 4
pi = 0.0
sent_len = 100
doc_synth = CVBSynDoc(beta, pi, sent_len, doc_len, n_docs)
data = Data(doc_synth)

#sigle_vs_md_eval(doc_synth, beta)

vi_tt_model = TopicTrackingVIModel(beta, data)
vi_tt_model.dp_segmentation()
print(eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth))
