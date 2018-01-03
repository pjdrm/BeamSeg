'''
Created on Nov 1, 2017

@author: pjdrm
'''
import numpy as np
import copy
from tqdm import trange
from dataset.synthetic_doc_cvb import CVBSynDoc, CVBSynDoc2
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
    
    def print_seg(self, u_clusters):
        print("==========================")
        for doc_i in range(self.data.n_docs):
            seg = self.get_segmentation(doc_i, u_clusters)
            print("Doc %d: %s" % (doc_i, str(seg)))
            
    def get_segmentation(self, doc_i, u_clusters):
        '''
        Returns the final segmentation for a document.
        This is done by backtracking the best segmentations
        in a bottom-up fashion.
        :param doc_i: document index
        '''
        hyp_seg = []
        for u_cluster in u_clusters:
            found_doc = False
            for u, doc_j in zip(u_cluster.u_list, u_cluster.doc_list):
                if doc_j == doc_i:
                    hyp_seg.append(0)
                    found_doc = True
            if found_doc:
                hyp_seg[-1] = 1
        return hyp_seg
    
    def get_all_segmentations(self):
        '''
        Returns a single vector with the final
        segmentation for all documents.
        '''
        all_segs = []
        for doc_i in range(self.data.n_docs):
            all_segs += self.get_segmentation(doc_i, self.best_segmentation[-1])
        return all_segs
            
    def get_last_cluster(self, doc_i, u_clusters):
        '''
        Returns the last cluster index where doc_i is present
        :param doc_i: document index
        :param u_clusters: list of sentence cluster corresponding to a segmentation
        '''
        found_doc = False
        for cluster_i, u_cluster in enumerate(u_clusters):
            if u_cluster.has_doc(doc_i):
                if not found_doc:
                    found_doc = True
            elif found_doc:
                return cluster_i-1
        return len(u_clusters)-1 #case where the last cluster was the last one in the list
    
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
    
    def fit_sentences(self, u_begin, u_end, docs, u_clusters):
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
                    u_cluster.add_sents(u_begin, u_end, doc)
                    break
    
    def new_seg_point(self, u_begin, u_end, doc_comb, u_clusters):
        '''
        Considers the segment u_begin to u_end as a new segmentation points for all
        documents in doc_comb. The sentences are added to the corresponding cluster
        (a new cluster is generated if necessary).
        :param u_begin: beginning sentence index
        :param u_end: end sentence index
        :param doc_comb: list of document indexes
        :param u_clusters: list of sentence cluster corresponding to a segmentation
        '''
        n_cluster = len(u_clusters)
        for doc_i in doc_comb:
            cluster_i = self.get_last_cluster(doc_i, u_clusters)
            if cluster_i+1 < n_cluster: #The language model corresponding to this cluster might already exists due to other documents having different segmentation at this stage
                u_clusters[cluster_i+1].add_sents(u_begin, u_end, doc_i)
            else:
                new_cluster = SentenceCluster(u_begin, u_end, [doc_i], self.data)
                u_clusters.append(new_cluster)
                n_cluster += 1
                
    def segment_u(self, u_begin, u_end):
        '''
        Estimates, for all documents, the best segmentation
        from u_end to u_begin (column index in the DP matrix).
        :param u_end: sentence index
        :param u_begin: language model index
        '''
        if u_begin == 0:#The first column corresponds to having all sentences from all docs in a single segment (there is only one language model)
            u_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), self.data)
            segmentation_ll = self.segment_ll(u_cluster.get_word_counts())
            return segmentation_ll, [u_cluster]
           
        best_seg_ll = -np.inf
        best_seg_clusters = None
        for doc_comb, other_docs in self.doc_combs_list:
            best_seg = copy.deepcopy(self.best_segmentation[u_begin-1])
            self.fit_sentences(u_begin, u_end, other_docs, best_seg) #Note that this changes best_seg
            self.new_seg_point(u_begin, u_end, doc_comb, best_seg) #Note that this changes best_seg
            segmentation_ll = 0.0
            for u_cluster in best_seg:
                segmentation_ll += self.segment_ll(u_cluster.get_word_counts())
            if segmentation_ll >= best_seg_ll:
                best_seg_ll = segmentation_ll
                best_seg_clusters = best_seg
        return best_seg_ll, best_seg_clusters
            
    def dp_segmentation(self):
        for u_end in range(self.data.max_doc_len):
            best_seg_ll = -np.inf
            best_seg_clusters = None
            for u_begin in range(u_end+1):
                seg_ll, seg_clusters = self.segment_u(u_begin, u_end)
                if seg_ll > best_seg_ll:
                    best_seg_ll = seg_ll
                    best_seg_clusters = seg_clusters
            #self.print_seg(best_seg_clusters)
            self.best_segmentation[u_end] = best_seg_clusters
        #print("==========================")
    
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
    def __init__(self, u_begin, u_end, docs, data):
        self.data = data
        self.u_list = []
        self.doc_list = []
        self.word_counts = np.zeros(self.data.W)
        for doc_i in docs:
            doc_i_len = self.data.doc_len(doc_i)
            #Accounting for documents with different lengths
            if u_begin > doc_i_len-1:
                continue
            
            if u_end > doc_i_len-1:
                u_end_true = doc_i_len-1#-1 (?)
            else:
                u_end_true = u_end
            seg_len = u_end_true-u_begin+1
            self.u_list += list(range(u_begin, u_end_true+1))
            self.doc_list += [doc_i]*seg_len
            self.word_counts += np.sum(self.data.doc_word_counts(doc_i)[u_begin:u_end_true+1], axis=0)
    
    def has_doc(self, doc_i):
        return doc_i in self.doc_list
    
    def add_sents(self, u_begin, u_end, doc_i):
        doc_i_len = self.data.doc_len(doc_i)
        #Accounting for documents with different lengths
        if u_begin > doc_i_len-1:
            return
        if u_end > doc_i_len-1:
            u_end = doc_i_len-1
            
        seg = list(range(u_begin, u_end+1))
        seg_len = u_end-u_begin+1
        self.u_list += seg
        self.doc_list += [doc_i]*seg_len
        self.word_counts += np.sum(self.data.doc_word_counts(doc_i)[u_begin:u_end+1], axis=0)
        
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
        sd_segs.append(vi_tt_model.get_segmentation(0, vi_tt_model.best_segmentation[-1]))
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
        md_segs.append(vi_tt_model.get_segmentation(doc_i, vi_tt_model.best_segmentation[-1]))
        
    gs_segs = []
    for gs_doc in doc_synth.get_single_docs():
        gs_segs.append(gs_doc.rho)
        
    for sd_seg, md_seg, gs_seg in zip(sd_segs, md_segs, gs_segs):
        print("GS: " + str(gs_seg.tolist()))
        print("SD: " + str(sd_seg))
        print("MD: " + str(md_seg)+"\n")
        
    print("Single:%s time: %f\nMulti: %s time: %f" % (str(single_doc_wd), sd_time, str(multi_doc_wd), md_time))
    
def md_eval(doc_synth, beta):
    vi_tt_model = TopicTrackingVIModel(beta, data)
    vi_tt_model.dp_segmentation()
    
    md_segs = []
    for doc_i in range(vi_tt_model.data.n_docs):
        md_segs.append(vi_tt_model.get_segmentation(doc_i, vi_tt_model.best_segmentation[-1]))
            
    gs_segs = []
    for gs_doc in doc_synth.get_single_docs():
        gs_segs.append(gs_doc.rho)
        
    for md_seg, gs_seg in zip(md_segs, gs_segs):
        print("GS: " + str(gs_seg.tolist()))
        print("MD: " + str(md_seg)+"\n")
        
    print(eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth))
    
W = 80
beta = np.array([0.3]*W)
n_docs = 2
doc_len = 40
pi = 0.4
sent_len = 10
#doc_synth = CVBSynDoc(beta, pi, sent_len, doc_len, n_docs)
doc_synth = CVBSynDoc2(beta, pi, sent_len, 3, n_docs)
data = Data(doc_synth)

#sigle_vs_md_eval(doc_synth, beta)
md_eval(doc_synth, beta)
#md_eval(doc_synth, beta)

