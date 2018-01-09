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
import toyplot
import toyplot.pdf

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
    
    def get_seg_diff(self, u_clusters):
        doc_segs_counts = {k:0 for k in range(self.data.n_docs)}
        for u_cluster in u_clusters:
            doc_set = set(u_cluster.doc_list)
            for doc_i in doc_set:
                doc_segs_counts[doc_i] += 1
        
        min_segs = np.inf
        max_segs = -np.inf
        for doc_i in doc_segs_counts:
            n_segs = doc_segs_counts[doc_i]
            if n_segs <= min_segs:
                min_segs = n_segs
            if n_segs >= max_segs:
                max_segs = n_segs
        return max_segs-min_segs
    
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
    
    def segmentation_ll(self, u_clusters):
        '''
        Returns the log likelihood of the segmentation of all documents.
        :param u_clusters: list of SentenceCluster corresponding to the best segmentation up to u-1
        '''
        segmentation_ll = 0.0
        for u_cluster in u_clusters:
            segmentation_ll += self.segment_ll(u_cluster.get_word_counts())
        return segmentation_ll
    
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
            #if self.valid_segmentation(best_seg):
            segmentation_ll = self.segmentation_ll(best_seg)
            if segmentation_ll >= best_seg_ll:
                best_seg_ll = segmentation_ll
                best_seg_clusters = best_seg
        return best_seg_ll, best_seg_clusters
    
    def segment_u_fast(self, u_begin, u_end):
        '''
        Estimates, for all documents, the best segmentation
        from u_end to u_begin (column index in the DP matrix).
        Implements a faster version that does not explroe all self.doc_combs_list.
        :param u_end: sentence index
        :param u_begin: language model index
        '''
        if u_begin == 0:#The first column corresponds to having all sentences from all docs in a single segment (there is only one language model)
            u_cluster = SentenceCluster(u_begin, u_end, list(range(self.data.n_docs)), self.data)
            segmentation_ll = self.segment_ll(u_cluster.get_word_counts())
            return segmentation_ll, [u_cluster]
           
        final_seg_clusters = copy.deepcopy(self.best_segmentation[u_begin-1])
        prev_doc_comb = set(range(self.data.n_docs))
        self.new_seg_point(u_begin, u_end, prev_doc_comb, final_seg_clusters)
        final_seg_ll = self.segmentation_ll(final_seg_clusters)
        all_docs = set(range(self.data.n_docs))
            
        while 1:
            doc_combs = combinations(prev_doc_comb, len(prev_doc_comb)-1)
            best_seg_ll = -np.inf
            best_doc_comb = None
            for doc_comb in doc_combs:
                if len(doc_comb) == 0:
                    break #Means we have reached a point where we dont segment any of the documents.
                doc_comb = set(doc_comb)
                best_seg = copy.deepcopy(self.best_segmentation[u_begin-1])
                other_docs = all_docs-doc_comb
                self.fit_sentences(u_begin, u_end, other_docs, best_seg) #Note that this changes best_seg
                self.new_seg_point(u_begin, u_end, doc_comb, best_seg) #Note that this changes best_seg
                segmentation_ll = self.segmentation_ll(best_seg)
                if segmentation_ll >= best_seg_ll:
                    best_seg_ll = segmentation_ll
                    best_seg_clusters = best_seg
                    best_doc_comb = doc_comb
                    
            if best_seg_ll > final_seg_ll:
                prev_doc_comb = best_doc_comb
                final_seg_ll = best_seg_ll
                final_seg_clusters = best_seg_clusters
            else:
                break
        
        return final_seg_ll, final_seg_clusters
            
    def dp_segmentation(self, fast_seg=False):
        if fast_seg:
            seg_func = self.segment_u_fast
        else:
            seg_func = self.segment_u
        
        t = trange(self.data.max_doc_len, desc='', leave=True)
        for u_end in t:
            best_seg_ll = -np.inf
            best_seg_clusters = None
            for u_begin in range(u_end+1):
                seg_ll, seg_clusters = seg_func(u_begin, u_end)
                if seg_ll > best_seg_ll:
                        best_seg_ll = seg_ll
                        best_seg_clusters = seg_clusters
                t.set_description("Matrix: (%d,%d)" % (u_end, u_begin))
            self.best_segmentation[u_end] = best_seg_clusters
            #self.print_seg(best_seg_clusters)
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
                u_end_true = doc_i_len-1
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

def print_segmentation(seg_desc, seg_results):
    for seg in seg_results:
        print("%s: %s" % (seg_desc, str(seg)))
        
def sigle_vs_md_eval(doc_synth, beta, md_all_combs=True, md_fast=True, print_flag=False):
    '''
    Print the WD results when segmenting single documents
    and all of them simultaneously (multi-doc model)
    :param doc_synth: collection of synthetic documents
    :param beta: beta prior vector
    :param print_flag: boolean to print or not the segmentation results
    '''
    single_docs = doc_synth.get_single_docs()
    single_doc_wd = []
    time_wd_results = []
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
    #single_doc_wd = ['%.3f' % wd for wd in single_doc_wd]
    time_wd_results.append(("SD", sd_time, ['%.3f' % wd for wd in single_doc_wd]))
    
    data = Data(doc_synth)
    if md_all_combs:
        vi_tt_model = TopicTrackingVIModel(beta, data)
        start = time.time()
        vi_tt_model.dp_segmentation()
        end = time.time()
        md_time = (end - start)
        multi_doc_wd = eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth)
        #multi_doc_wd = ['%.3f' % wd for wd in multi_doc_wd]
        time_wd_results.append(("MD", md_time, ['%.3f' % wd for wd in multi_doc_wd]))
        
        md_segs = []
        for doc_i in range(vi_tt_model.data.n_docs):
            md_segs.append(vi_tt_model.get_segmentation(doc_i, vi_tt_model.best_segmentation[-1]))
    
    if md_fast: 
        md_fast_segs = []
        vi_tt_model = TopicTrackingVIModel(beta, data)
        start = time.time()
        vi_tt_model.dp_segmentation(fast_seg=True)
        end = time.time()
        md_fast_time = (end - start)
        multi_fast_doc_wd = eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth)
        #multi_fast_doc_wd = ['%.3f' % wd for wd in multi_fast_doc_wd]
        time_wd_results.append(("MF", md_fast_time, ['%.3f' % wd for wd in multi_fast_doc_wd]))
        for doc_i in range(vi_tt_model.data.n_docs):
            md_fast_segs.append(vi_tt_model.get_segmentation(doc_i, vi_tt_model.best_segmentation[-1]))
        
    if print_flag:
        gs_segs = []
        for gs_doc in doc_synth.get_single_docs():
            gs_segs.append(gs_doc.rho.tolist())
            
        print_segmentation("GS", gs_segs)
        print_segmentation("SD", sd_segs)
        if md_all_combs and md_fast:
            for doc_i in range(doc_synth.n_docs):
                print("%s: %s" % ("GS", str(gs_segs[doc_i])))
                print("%s: %s" % ("SD", str(sd_segs[doc_i])))
                print("%s: %s" % ("MD", str(md_segs[doc_i])))
                print("%s: %s\n" % ("MF", str(md_fast_segs[doc_i])))
        elif md_all_combs:
            for doc_i in range(doc_synth.n_docs):
                print("%s: %s" % ("GS", str(gs_segs[doc_i])))
                print("%s: %s" % ("SD", str(sd_segs[doc_i])))
                print("%s: %s\n" % ("MD", str(md_segs[doc_i])))
        else:
            for doc_i in range(doc_synth.n_docs):
                print("%s: %s" % ("GS", str(gs_segs[doc_i])))
                print("%s: %s" % ("SD", str(sd_segs[doc_i])))
                print("%s: %s\n" % ("MF", str(md_fast_segs[doc_i])))
    else:
        return single_doc_wd, multi_fast_doc_wd
      
    for time_res in time_wd_results:  
        print("%s: %s time: %f" % (time_res[0], time_res[2], time_res[1]))
    
def md_eval(doc_synth, beta):
    vi_tt_model = TopicTrackingVIModel(beta, data)
    start = time.time()
    vi_tt_model.dp_segmentation(fast_seg=True)
    end = time.time()
    seg_time = (end - start)
    md_segs = []
    for doc_i in range(vi_tt_model.data.n_docs):
        md_segs.append(vi_tt_model.get_segmentation(doc_i, vi_tt_model.best_segmentation[-1]))
            
    gs_segs = []
    for gs_doc in doc_synth.get_single_docs():
        gs_segs.append(gs_doc.rho)
        
    for md_seg, gs_seg in zip(md_segs, gs_segs):
        print("GS: " + str(gs_seg.tolist()))
        print("MD: " + str(md_seg)+"\n")
    print("Time: %f" % seg_time)
        
    print(eval_tools.wd_evaluator(vi_tt_model.get_all_segmentations(), doc_synth))
    
def merge_docs(target_docs):
    target_docs_copy = copy.deepcopy(target_docs)
    merged_doc = target_docs_copy[0]
    all_rho = []
    all_docs_index = []
    all_U_W_counts = None
    carry_index = 0
    for doc_synth in target_docs:
        new_index = (np.array(doc_synth.docs_index)+carry_index).tolist()
        carry_index = new_index[-1]
        all_docs_index += new_index
        doc_synth.rho[-1] = 1
        all_rho += doc_synth.rho.tolist()
        if all_U_W_counts is None:
            all_U_W_counts = doc_synth.U_W_counts
        else:
            all_U_W_counts = np.vstack((all_U_W_counts, doc_synth.U_W_counts))
            
    all_rho[-1] = 0
    
    merged_doc.n_docs = len(target_docs)
    merged_doc.rho = all_rho
    merged_doc.docs_index = all_docs_index
    merged_doc.U_W_counts = all_U_W_counts
    merged_doc.isMD = True
    
    return merged_doc
    
def incremental_eval(doc_synth, beta):
    def grouped_bars(axes, data, group_names, group_width=None):
        if group_width is None:
            group_width=1 - 1.0 / (data.shape[1] + 1)
            
        group_left_edges = np.arange(data.shape[0], dtype="float") - (group_width / 2.0)
        bar_width = group_width / data.shape[1]
        
        marks = []
        axes.x.ticks.locator = toyplot.locator.Explicit(labels=group_names)
        for index, series in enumerate(data.T):
            left_edges = group_left_edges + (index * bar_width)
            right_edges = group_left_edges + ((index + 1) * bar_width)
            marks.append(axes.bars(left_edges, right_edges, series, opacity=0.5))
            
        return marks

    single_docs = doc_synth.get_single_docs()
    all_sd_results = []
    all_md_results = []
    for i in range(1, doc_synth.n_docs+1):
        target_docs = single_docs[:i]
        merged_doc_synth = merge_docs(target_docs)
        sd_results, md_results = sigle_vs_md_eval(merged_doc_synth, beta, md_all_combs=False)
        all_sd_results.append(sd_results)
        all_md_results.append(md_results)
        
    final_results = []
    for sd_wds, mf_wds in zip(all_sd_results, all_md_results):
        n_ties = 0.0
        n_mf_win = 0.0
        n_mf_lost = 0.0
        for sd_wd, mf_wd in zip(sd_wds, mf_wds):
            if sd_wd == mf_wd:
                n_ties += 1.0
            elif mf_wd > sd_wd:
                n_mf_win += 1.0
            else:
                n_mf_lost += 1.0
        n_total = n_ties+n_mf_win+n_mf_lost
        n_ties_percent = n_ties/n_total
        n_mf_win_percent = n_mf_win/n_total
        n_mf_lost_percent = n_mf_lost/n_total
        final_results.append(np.array([n_ties_percent, n_mf_win_percent, n_mf_lost_percent])*100.0)
        
    group_names = list(range(1, doc_synth.n_docs+1))
    canvas = toyplot.Canvas(width=300, height=300)
    axes = canvas.cartesian()
    axes.x.label.text = "#Docs"
    axes.y.label.text = "Percentage"
    marks = grouped_bars(axes, np.array(final_results), group_names)
    canvas.legend([
    ("Tie", marks[0]),
    ("Win", marks[1]),
    ("Lose", marks[2])
    ],
    corner=("top-right", 0, 100, 50),
    );
    toyplot.pdf.render(canvas, "incremental_eval_results.pdf")
    
    
    
W = 300
beta = np.array([0.3]*W)
n_docs = 3
doc_len = 20
pi = 0.1
sent_len = 10
#doc_synth = CVBSynDoc(beta, pi, sent_len, doc_len, n_docs)
n_seg = 3
doc_synth = CVBSynDoc2(beta, pi, sent_len, n_seg, n_docs)
data = Data(doc_synth)

#incremental_eval(doc_synth, beta)
sigle_vs_md_eval(doc_synth, beta, md_all_combs=True, md_fast=True, print_flag=True)
#md_eval(doc_synth, beta)
