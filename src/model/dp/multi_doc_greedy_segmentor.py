'''
Created on Mar 2, 2018

@author: pjdrm
'''
from model.dp.segmentor import AbstractSegmentor, SEG_TT
import numpy as np
import copy
import operator
from tqdm import trange
import toyplot
import toyplot.pdf
import shutil
import os
import ray
import itertools
import multiprocessing

class MultiDocGreedySeg(AbstractSegmentor):
    
    def __init__(self, data, seg_config=None):
        super(MultiDocGreedySeg, self).__init__(data,\
                                                seg_config=seg_config,\
                                                desc="greedy")
        self.max_topics = self.data.max_doc_len if seg_config["max_topics"] is None else seg_config["max_topics"]
        self.max_cache = seg_config["max_cache"]
        self.phi_log_dir = seg_config["phi_log_dir"]
        self.run_parallel = seg_config["run_parallel"]
        self.check_cache_flag = seg_config["check_cache_flag"]
        self.log_flag = seg_config["log_flag"]
        if "u_order" not in seg_config or seg_config["u_order"] is None:
            self.u_order = None
        elif seg_config["u_order"] == "window":
            self.u_order = self.window_order(self.data.max_doc_len, self.data.n_docs, seg_config["window_size"])
        elif seg_config["u_order"] == "seg_dur_prior":
            self.u_order = self.seg_dur_prior_order(self.data.max_doc_len, self.data.n_docs, seg_config["seg_dur_prior"])
            
        if "cache_prune" not in seg_config or seg_config["cache_prune"] == "topn_cache_prune":
            self.cache_prune = self.topn_cache_prune
        elif seg_config["cache_prune"] == "k_slot_cache_prune":
            self.cache_prune = self.k_slot_cache_prune
            
        self.n_cpus = multiprocessing.cpu_count()
        shutil.rmtree(self.phi_log_dir) if os.path.isdir(self.phi_log_dir) else None
        os.makedirs(self.phi_log_dir)
        
    def log_phi_tt(self, u, cached_segs):
        '''
        for cached_seg in cached_segs:
            phi = cached_seg[2]
            all_phi.append(phi)
        '''
        phi = cached_segs[0][2]
        
        for t, phi_t in enumerate(phi):
            topic_plot_dict = {}
            for wi, word_prob in enumerate(phi_t):
                topic_plot_dict[wi] = word_prob
            sorted_prob = sorted(topic_plot_dict.items(), key=operator.itemgetter(1), reverse=True)
            sorted_word_probs = []
            words_sorted = []
            for wi, prob in sorted_prob:
                if prob <= 0.001:
                    break
                sorted_word_probs.append(prob)
                words_sorted.append(str(prob)[:4]+" "+self.data.doc_synth.inv_vocab[wi])
            
            if len(words_sorted) >= 70:
                h = 1200
                sorted_word_probs = sorted_word_probs[:70]
                words_sorted = words_sorted[:70]
            else:
                h = 800
            sorted_word_probs.reverse()
            words_sorted.reverse()
            if len(cached_segs[0][1]) == t:
                label = "Extra"
            else:
                label = "t " + str(cached_segs[0][1][t].k)
            title = "Topic " + label
            canvas = toyplot.Canvas(width=500, height=h)
            axes = canvas.cartesian(label=title, margin=100)
            axes.bars(sorted_word_probs, along='y')
            axes.y.ticks.locator = toyplot.locator.Explicit(labels=words_sorted)
            axes.y.ticks.labels.angle = -90
            
            toyplot.pdf.render(canvas, self.phi_log_dir+"/phi_u"+str(u)+"_t"+str(t)+".pdf")
        
        
    def get_final_segmentation(self, doc_i):
        u_clusters = self.best_segmentation[-1][0][1]
        hyp_seg = self.get_segmentation(doc_i, u_clusters)
        return hyp_seg
    
    def compute_seg_ll_seq(self, cached_segs, doc_i, u):
        '''
        Computes in sequentially the segmentation likelihood of assigning u to
        some topic k starting from a segmentation in cached_segs
        :param cached_segs: u_clusters for which we want to know the likelihood
        :param doc_i: document index from which u comes
        :param u: utterance index
        '''
        b = 0
        doc_i_segs = []
        for cached_seg in cached_segs:
            cached_u_clusters = cached_seg[1]
            phi_tt = cached_seg[2]
            if b == 23:
                a = 0
            b += 1
            possible_clusters = self.get_valid_insert_clusters(doc_i, cached_u_clusters)
            for k in range(self.max_topics):
                current_u_clusters = copy.deepcopy(cached_u_clusters)
                current_u_clusters = self.assign_target_k(u, u, doc_i, k, possible_clusters, current_u_clusters)
                phi_tt = None
                if self.seg_func_desc == SEG_TT:
                    seg_ll, phi_tt = self.segmentation_ll(current_u_clusters)
                else:
                    seg_ll = self.segmentation_ll(current_u_clusters)
                #This was here before but seems wrong, prior is already added in segmentation_ll (even in pre topic tracking version)
                #if self.use_dur_prior:
                #    seg_ll += self.segmentation_log_prior(current_u_clusters)
                doc_i_segs.append((seg_ll, current_u_clusters, phi_tt, k))
        return doc_i_segs
    
    def check_cache(self, doc_i, u, no_dups_doc_i_segs):
        cached_segs = []
        gave_warn = False
        cached_correct_seg = False
        found_correct_seg = False
        for i, seg_result in enumerate(no_dups_doc_i_segs):
            seg_ll = seg_result[0]
            seg_clusters = seg_result[1]
            
            if u > 1:
                n_correct_segs = 0
                for doc_j in range(0, doc_i+1):
                    rho_seg = self.get_segmentation(doc_j, seg_clusters)
                    rho_gs = list(self.data.docs_rho_gs[doc_j][:u+1])
                    rho_seg[-1] = 1
                    rho_gs[-1] = 1
                    if str(rho_seg) == str(rho_gs):
                        n_correct_segs += 1
                if n_correct_segs == doc_i+1: 
                    is_correct_seg = True
                    found_correct_seg = True
                else:
                    is_correct_seg = False
            else:
                is_correct_seg = True
                found_correct_seg = True
            
            is_cached = self.is_cached_seg(seg_ll, cached_segs)
            if not is_cached:
                if len(cached_segs) < self.max_cache:
                    if is_correct_seg:
                        cached_correct_seg = True
                    cached_segs.append(seg_result)
                    cached_segs = sorted(cached_segs, key=operator.itemgetter(0), reverse=True)
                    
                elif seg_ll > cached_segs[-1][0]:
                    if is_correct_seg:
                        cached_correct_seg = True
                    cached_segs[-1] = (seg_ll, seg_clusters)
                    cached_segs = sorted(cached_segs, key=operator.itemgetter(0), reverse=True)
                elif is_correct_seg and not gave_warn and not cached_correct_seg:
                    print("\nWARNING NEED CACHE LEN %d"%i)
                    gave_warn = True
        if not found_correct_seg:
            print("\nLOST CORRECT SEG u: %d"%u)
        return cached_segs
    
    def seg_dur_prior_order(self, max_u, n_docs, seg_dur_prior):
        doc_windows = []
        max_windows = -1
        for doc_i in range(n_docs):
            u = list(range(max_u))
            window = int(seg_dur_prior[doc_i][0])
            u_window = [u[i:i+window] for i in range(0, max_u, window)]
            if len(u_window) > max_windows:
                max_windows = len(u_window)
            doc_windows.append(u_window)
        
        u_order = []
        for window in range(max_windows):
            for doc_i in range(n_docs):
                if window >= len(doc_windows[doc_i]):
                    continue
                for u in doc_windows[doc_i][window]: 
                    u_order.append((u, doc_i))
                    
        return u_order
        
    def window_order(self, max_u, n_docs, window):
        u_order = []
        u = list(range(max_u))
        u_window = [u[i:i+window] for i in range(0, max_u, window)]
        
        for window in u_window:
            for doc_i in range(n_docs):
                for u in window:
                    u_order.append((u, doc_i))
        return u_order
    
    def window_order_v2(self, max_u, n_docs, u_w_size, doc_w_size=2):
        u_order = []
        u = list(range(max_u))
        doc_list = list(range(n_docs))
        u_windows = [u[i:i+u_w_size] for i in range(0, max_u, u_w_size)]
        doc_seqs = [doc_list[i:i+doc_w_size] for i in range(0, n_docs, doc_w_size)]
        for doc_seq in doc_seqs:
            for u_window in u_windows:
                for doc_i in doc_seq:
                    for u in u_window:
                        u_order.append((u, doc_i))
        return u_order
        
    def remove_seg_dups(self, doc_i_segs):
        '''
        Removes duplicate segmentations based on equal segmentation likehoods.
        Ensures that the cache does not end up with unecessary duplciates.
        :param doc_i_segs: list of tuples in the format (seg_ll, current_u_clusters, phi_tt, k)
        '''
        doc_i_segs = sorted(doc_i_segs, key=operator.itemgetter(0), reverse=True)
        no_dups_doc_i_segs = []
        prev_seg_ll = -np.inf
        for seg_result in doc_i_segs:
            seg_ll = seg_result[0]
            seg_clusters = seg_result[1]
            if seg_ll != prev_seg_ll:
                no_dups_doc_i_segs.append(seg_result)
            prev_seg_ll = seg_ll
        return no_dups_doc_i_segs
    
    def topn_cache_prune(self, cached_segs):
        return cached_segs[:self.max_cache]
        
    def k_slot_cache_prune(self, cached_segs):
        max_segs_k = int(self.max_cache/self.max_topics)
        b = 0
        doc_i_segs = []
        k_segs_dict = {}
        for k in range(self.max_topics):
            k_segs_dict[k] = []
            
        for cached_seg in cached_segs:
            k = cached_seg[3]
            k_segs_dict[k].append(cached_seg)
            
        for k in k_segs_dict:
            k_segs = k_segs_dict[k]
            k_segs = sorted(k_segs, key=operator.itemgetter(0), reverse=True)
            doc_i_segs += k_segs[:max_segs_k]
        return doc_i_segs
    
    def greedy_segmentation_step(self, u_order=None):
        '''
        Similar to vi_segmentation_step, but considers all
        valid u_clusters where a sentence can be inserted.
        Technically does not use VI. 
        '''
        if u_order == None:
            u_order = []
            for u in range(self.data.max_doc_len):
                for doc_i in range(self.data.n_docs):
                    u_order.append((u, doc_i))
                    
        with open(self.log_dir+"dp_tracker_"+self.desc+".txt", "a+") as f:
            t = trange(len(u_order), desc='', leave=True)
            cached_segs = [(-np.inf, [], None)]
            for i in t:
                u = u_order[i][0]
                doc_i = u_order[i][1]
                if u == 18:
                    a = 0

                t.set_description("(%d, %d)" % (u, doc_i))
                if u > self.data.doc_len(doc_i)-1:
                    continue
                
                if self.run_parallel:
                    n = int(self.max_cache/(self.n_cpus+1)) #TODO: adjust to number of CPUs
                    cached_segs_split = [cached_segs[i:i+n] for i in range(0, len(cached_segs), n)]
                    results = ray.get([compute_seg_ll_parallel.remote(self, cached_segs_job, doc_i, u) for cached_segs_job in cached_segs_split])
                    doc_i_segs = list(itertools.chain.from_iterable(results))
                else:
                    doc_i_segs = self.compute_seg_ll_seq(cached_segs, doc_i, u)
                        
                no_dups_doc_i_segs = self.remove_seg_dups(doc_i_segs)
                cached_segs = self.cache_prune(no_dups_doc_i_segs)
                
                if self.log_flag:
                    #The code below was assuming we iterated sequentially
                    '''
                    gs_seg = self.data.get_gs_u_clusters(u)
                    if self.seg_func_desc == SEG_TT:
                        gs_ll, gs_phi_tt = self.segmentation_ll(gs_seg)
                    else:
                        gs_ll = self.segmentation_ll(gs_seg)
                    f.write("(%d) gs_ll: %.3f\n\n"%(u, gs_ll))
                    '''
                                    
                    for i, cached_seg in enumerate(cached_segs):
                        f.write("(%d %d)\t%d\tll: %.3f\n"%(u, doc_i, i, cached_seg[0]))
                        for doc_j in range(self.data.n_docs):
                            f.write(str(self.get_segmentation(doc_j, cached_seg[1]))+" "
                                    +str(self.get_seg_with_topics(doc_j, cached_seg[1]))+"\n")
                        f.write("\n")
                    if self.seg_func_desc == SEG_TT:
                        self.log_phi_tt(u, cached_segs)
                    f.write("===============\n")
        cached_segs = sorted(cached_segs, key=operator.itemgetter(0), reverse=True)
        self.best_segmentation[-1] = cached_segs
        if self.seg_func_desc == SEG_TT:
            seg_ll_gs, phi_tt = self.segmentation_ll(self.data.get_rho_u_clusters())
        else:
            seg_ll_gs = self.segmentation_ll(self.data.get_rho_u_clusters())
        print("\nBest found ll: %f\nGS seg_ll: %f\n" % (cached_segs[0][0], seg_ll_gs))
        
    def segment_docs(self):
        self.set_gl_data(self.data)
        self.greedy_segmentation_step(self.u_order)
        
@ray.remote
def compute_seg_ll_parallel(segmentor, cached_segs, doc_i, u):
    '''
    Computes in parallel the segmentation likelihood of assigning u to
    some topic k starting from a segmetnation in cached_segs
    :param cached_segs: u_clusters for which we want to know the likelihood
    :param doc_i: document index from which u comes
    :param u: utterance index
    '''
    segmentor.set_gl_data(segmentor.data)
    b = 0
    doc_i_segs = []
    for cached_seg in cached_segs:
        cached_u_clusters = cached_seg[1]
        phi_tt = cached_seg[2]
        if b == 28:
            a = 0
        b += 1
        possible_clusters = segmentor.get_valid_insert_clusters(doc_i, cached_u_clusters)
        for k in range(segmentor.max_topics):
            current_u_clusters = copy.deepcopy(cached_u_clusters)
            current_u_clusters = segmentor.assign_target_k(u, u, doc_i, k, possible_clusters, current_u_clusters)
            phi_tt = None
            if segmentor.seg_func_desc == SEG_TT:
                seg_ll, phi_tt = segmentor.segmentation_ll(current_u_clusters)
            else:
                seg_ll = segmentor.segmentation_ll(current_u_clusters)
            #This was here before but seems wrong, prior is already added in segmentation_ll (even in pre topic tracking version)
            #if self.use_dur_prior:
            #    seg_ll += self.segmentation_log_prior(current_u_clusters)
            doc_i_segs.append((seg_ll, current_u_clusters, phi_tt, k))
    return doc_i_segs     
