'''
Created on Mar 2, 2018

@author: pjdrm
'''
from model.dp.segmentor import AbstractSegmentor
import numpy as np
import copy
import operator
from tqdm import trange

class MultiDocGreedySeg(AbstractSegmentor):
    
    def __init__(self, alpha, data, max_topics=None, seg_prior=None, use_prior=True):
        super(MultiDocGreedySeg, self).__init__(alpha,\
                                                data,\
                                                seg_prior=seg_prior,\
                                                use_prior=use_prior,\
                                                desc="greedy")
        self.max_topics = self.data.max_doc_len if max_topics is None else max_topics
        self.max_cache = 50
        
    def get_final_segmentation(self, doc_i):
        u_clusters = self.best_segmentation[-1][0][1]
        hyp_seg = self.get_segmentation(doc_i, u_clusters)
        return hyp_seg
    
    def mcmc_segmentation_step(self):
        '''
        Similar to vi_segmentation_step, but considers all
        valid u_clusters where a sentence can be inserted.
        Technically does not use VI. 
        '''
        with open(self.log_dir+"dp_tracker_"+self.desc+".txt", "a+") as f:
            t = trange(self.data.max_doc_len, desc='', leave=True)
            cached_segs = [(-np.inf, [])]
            for u in t:
                if u == 14:
                    a = 0
                for doc_i in range(self.data.n_docs):
                    t.set_description("(%d, %d)" % (u, doc_i))
                    if u > self.data.doc_len(doc_i)-1:
                        continue
                    
                    doc_i_segs = []
                    b = 0
                    for cached_seg_ll, cached_u_clusters in cached_segs:
                        if b == 403:
                            a = 0
                        b += 1
                        possible_clusters = self.get_valid_insert_clusters(doc_i, cached_u_clusters)
                        for k in range(self.max_topics):
                            current_u_clusters = copy.deepcopy(cached_u_clusters)
                            current_u_clusters = self.assign_target_k(u, u, doc_i, k, possible_clusters, current_u_clusters)
                            seg_ll = self.segmentation_ll(current_u_clusters)
                            if self.use_prior:
                                seg_ll += self.segmentation_log_prior(current_u_clusters)
                            doc_i_segs.append((seg_ll, current_u_clusters))
                            
                    doc_i_segs = sorted(doc_i_segs, key=operator.itemgetter(0), reverse=True)
                    no_dups_doc_i_segs = []
                    cached_segs = []
                    prev_seg_ll = -np.inf
                    for seg_result in doc_i_segs:
                        seg_ll = seg_result[0]
                        seg_clusters = seg_result[1]
                        if seg_ll != prev_seg_ll:
                            no_dups_doc_i_segs.append(seg_result)
                        prev_seg_ll = seg_ll
                    
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
                                cached_segs.append((seg_ll, seg_clusters))
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
                for cached_seg in cached_segs:
                    f.write("(%d)\tll: %.3f\n"%(u, cached_seg[0]))
                    for doc_i in range(self.data.n_docs):
                        f.write(str(self.get_segmentation(doc_i, cached_seg[1]))+" "
                                +str(self.get_seg_with_topics(doc_i, cached_seg[1]))+"\n")
                    f.write("\n")
                f.write("===============\n")
        cached_segs = sorted(cached_segs, key=operator.itemgetter(0), reverse=True)
        self.best_segmentation[-1] = cached_segs
        print("\nBest found ll: %f\nGS seg_ll: %f\n" % (cached_segs[0][0], self.segmentation_ll(self.data.get_rho_u_clusters())))
        
    def segment_docs(self):
        self.set_gl_data(self.data)
        self.mcmc_segmentation_step()