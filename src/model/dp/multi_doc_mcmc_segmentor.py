'''
Created on Apr 4, 2018

@author: pjdrm
'''
from model.dp.segmentor import AbstractSegmentor, SentenceCluster
import numpy as np
import copy
import operator
from tqdm import trange

class MultiDocMCMCSeg(AbstractSegmentor):
    
    def __init__(self, beta, data, max_topics=None, seg_dur=10.0, std=3.0, use_prior=True):
        super(MultiDocMCMCSeg, self).__init__(beta,\
                                                data,\
                                                seg_dur=seg_dur,\
                                                std=std,\
                                                use_prior=use_prior,\
                                                desc="mcmc")
        self.max_topics = self.data.max_doc_len if max_topics is None else max_topics
    
    def rnd_init_seg(self):
        pi = 1.0/self.seg_dur
        initial_u_clusters = []
        for doc_i in range(self.data.n_docs):
            doc_i_rho = []
            n_segs = 0
            for u in range(self.data.doc_len(doc_i)-1):
                u_rho = np.random.binomial(1, pi)
                doc_i_rho.append(u_rho)
                if u_rho == 1:
                    n_segs += 1
                    if n_segs == self.max_topics:
                        break
            if len(doc_i_rho)+1 != self.data.doc_len(doc_i):
                n_segs -= 1
                doc_i_rho[-1] = 0
                for u in range(self.data.doc_len(doc_i)-len(doc_i_rho)-1):
                    doc_i_rho.append(0)
            doc_i_rho.append(1)
            n_segs += 1
            possible_topics = list(range(self.max_topics))
            doc_i_topic_seq = []
            for i in range(n_segs):
                possible_topics_l = len(possible_topics)
                draw = np.random.multinomial(1, [1.0/possible_topics_l]*possible_topics_l)
                topic_index = np.nonzero(draw)[0][0] #w is a vocabulary index
                topic = possible_topics.pop(topic_index)
                doc_i_topic_seq.append(topic)
            k_index = 0
            for u, u_rho in enumerate(doc_i_rho):
                k = doc_i_topic_seq[k_index]
                u_k_cluster = self.get_k_cluster(k, initial_u_clusters)
                if u_k_cluster is not None:
                    u_k_cluster.add_sents(u, u, doc_i)
                else:
                    new_u_cluster = SentenceCluster(u, u, [doc_i], k)
                    initial_u_clusters.append(new_u_cluster)
                if u_rho == 1:
                    k_index += 1
        return initial_u_clusters
                
    def get_final_segmentation(self, doc_i):
        u_clusters = self.best_segmentation[-1][0][1]
        hyp_seg = self.get_segmentation(doc_i, u_clusters)
        return hyp_seg
    
    def test_splits(self, possible_k, doc_i, u_begin, u_end, best_seg_ll, u_clusters):
        best_k = None
        best_u_clusters = None
        found_better_seg = False
        for k in possible_k:
            current_u_clusters = copy.deepcopy(u_clusters)
            u_k_cluster = self.get_k_cluster(k, current_u_clusters)
            if u_k_cluster is not None:
                u_k_cluster.add_sents(u_begin, u_end, doc_i)
            else:
                new_cluster = SentenceCluster(u_begin, u_end, [doc_i], k)
                current_u_clusters.append(new_cluster)
                
            seg_ll = self.segmentation_ll(current_u_clusters)
            if self.use_prior:
                seg_ll += self.segmentation_log_prior(current_u_clusters)
                
            if seg_ll > best_seg_ll:
                found_better_seg = True
                best_k = k
                best_seg_ll = seg_ll
                best_u_clusters = current_u_clusters
                
        return found_better_seg, best_k, best_seg_ll, best_u_clusters
        
    def mcmc_segmentation_step(self, best_u_clusters):
        doc_i_rho = [self.get_final_segmentation(doc_i) for doc_i in range(self.data.n_docs)]
        best_seg_ll = self.segmentation_ll(best_u_clusters)
        if self.use_prior:
            best_seg_ll += self.segmentation_log_prior(best_u_clusters)
                            
        with open(self.log_dir+"dp_tracker_"+self.desc+".txt", "a+") as f:
            t = trange(self.data.max_doc_len, desc='', leave=True)
            for u in t:
                if u == 6:
                    a = 3
                for doc_i in range(self.data.n_docs):
                    t.set_description("(%d, %d)" % (u, doc_i))
                    if u > self.data.doc_len(doc_i)-1:
                        continue
                    
                    current_k, u_begin, u_end = self.get_u_segment(doc_i, u, best_u_clusters)
                    rho_u = doc_i_rho[doc_i][u]
                    if rho_u == 0:
                        #Split case
                        possible_k = list(range(self.max_topics))
                        for doc_i_k in self.get_doc_i_clusters(doc_i, best_u_clusters):
                            if doc_i_k in possible_k:
                                possible_k.remove(doc_i_k)
                                
                        #Split case 1: keep topic on first u_cluster
                        u_clusters_minus_next = copy.deepcopy(best_u_clusters)
                        current_u_cluster = self.get_k_cluster(current_k, u_clusters_minus_next)
                        current_u_cluster.remove_seg(doc_i, u+1, u_end)
                        if len(current_u_cluster.get_docs()) == 0:
                            u_clusters_minus_next.remove(current_u_cluster)
                        
                        found_better_seg,\
                        split_k,\
                        split_seg_ll,\
                        split_u_clusters = self.test_splits(possible_k,\
                                                           doc_i,\
                                                           u+1,\
                                                           u_end,\
                                                           best_seg_ll,\
                                                           u_clusters_minus_next)
                        
                        if found_better_seg:
                            best_seg_ll = split_seg_ll
                            best_u_clusters = split_u_clusters
                            doc_i_rho[doc_i] = self.get_segmentation(doc_i, best_u_clusters)
                            
                        #Split case 2: change topic on first u_cluster
                        u_clusters_minus_current = copy.deepcopy(best_u_clusters)
                        current_u_cluster = self.get_k_cluster(current_k, u_clusters_minus_current)
                        current_u_cluster.remove_seg(doc_i, u_begin, u)
                        if len(current_u_cluster.get_docs()) == 0:
                            u_clusters_minus_current.remove(current_u_cluster)
                            
                        found_better_seg,\
                        split_k,\
                        split_seg_ll,\
                        split_u_clusters = self.test_splits(possible_k,\
                                                           doc_i,\
                                                           u_begin,\
                                                           u_end,\
                                                           best_seg_ll,\
                                                           u_clusters_minus_current)
                        if found_better_seg:
                            current_k = split_k
                            best_seg_ll = split_seg_ll
                            best_u_clusters = split_u_clusters
                            doc_i_rho[doc_i] = self.get_segmentation(doc_i, best_u_clusters)
                    else:
                        #Merge case
                        if u == self.data.doc_len(doc_i)-1:
                            continue #this is the last sentence, cant perform merge
                        
                        next_u_cluster = self.get_next_cluster(current_k, doc_i, best_u_clusters)
                        u_begin_next_c, u_end_next_c = next_u_cluster.get_segment(doc_i)
                        u_clusters_minus_merge = copy.deepcopy(best_u_clusters)
                        uc1 = self.get_k_cluster(current_k, u_clusters_minus_merge)
                        uc1.remove_seg(doc_i, u_begin, u_end)
                        if len(uc1.get_docs()) == 0:
                            u_clusters_minus_merge.remove(uc1)
                        uc2 = self.get_k_cluster(next_u_cluster.k, u_clusters_minus_merge)
                        uc2.remove_seg(doc_i, u_begin_next_c, u_end_next_c)
                        if len(uc2.get_docs()) == 0:
                            u_clusters_minus_merge.remove(uc2)
                            
                        for k in range(self.max_topics):
                            current_u_clusters = copy.deepcopy(u_clusters_minus_merge)
                            u_k_cluster = self.get_k_cluster(k, current_u_clusters)
                            if k != current_k and\
                               k != next_u_cluster.k and\
                               u_k_cluster is not None\
                               and u_k_cluster.has_doc(doc_i):
                                continue #Means some other segments has this topics and we cant repeat it
                            
                            if u_k_cluster is not None:
                                u_k_cluster.add_sents(u_begin, u_end_next_c, doc_i)
                            else:
                                new_u_cluster = SentenceCluster(u_begin, u_end_next_c, [doc_i], k)
                                current_u_clusters.append(new_u_cluster)
                        
                            seg_ll = self.segmentation_ll(current_u_clusters)
                            if self.use_prior:
                                seg_ll += self.segmentation_log_prior(current_u_clusters)
                                
                            if seg_ll > best_seg_ll:
                                best_seg_ll = seg_ll
                                best_u_clusters = current_u_clusters
                                doc_i_rho[doc_i] = self.get_segmentation(doc_i, best_u_clusters)
                    
            f.write("ll: %.3f\n"%best_seg_ll)
            for doc_i in range(self.data.n_docs):
                f.write("GS: "+str(self.data.docs_rho_gs[doc_i].tolist())+"\n"+\
                        "HYP "+str(self.get_segmentation(doc_i, best_u_clusters))+"\n"+\
                        "K:  "+str(self.get_seg_with_topics(doc_i, best_u_clusters))+"\n\n")
            f.write("===============\n")
        self.best_segmentation[-1] = [(best_seg_ll, best_u_clusters)]
        return best_u_clusters
        #print("\nBest found ll: %f\nGS seg_ll: %f\n" % (cached_segs[0][0], self.segmentation_ll(self.data.get_rho_u_clusters())))
        
    def segment_docs(self):
        self.set_gl_data(self.data)
        u_clusters = self.rnd_init_seg()
        self.best_segmentation[-1] = [(-np.inf, u_clusters)]
        for i in range(10):
            if i == 5:
                a = 0
            u_clusters = self.mcmc_segmentation_step(u_clusters)
        print("GS ll: %.3f" % self.segmentation_ll(self.data.get_rho_u_clusters()))
        
            
            
            
            