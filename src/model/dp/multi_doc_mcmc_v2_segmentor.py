'''
Created on Apr 4, 2018

@author: pjdrm
'''
from model.dp.segmentor import AbstractSegmentor, SentenceCluster
import numpy as np
import copy
import operator
from tqdm import trange

class MultiDocMCMCSegV2(AbstractSegmentor):
    
    def __init__(self, alpha, data, max_topics=None, seg_dur=10.0, std=3.0, use_prior=True):
        super(MultiDocMCMCSegV2, self).__init__(alpha,\
                                                data,\
                                                seg_dur=seg_dur,\
                                                std=std,\
                                                use_prior=use_prior,\
                                                desc="mcmc_v2")
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
    
    def test_split(self, k, doc_i, u_begin, u_end, u_clusters):
        u_k_cluster = self.get_k_cluster(k, u_clusters)
        if u_k_cluster is not None:
            u_k_cluster.add_sents(u_begin, u_end, doc_i)
        else:
            new_cluster = SentenceCluster(u_begin, u_end, [doc_i], k)
            u_clusters.append(new_cluster)
            
        seg_ll = self.segmentation_ll(u_clusters)
        if self.use_prior:
            seg_ll += self.segmentation_log_prior(u_clusters)
            
        return seg_ll, u_clusters
        
    def accept_move(self, move_u_clusters, move_seg_ll, best_seg_ll):
        uniform_draw = np.random.uniform(0,1)
        ratio = move_seg_ll/best_seg_ll
        if uniform_draw < ratio:
            return True
        else:
            return False
        
    def mcmc_segmentation_step(self, best_u_clusters):
        doc_i_rho = [self.get_final_segmentation(doc_i) for doc_i in range(self.data.n_docs)]
        best_seg_ll = self.segmentation_ll(best_u_clusters)
        if self.use_prior:
            best_seg_ll += self.segmentation_log_prior(best_u_clusters)
                            
        with open(self.log_dir+"dp_tracker_"+self.desc+".txt", "a+") as f:
            draw = np.random.multinomial(1, [1.0/self.data.total_sents]*self.data.total_sents)
            u_move = np.nonzero(draw)[0][0]
            doc_i_move = self.data.get_doc_i(u_move)
            if doc_i_move > 0:
                u_move = u_move-self.data.docs_index[doc_i_move-1]
            
            current_k, u_begin, u_end = self.get_u_segment(doc_i_move, u_move, best_u_clusters)
            rho_u = doc_i_rho[doc_i_move][u_move]
            
            if rho_u == 0:
                u_clusters_split_test = copy.deepcopy(best_u_clusters)
                
                possible_k = list(range(self.max_topics))
                for doc_i_k in self.get_doc_i_clusters(doc_i_move, u_clusters_split_test):
                    if doc_i_k in possible_k:
                        possible_k.remove(doc_i_k)
                possible_k.append(current_k) #Adding current_k because if I choose it means I am going to change the next cluster topic
                draw = np.random.multinomial(1, [1.0/len(possible_k)]*len(possible_k))
                k_move = possible_k[np.nonzero(draw)[0][0]]
                current_u_cluster = self.get_k_cluster(current_k, u_clusters_split_test)
                if current_k == k_move:
                    #Split case 1: same topic and change of the next segment
                    if len(possible_k) == 1 and possible_k[0] == current_k:
                        #Case where we wanted to change the topic but no others are available
                        return best_u_clusters
                    current_u_cluster.remove_seg(doc_i_move, u_move+1, u_end)
                    possible_k.remove(current_k)
                    draw = np.random.multinomial(1, [1.0/len(possible_k)]*len(possible_k))
                    k_move = possible_k[np.nonzero(draw)[0][0]]
                    u_begin_split = u_move+1
                    u_end_split = u_end
                    
                else:
                    #Split case 2: change topic of current segment
                    current_u_cluster.remove_seg(doc_i_move, u_begin, u_move)
                    u_begin_split = u_begin
                    u_end_split = u_move
                
                move_seg_ll,\
                move_u_clusters = self.test_split(k_move,\
                                                   doc_i_move,\
                                                   u_begin_split,\
                                                   u_end_split,\
                                                   u_clusters_split_test)
            else:
                #Merge case
                if u_move == self.data.doc_len(doc_i_move)-1:
                    return best_u_clusters#this is the last sentence, cant perform merge
                
                u_clusters_merge = copy.deepcopy(best_u_clusters)
                possible_k = list(range(self.max_topics))
                for doc_i_k in self.get_doc_i_clusters(doc_i_move, u_clusters_merge):
                    if doc_i_k in possible_k:
                        possible_k.remove(doc_i_k)
                        
                next_u_cluster = self.get_next_cluster(current_k, doc_i_move, best_u_clusters)
                u_begin_next_c, u_end_next_c = next_u_cluster.get_segment(doc_i_move)
                uc1 = self.get_k_cluster(current_k, u_clusters_merge)
                uc1.remove_seg(doc_i_move, u_begin, u_end)
                if len(uc1.get_docs()) == 0:
                    u_clusters_merge.remove(uc1)
                uc2 = self.get_k_cluster(next_u_cluster.k, u_clusters_merge)
                uc2.remove_seg(doc_i_move, u_begin_next_c, u_end_next_c)
                if len(uc2.get_docs()) == 0:
                    u_clusters_merge.remove(uc2)
                    
                possible_k.append(current_k)
                possible_k.append(next_u_cluster.k)
                draw = np.random.multinomial(1, [1.0/len(possible_k)]*len(possible_k))
                k_move = possible_k[np.nonzero(draw)[0][0]]
                
                u_k_cluster = self.get_k_cluster(k_move, u_clusters_merge)
                            
                if u_k_cluster is not None:
                    u_k_cluster.add_sents(u_begin, u_end_next_c, doc_i_move)
                else:
                    new_u_cluster = SentenceCluster(u_begin, u_end_next_c, [doc_i_move], k_move)
                    u_clusters_merge.append(new_u_cluster)
            
                move_u_clusters = u_clusters_merge
                move_seg_ll = self.segmentation_ll(u_clusters_merge)
                if self.use_prior:
                    move_seg_ll += self.segmentation_log_prior(u_clusters_merge)
                    
            accept = self.accept_move(move_u_clusters, move_seg_ll, best_seg_ll)
            if accept:
                best_u_clusters = move_u_clusters
                best_seg_ll = move_seg_ll
                
                    
            f.write("ll: %.3f\n"%best_seg_ll)
            for doc_i in range(self.data.n_docs):
                f.write("GS: "+str(self.data.docs_rho_gs[doc_i].tolist())+"\n"+\
                        "HYP "+str(self.get_segmentation(doc_i, best_u_clusters))+"\n"+\
                        "K:  "+str(self.get_seg_with_topics(doc_i, best_u_clusters))+"\n\n")
            f.write("===============\n")
        self.best_segmentation[-1] = [(best_seg_ll, best_u_clusters)]
        return best_u_clusters
        #print("\nBest found ll: %f\nGS move_seg_ll: %f\n" % (cached_segs[0][0], self.segmentation_ll(self.data.get_rho_u_clusters())))
        
    def segment_docs(self):
        self.set_gl_data(self.data)
        u_clusters = self.rnd_init_seg()
        self.best_segmentation[-1] = [(-np.inf, u_clusters)]
        t = trange(60000, desc='', leave=True)
        for i in t:
            t.set_description("Iter %d" % i)
            if i == 4:
                a = 0
            u_clusters = self.mcmc_segmentation_step(u_clusters)
        print("GS ll: %.3f" % self.segmentation_ll(self.data.get_rho_u_clusters()))
        