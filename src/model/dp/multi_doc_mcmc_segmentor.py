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
    
    def __init__(self, data, seg_config=None):
        super(MultiDocMCMCSegV2, self).__init__(data,\
                                                seg_config=seg_config,\
                                                desc="mcmc_v2")
        self.max_topics = self.data.max_doc_len if seg_config["max_topics"] is None else seg_config["max_topics"]
        self.total_accepts = 0
        self.n_samples = {}
        for doc_i in range(self.data.n_docs):
            self.n_samples[doc_i] = np.zeros(self.data.doc_len(doc_i))
    
    def rnd_init_seg(self):
        pi = 1.0/np.average(self.seg_dur_prior)
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
            
        seg_ll = self.segmentation_ll(u_clusters)[0]
        return seg_ll, u_clusters
        
    def accept_move(self, move_seg_ll, best_seg_ll):
        #TODO: do a version similar to the split-merge sampler.
        #The problem now is that I should have split all document in
        #the same two topics. But for now the split is in random possible
        #topics for each documents. This is to not have the same topic
        #ordering in all documents
        uniform_draw = np.random.uniform(0,1)
        ratio = move_seg_ll/best_seg_ll
        if uniform_draw < ratio:
            return True
        else:
            return False
        
    def sample_u(self, doc_i, k, u_clusters):
        u_k_cluster = self.get_k_cluster(k, u_clusters)
        u_begin, u_end = u_k_cluster.get_segment(doc_i)
        seg_len = u_end-u_begin+1
        draw = np.random.multinomial(1, [1.0/seg_len]*seg_len)
        u_sample = u_begin+np.nonzero(draw)[0][0]
        return u_sample
    
    def sample_rnd_k(self, u_clusters):
        draw = np.random.multinomial(1, [1.0/self.data.total_sents]*self.data.total_sents)
        u_move = np.nonzero(draw)[0][0]
        doc_i_move = self.data.get_doc_i(u_move)
        if doc_i_move > 0:
            u_move = u_move-self.data.docs_index[doc_i_move-1]
        k1, u_begin, u_end = self.get_u_segment(doc_i_move, u_move, u_clusters)
        
        u_move = np.nonzero(draw)[0][0]
        doc_i_move = self.data.get_doc_i(u_move)
        if doc_i_move > 0:
            u_move = u_move-self.data.docs_index[doc_i_move-1]
        k2, u_begin, u_end = self.get_u_segment(doc_i_move, u_move, u_clusters)
        return k1, k2
        
    def mcmc_segmentation_step(self, best_u_clusters):
        doc_i_rho = [self.get_final_segmentation(doc_i) for doc_i in range(self.data.n_docs)]
        best_seg_ll = self.segmentation_ll(best_u_clusters)[0]
                            
        with open(self.log_dir+"dp_tracker_"+self.desc+".txt", "a+") as f:
            k1, k2 = self.sample_rnd_k(best_u_clusters)
            
            if k1 == k2:
                #Split case
                u_clusters_split_test = copy.deepcopy(best_u_clusters)
                for doc_i in range(self.data.n_docs):
                    possible_k = list(range(self.max_topics))
                    for doc_i_k in self.get_doc_i_clusters(doc_i, u_clusters_split_test):
                        if doc_i_k in possible_k:
                            possible_k.remove(doc_i_k)
                    draw = np.random.multinomial(1, [1.0/len(possible_k)]*len(possible_k))
                    k_split = possible_k[np.nonzero(draw)[0][0]]
                    u_end_split = self.sample_u(doc_i, k1, u_clusters_split_test)
                    u_k_cluster = self.get_k_cluster(k1, u_clusters_split_test)
                    u_begin, u_end = u_k_cluster.get_segment(doc_i)
                    
                    u_k_cluster.remove_seg(doc_i, u_begin, u_end_split)
                    u_begin_split = u_begin
                    
                    move_seg_ll,\
                    move_u_clusters = self.test_split(k_split,\
                                                      doc_i,\
                                                      u_begin_split,\
                                                      u_end_split,\
                                                      u_clusters_split_test)
            else:
                #Merge case
                u_clusters_merge = copy.deepcopy(best_u_clusters)
                u_k_cluster = self.get_k_cluster(k1, u_clusters_merge)
                for doc_i in u_k_cluster.get_docs():
                    u_begin, u_end = u_k_cluster.get_segment(doc_i)
                    if u_end == self.data.doc_len(doc_i)-1:
                        continue#this is the last sentence, cant perform merge
                    
                    uc1 = self.get_k_cluster(k1, u_clusters_merge)
                    uc1.remove_seg(doc_i, u_begin, u_end)
                    if len(uc1.get_docs()) == 0:
                        u_clusters_merge.remove(uc1)
                    
                    next_u_cluster = self.get_next_cluster(k1, doc_i, u_clusters_merge)
                    u_begin_next_c, u_end_next_c = next_u_cluster.get_segment(doc_i)
                    next_u_cluster.remove_seg(doc_i, u_begin_next_c, u_end_next_c)
                    next_u_cluster.add_sents(u_begin, u_end_next_c, doc_i)
                    move_u_clusters = u_clusters_merge
                    move_seg_ll = self.segmentation_ll(u_clusters_merge)[0]
                    
            accept = self.accept_move(move_seg_ll, best_seg_ll)
            if accept:
                self.total_accepts += 1
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
        iters = 500000
        self.set_gl_data(self.data)
        u_clusters = self.rnd_init_seg()
        self.best_segmentation[-1] = [(-np.inf, u_clusters)]
        t = trange(iters, desc='', leave=True)
        for i in t:
            t.set_description("Iter %d" % i)
            if i == 4:
                a = 0
            u_clusters = self.mcmc_segmentation_step(u_clusters)
        print("#accepts: %d #rejects: %d" % (self.total_accepts, iters-self.total_accepts))
        for doc_i in self.n_samples:
            print("doc_%d u_sampled: %s" %(doc_i, str(self.n_samples[doc_i])))
        #self.best_segmentation[-1] = self.samples_decoder()
        #print("GS ll: %.3f" % self.segmentation_ll(self.data.get_rho_u_clusters()))
        