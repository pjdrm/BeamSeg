'''
Created on Mar 2, 2018

@author: pjdrm
'''
from model.dp.segmentor import AbstractSegmentor, SentenceCluster
import numpy as np
import copy

class MultiDocGreedySeg(AbstractSegmentor):
    
    def __init__(self, beta, data, max_topics=None):
        super(MultiDocGreedySeg, self).__init__(beta, data)
        self.max_topics = self.data.max_doc_len if max_topics is None else max_topics
        
    def greedy_segmentation_step(self):
        '''
        Similar to vi_segmentation_step, but considers all
        valid u_clusters where a sentence can be inserted.
        Technically does not use VI. 
        '''
        final_u_clusters = []
        for doc_i in range(self.data.n_docs):
            for u in range(self.data.doc_len(doc_i)):
                possible_clusters = self.get_valid_insert_clusters(doc_i, final_u_clusters)
                best_seg_ll = -np.inf
                best_u_clusters = None
                for k in possible_clusters:
                    current_u_clusters = copy.deepcopy(final_u_clusters)
                    u_k_cluster = None
                    for u_cluster in current_u_clusters:
                        if u_cluster.k == k:
                            u_k_cluster = u_cluster
                            break
                        
                    if u_k_cluster is None:
                        u_k_cluster = SentenceCluster(u, u, [doc_i], self.data, k)
                        current_u_clusters.append(u_k_cluster)
                    else:
                        u_k_cluster.add_sents(u, u, doc_i)
                        
                    seg_ll = self.segmentation_ll(current_u_clusters)
                    if seg_ll > best_seg_ll:
                        best_seg_ll = seg_ll
                        best_u_clusters = current_u_clusters
                final_u_clusters = best_u_clusters
        self.best_segmentation[-1] = final_u_clusters
        
    def segmentation_ll(self, u_clusters):
        '''
        Returns the log likelihood of the segmentation of all documents.
        :param u_clusters: list of SentenceCluster corresponding to the best segmentation up to u-1
        '''
        segmentation_ll = 0.0
        for u_cluster in u_clusters:
            word_counts = u_cluster.get_word_counts()
            segmentation_ll += self.segment_ll(word_counts)
        return segmentation_ll
    
    def segment_docs(self):
        self.greedy_segmentation_step()