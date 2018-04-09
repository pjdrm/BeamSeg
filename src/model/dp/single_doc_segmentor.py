'''
Created on Mar 2, 2018

@author: pjdrm
'''
from model.dp.segmentor import Data
import model.dp.multi_doc_dp_segmentor as dp_seg

class SingleDocDPSeg():
    
    def __init__(self, beta, single_docs, data, seg_type=None):
        self.beta = beta
        self.single_docs = single_docs
        self.seg_type = seg_type
        self.sd_segs = []
        self.sd_topics = []
        self.data = data
        self.best_segmentation = [[(None, [])]]
        self.desc = "SD_seg"
        
    def segment_docs(self):
        for doc in self.single_docs:
            data = Data(doc)
            dp_model = dp_seg.MultiDocDPSeg(self.beta, data, desc=self.desc)
            dp_model.max_row_cache = 1
            dp_model.segment_docs()
            u_clusters = dp_model.best_segmentation[-1][0][1]
            self.sd_segs.append(dp_model.get_final_segmentation(0))
            self.sd_topics.append(dp_model.get_seg_with_topics(0, u_clusters))
    
    def get_seg_with_topics(self, doc_i, u_clusters):
        return self.sd_topics[doc_i]
        
    def get_final_segmentation(self, doc_i):
        return self.sd_segs[doc_i]
    
    def get_all_segmentations(self):
        '''
        Returns a single vector with the final
        segmentation for all documents.
        '''
        all_segs = []
        for doc_i in range(self.data.n_docs):
            all_segs += self.get_final_segmentation(doc_i)
        return all_segs