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
        self.data = data
        
    def segment_docs(self):
        for doc in self.single_docs:
            data = Data(doc)
            dp_model = dp_seg.MultiDocDPSeg(self.beta, data)
            dp_model.segment_docs()
            self.sd_segs.append(dp_model.get_segmentation(0))
    
    def get_segmentation(self, doc_i):
        return self.sd_segs[doc_i]
    
    def get_all_segmentations(self):
        '''
        Returns a single vector with the final
        segmentation for all documents.
        '''
        all_segs = []
        for doc_i in range(self.data.n_docs):
            all_segs += self.get_segmentation(doc_i)
        return all_segs