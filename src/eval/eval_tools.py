'''
Created on Feb 9, 2017

@author: root
'''
from segeval import window_diff

def segeval_converter(segmentation):
    segeval_format = []
    sent_count = 0
    for sent in segmentation:
        if sent == 1:
            segeval_format.append(sent_count+1)
            sent_count = 0
        else:
            sent_count += 1
    segeval_format.append(sent_count)
    return segeval_format

def wd(hyp_seg, ref_seg):
    hyp_seg = segeval_converter(hyp_seg)
    ref_seg = segeval_converter(ref_seg)
    wd = window_diff(hyp_seg, ref_seg)
    return float(wd)

def wd_evaluator(estimated_rho, doc):
    class_name = doc.__class__.__name__
    if class_name  == "SyntheticRndTopicMultiDoc" or class_name == "SyntheticDittoDocs" or class_name == "RndTopicsParallelModel":
        doc_begin = 0
        wd_results = []
        for doc_end in doc.docs_index:
            doc_rho = doc.rho[doc_begin:doc_end]
            doc_estimated_rho = estimated_rho[doc_begin:doc_end]
            doc_estimated_rho[-1] = 0
            wd_results.append(wd(doc_estimated_rho, doc_rho))
        return wd_results
    else:
        return [wd(estimated_rho, doc.rho)]
            
