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
    return wd
