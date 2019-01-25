'''
Created on Jan 23, 2019

@author: pjdrm
'''
import segeval
from eval.eval_tools import segeval_converter, wd
import os

def test_wd(segs_dir):
    def conv_seg(seg):
        new_seg = []
        prev_bound = 0
        for bond in seg:
            for i in range(prev_bound, bond):
                new_seg.append(0)
            prev_bound = bond
            new_seg[-1] = 1
        return new_seg
    
    for fp in os.listdir(segs_dir):
        with open(segs_dir+"/"+fp) as f:
            lins = f.readlines()
            ref = conv_seg(eval(lins[0][3:]))
            hyp = conv_seg(eval(lins[1][3:]))
            print("docId: %s pk -1 wd %f" % (fp, wd(hyp, ref)))
        
def get_boundary_stats(fp):
    with open(fp) as f:
        lins = f.readlines()
    
    doc_names = eval(lins[-1])
    j = 0
    for i in range(len(lins)):
        lin = lins[i]
        if lin.startswith("GS:"):
            print(doc_names[j])
            j += 1
            ref = segeval_converter(eval(lins[i+1]))
            hyp = segeval_converter(eval(lins[i+3]))
            stats = segeval.boundary_statistics(hyp, ref)
            stats["transpositions"] = len(stats["transpositions"])
            stats["matches"] = len(stats["matches"])
            stats["substitutions"] = len(stats["substitutions"])
            stats["full_misses"] = len(stats["full_misses"])
            stats["additions"] = len(stats["additions"])
            print(stats)
            
#get_boundary_stats("/home/pjdrm/workspace/TopicTrackingSegmentation/thesis_exp/beamseg/bio_d1_segtt_modality_gp.txt")
test_wd("/media/pjdrm/EMTEC C450/bayesseg_segs/L20")
            