import os
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from scipy import stats
import shutil
plt.style.use('ggplot')
mpl.rcParams['axes.facecolor'] = '#FFFFFF'
plt.rcParams["figure.figsize"] = [16.0, 3.0]

def plot_annotations(ref_seg, hyp_seg, outFile):
    df2 = pd.DataFrame(np.array([ref_seg, hyp_seg]).T, columns=["Reference", "BeamSeg"])
    ax = df2.plot.bar(stacked=True, width=1)
    ax.yaxis.set_ticks_position('left')
    ax.axhline(y = 0, linewidth=1.5, color='k')
    ax.axhline(y = 1.995, linewidth=1.3, color='k')
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') 
    plt.yticks(np.arange(0, 3, 1))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.xlabel('Sentences', fontsize=14, color="k")
    plt.ylabel('#Boundaries', fontsize=14, color="k")
    plt.legend(loc='upper left', fancybox=None, ncol=1, fontsize = 14)
    plt.savefig(outFile, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()
    
def load_segs(filePath):
    with open(filePath) as f:
        lins = f.readlines()
        ref_seg = eval(lins[0][3:])
        hyp_seg = eval(lins[1][3:])
        return convert_to_zero_one_seg_seg(ref_seg), convert_to_zero_one_seg_seg(hyp_seg)
    
def convert_to_zero_one_seg_seg(seg):
    seg_converted = [0]*seg[-1]
    sum_index = 0
    for bound_index in seg[:-1]:
        seg_converted[bound_index-1] = 1
    return seg_converted

def seg_count_diff(file_path):
    with open(file_path) as f:
        lins = f.readlines()
        ref_seg = eval(lins[0].replace("R: ", ""))
        ref_seg_count = len(ref_seg)+1
        hyp_seg = eval(lins[1].replace("H: ", ""))
        hyp_seg_count = len(hyp_seg)+1
        doc_len = hyp_seg[-1]
    return ref_seg_count - hyp_seg_count, doc_len

def seg_granularity_analysis(seg_files_dir):
    total_over_segs = 0
    total_under_segs = 0
    total_regular_segs = 0
    seg_diffs = []
    doc_lens = []
    thr = 3
    for seg_file in os.listdir(seg_files_dir):
        seg_diff, doc_len = seg_count_diff(os.path.join(seg_files_dir, seg_file))
        seg_diffs.append(seg_diff)
        doc_lens.append(doc_len)
        if seg_diff < 0 and abs(seg_diff) >= thr:
            total_over_segs += 1
        elif seg_diff > 0 and abs(seg_diff) >= thr:
            total_under_segs += 1
        else:
            total_regular_segs += 1
        print(seg_file + "\t" + str(seg_diff) + "\t" + str(doc_len))
    print("#RegSeg: " + str(total_regular_segs) + " #OverSeg: " + str(total_over_segs) + " #UnderSeg: " + str(total_under_segs))
    print("p-val: " + str(stats.pearsonr(seg_diffs, doc_lens)[1]))
    
def sent_sim_analysis(sim_vals_file):
    with open(sim_vals_file) as f:
        lins = f.readlines()
        sim_matrix = []
        sim_lin = []
        for lin in lins:
            if lin == "New target sentence\n":
                sim_matrix.append(sim_lin)
                sim_lin = []
            else:
                sim_lin.append(lin.strip())
    
    matrix_str = ""
    for i in range(len(sim_matrix)):
        for j in range(len(sim_matrix[i])):
            if float(sim_matrix[i][j]) == -1:
                sim_matrix[i][j] = str(0)
            matrix_str += sim_matrix[i][j] + "\t"
        matrix_str += "\n"
    with open("sim_matrix.txt", "w+") as outf:
        outf.write(matrix_str)
        
def sent_sim_plot():
    avg_sim = [0.3813882969, 0.4784697848, 0.6083460428, 0.4082333656, 0.3336742835, 0.3508654184, 0.5295317283, 0.4779977163, 0.3782607864, 0.5742298944, 0.3753433394, 0.5644519031, 0.5037548737, 0.5947106638, 0.6352597551, 0.5621225891, 0.5821898844, 0.638167512, 0.6096945988, 0.494339252, 0.4179406308, 0.4605510883, 0.3873341894, 0.3684461848, 0.4956966548, 0.6175870463, 0.3980198082, 0.572019192, 0.5378988963, 0.526260024, 0.6954863859, 0.6841880824, 0.5032922049, 0.5903061542, 0.5487936519, 0.5213120981, 0.5108751447, 0.9102939456, 0.5229270893, 0.610839578, 0.5578184962, 0.5954091258, 0.4706422372, 0.5845502022, 0.6737175324, 0.6389639785, 0.5565976164, 0.6162520486, 0.4119815417, 0.5784226247, 0.8893247106, 0.4748764222, 0.7391116674, 0.4873671517, 0.5227650157, 0.3439447048, 0.6106265872, 0.7505945434, 0.6271008097, 0.6792056097, 0.4794650684, 0.5999641491, 0.4661486249, 0.4160364234, 0.3341410029, 0.3385070888, 0.5527688464, 0.6132419062, 0.549986532, 0.6132419062, 0.3971256458, 0.4312596105, 0.6228243124, 0.4797455723, 0.5034617508, 0.6106265872, 0.7505945434, 0.6271008097, 0.512025667, 0.6481925918, 0.6481925918, 0.5898651796, 0.4999826444, 0.539671736, 0.6132419062, 0.6296220478]
    df2 = pd.DataFrame(np.array([avg_sim]).T, columns=[""])
    ax = df2.plot.bar(width=0.6, legend=None, color = ["#60A1C7"])
    ax.yaxis.set_ticks_position('left')
    ax.spines['bottom'].set_color('#A9A9A9')
    ax.spines['left'].set_color('#A9A9A9')
    #ax.axhline(y = 0, linewidth=1.5, color='k')
    #ax.axhline(y = 1.995, linewidth=1.3, color='k')
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') 
    #plt.yticks(np.arange(0, 3, 1))
    plt.setp(ax.get_xticklabels(), visible=True)
    plt.xlabel('Sentences', fontsize=14, color="k")
    plt.ylabel('Average similarity', fontsize=14, color="k")
    #plt.legend(visible=False)
    plt.savefig("sent_sim.pdf", bbox_inches='tight')
    
            
def run(dirPath, outDir):
    for filePath in os.listdir(dirPath):
        print(filePath)
        ref_seg, hyp_seg = load_segs(os.path.join(dirPath, filePath))
        if "html" in filePath:
            doc_type = "html"
        elif "ppt" in filePath:
            doc_type = "ppt"
        elif "pdf" in filePath:
            doc_type = "pdf"
        else:
            doc_type = "video" 
        plot_annotations(ref_seg, hyp_seg, os.path.join(outDir, doc_type, filePath.replace("txt", "png")))

def plot_annotations_v2(annotations, fileName, annotators_id, outFile):
    mpl.rcParams['axes.facecolor'] = '#FFFFFF'
    plt.rcParams["figure.figsize"] = [28.0, 4.0]
    df2 = pd.DataFrame(annotations, columns=annotators_id)
    ax = df2.plot.bar(stacked=True, width=0.8, color = ["#a6d96a", "#2b83ba", "#d7191c", "#ffc962", "#ae23c6"])
    ax.yaxis.set_ticks_position('left')
    ax.axhline(y = 0.01, linewidth=2.0, color='k')
    #ax.axhline(y = 3.0, linewidth=0.005, color='k')
    plt.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    left='off',
    labelbottom='off')
    #plt.yticks(np.arange(0, len(annotators_id)+1, 1), color = "k")
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.setp(ax.get_yticklabels(), visible=False)
    fs = 18
    plt.xlabel('Sentences', fontsize=fs, color = "k", labelpad=15)
    #plt.ylabel('#Boundaries', fontsize=fs, color = "k")
    plt.legend(bbox_to_anchor=(0.98, 1.1), loc=2, fancybox=None, ncol=1, fontsize = fs, frameon=False)
    plt.savefig(outFile, bbox_inches='tight')

def convert_zero_one_seg(seg):
    ret_seg = []
    prev_topic = seg[0]
    for t in seg:
        if t != prev_topic:
            ret_seg[-1] = 1
        ret_seg.append(0)
        prev_topic = t
    return ret_seg

def get_segmentation(seg_fp):
    refs = []
    hyps = []
    with open(seg_fp) as f:
        lins = f.readlines()
    
    for i, l in enumerate(lins):
        if lins[i+1].startswith("Baseline"):
            break
        if "GS:" in l:
            r = eval(lins[i+1])
            r[-1] = 0
            refs.append(r)
        if l.startswith("sf") and i >= 8:
            h = eval(lins[i+1])
            h[-1] = 0
            hyps.append(h)
    docs = eval(lins[-1])
    return refs, hyps, docs

def find_results(tc, segs_dict, results):
    results_segs = []
    for doc_target, seg_target in zip(segs_dict[tc]["docs"], segs_dict[tc]["refs"]):
        for doc, seg in zip(results["docs"], results["hyps"]):
            if doc_target == doc:
                if len(seg) != len(seg_target):
                    return None
                    #print("WARNING: %s %d %d"%(doc_target, len(seg), len(seg_target)))
                results_segs.append(seg)
                break
    if len(results_segs) == 0:
        return None
    return results_segs

def merge_segs(segs_dict, segs_other_dict):
    for tc in segs_dict:
        for model_desc in segs_other_dict:
            results_segs = find_results(tc, segs_dict, segs_other_dict[model_desc])
            if results_segs is not None:
                segs_dict[tc][model_desc] = results_segs
    return segs_dict

def get_doc_type(doc):
    if "html" in doc:
        return "html"
    elif "ppt" in doc:
        return "ppt"
    elif "pdf" in doc:
        return "pdf"
    else:
        return "video"
    
def plot_model_segs(models_l, domain, out_dir, group_mod=False):
    name_map = {"Reference": "Reference",
                "segtt_modality_gp": "BeamSeg-D-GP-M",
                "segtt_dataset_gp": "BeamSeg-D-GP-D",
                "segbl_modality_gp": "BeamSeg-I-GP-M",
                "segbl_dataset_gp": "BeamSeg-I-GP-D",
                "segtt_modality_bb": "BeamSeg-D-BB",
                "segbl_modality_bb": "BeamSeg-I-BB",
                "multiseg": "MultiSeg",
                "cvs": "CVS",
                "bayesseg-MD": "Bayesseg-MD"}
    #for d in os.listdir(out_dir):
    #    os.remove(out_dir+"/"+d)
        
    #if group_mod:
    #    os.mkdir(out_dir+"/html")
    #    os.mkdir(out_dir+"/ppt")
    #    os.mkdir(out_dir+"/pdf")
    #    os.mkdir(out_dir+"/video")
        
    segs_dict = {}
    for root_dir, model_desc in models_l:
        if "beamseg" in root_dir:
            for seg_fp in os.listdir(root_dir):
                if domain in seg_fp and model_desc in seg_fp:
                    refs, model_segs, docs = get_segmentation(root_dir+"/"+seg_fp)
                    test_case = seg_fp.replace("_"+model_desc+".txt", "")
                    if test_case not in segs_dict:
                        segs_dict[test_case] = {}
                    segs_dict[test_case][model_desc] = model_segs
                    segs_dict[test_case]["refs"] = refs
                    segs_dict[test_case]["docs"] = docs

    segs_other_dict = {}
    for root_dir, model_desc in models_l:
        if "beamseg" not in root_dir:
            segs_fp = root_dir+"/segmentations/"
            if domain == "bio":
                segs_fp += "mw_bio/results_segmentation.txt"
            elif domain == "news":
                segs_fp += "mw_news/results_segmentation.txt"
            elif domain == "lectures":
                segs_fp += "mw_lectures/results_segmentation.txt"
            elif domain == "avl":
                segs_fp += "avl_trees/results_segmentation.txt"

            segs_other_dict[model_desc] = {"docs": [], "hyps": []}
            if model_desc == "bayesseg-MD":
                for doc_name in os.listdir(segs_fp.replace("results_segmentation.txt", "")):
                    with open(segs_fp.replace("results_segmentation.txt", "")+"/"+doc_name) as f:
                        lins = f.readlines()
                        ref_file = eval(lins[0].split("R: ")[1])
                        hyp_file = eval(lins[1].split("H: ")[1])
                        
                        ref = [0]*ref_file[-1]
                        for b in ref_file:
                            ref[b-1] = 1
                        ref[-1] = 0
                        
                        hyp = [0]*hyp_file[-1]
                        for b in hyp_file:
                            hyp[b-1] = 1
                        hyp[-1] = 0
                        segs_other_dict[model_desc]["docs"].append(doc_name)
                        segs_other_dict[model_desc]["hyps"].append(hyp)
                        
            else:
                with open(segs_fp) as f:
                    lins = f.readlines()
                for l in lins:
                    if "docId: " in l:
                        doc_name = l.split(" ")[1]
                        if model_desc != "cvs":
                            hyp = eval(l.split("Hyp: ")[1])
                            hyp[-1] = 0
                        else:
                            doc_name = doc_name.replace(".", "").replace("ref", ".txt")
                            hyp = []
                            hyp_file = eval(l.split("ref ")[1])
                            for seg in hyp_file:
                                hyp += [0]*seg
                                hyp[-1] = 1
                            hyp[-1] = 0
                        segs_other_dict[model_desc]["docs"].append(doc_name)
                        segs_other_dict[model_desc]["hyps"].append(hyp)

    segs_dict = merge_segs(segs_dict, segs_other_dict)
    for tc in segs_dict:
        all_segs = []
        all_models = []
        flag = False
        for model in segs_dict[tc]:
            if model == "docs" or model == "refs":
                continue
            model_segs = segs_dict[tc][model]
            ref_segs = segs_dict[tc]["refs"]
            docs = segs_dict[tc]["docs"]
            if len(model_segs) != len(ref_segs):
                flag = True
            all_segs.append(model_segs)
            all_models.append(model)
            
        if flag:
            continue
        all_models = ["Reference"]+all_models
        for i in range(len(all_models)):
            all_models[i] = name_map[all_models[i]]
        for j in range(len(all_segs[0])):
            current_segs = []
            for i in range(len(all_segs)):
                current_segs.append(all_segs[i][j])
            current_segs = [ref_segs[j]]+current_segs
            doc = docs[j].replace(".txt", "")
            out_fn = "_".join(all_models)
            tc_final = tc
            if group_mod:
                doc_type = get_doc_type(doc)
                tc_final = doc_type+"/"+tc
            plot_annotations_v2(np.array(current_segs).T, "", all_models, out_dir+"/"+tc_final+"_"+out_fn+"-"+doc+".png")



m1 = ["/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/beamseg", "segtt_modality_gp"]
m2 = ["/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/beamseg", "segbl_modality_gp"]
m3 = ["/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/beamseg", "segtt_modality_bb"]
m4 = ["/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/beamseg", "segbl_modality_bb"]
m5 = ["/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/multiseg", "multiseg"]
m6 = ["/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/cvs", "cvs"]
m7 = ["/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/beamseg", "segtt_dataset_gp"]
m8 = ["/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/bayesseg-MD", "bayesseg-MD"]
m9 = ["/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/beamseg", "segtt_dataset_gp"]
m10 = ["/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/beamseg", "segbl_dataset_gp"]
models_l = [m1, m9, m2, m10]
plot_model_segs(models_l, "L20", "/home/pjdrm/Desktop/seg_plots", True)
