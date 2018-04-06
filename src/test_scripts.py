'''
Created on Feb 22, 2018

@author: pjdrm
'''
from dataset.synthetic_doc_cvb import CVBSynDoc2, CVBSynSkipTopics
import time
from model.dp.segmentor import Data
import model.dp.multi_doc_dp_segmentor as dp_seg
import model.dp.multi_doc_dp_segmentor_single_cache as dp_seg_sc
import model.dp.multi_doc_vi_segmentor as vi_seg
import model.dp.single_doc_segmentor as sd_seg
import model.dp.multi_doc_greedy_segmentor as greedy_seg
import model.dp.multi_doc_mcmc_segmentor as mcmc_seg
import copy
import numpy as np
import toyplot
import toyplot.pdf
import json
from eval.eval_tools import wd_evaluator
from dataset.real_doc import MultiDocument

def md_eval(doc_synth, models, models_desc):
    seg_times = []
    for model in models:
        start = time.time()
        model.segment_docs()
        end = time.time()
        seg_time = (end - start)
        seg_times.append(seg_time)
        
    for i in range(len(models)):
        print(str(models_desc[i])+" Time: %f" % seg_times[i])
        
    md_segs = [[] for i in range(len(models))]
    for i, model in enumerate(models):
        for doc_i in range(model.data.n_docs):
            md_segs[i].append(model.get_final_segmentation(doc_i))
            
    gs_segs = []
    for gs_doc in models[0].data.docs_rho_gs:
        gs_segs.append(gs_doc)
        
    for j, gs_seg in enumerate(gs_segs):
        print("GS:  " + str(gs_seg.tolist()))
        for i in range(len(models)):
            print(str(models_desc[i]) + ": " + str(md_segs[i][j]))
        print("")
        
    for i, model in enumerate(models):
        print(str(models_desc[i])+" WD: "+str(wd_evaluator(model.get_all_segmentations(), doc_synth)))
    
def single_vs_md_eval(doc_synth, beta, md_all_combs=True, md_fast=True, print_flag=False):
    '''
    Print the WD results when segmenting single documents
    and all of them simultaneously (multi-doc model)
    :param doc_synth: collection of synthetic documents
    :param beta: beta prior vector
    :param print_flag: boolean to print or not the segmentation results
    '''
    single_docs = doc_synth.get_single_docs()
    single_doc_wd = []
    time_wd_results = []
    start = time.time()
    sd_segs = []
    for doc in single_docs:
        data = Data(doc)
        dp_model = sd_seg.SingleDocDPSeg(beta, single_docs, data)
        dp_model.segment_docs()
        sd_segs.append(dp_model.get_final_segmentation(0))
        single_doc_wd += wd_evaluator(dp_model.get_all_segmentations(), doc)
    end = time.time()
    sd_time = (end - start)
    #single_doc_wd = ['%.3f' % wd for wd in single_doc_wd]
    time_wd_results.append(("SD", sd_time, ['%.3f' % wd for wd in single_doc_wd]))
    
    data = Data(doc_synth)
    if md_all_combs:
        dp_model = dp_seg.MultiDocDPSeg(beta, data, seg_type=dp_seg.SEG_ALL_COMBS)
        start = time.time()
        dp_model.segment_docs()
        end = time.time()
        md_time = (end - start)
        multi_doc_wd = wd_evaluator(dp_model.get_all_segmentations(), doc_synth)
        #multi_doc_wd = ['%.3f' % wd for wd in multi_doc_wd]
        time_wd_results.append(("MD", md_time, ['%.3f' % wd for wd in multi_doc_wd]))
        
        md_segs = []
        for doc_i in range(dp_model.data.n_docs):
            md_segs.append(dp_model.get_final_segmentation(doc_i))
    
    if md_fast: 
        md_fast_segs = []
        dp_model = dp_seg.MultiDocDPSeg(beta, data, seg_type=dp_seg.SEG_FAST)
        start = time.time()
        dp_model.dp_segmentation_step()
        end = time.time()
        md_fast_time = (end - start)
        multi_fast_doc_wd = wd_evaluator(dp_model.get_all_segmentations(), doc_synth)
        time_wd_results.append(("MF", md_fast_time, ['%.3f' % wd for wd in multi_fast_doc_wd]))
        for doc_i in range(dp_model.data.n_docs):
            md_fast_segs.append(dp_model.get_final_segmentation(doc_i))
            
    if print_flag:
        gs_segs = []
        for gs_doc in doc_synth.get_single_docs():
            gs_segs.append(gs_doc.rho.tolist())
            
        if md_all_combs and md_fast:
            for doc_i in range(doc_synth.n_docs):
                print("%s: %s" % ("GS", str(gs_segs[doc_i])))
                print("%s: %s" % ("SD", str(sd_segs[doc_i])))
                print("%s: %s" % ("MD", str(md_segs[doc_i])))
                print("%s: %s\n" % ("MF", str(md_fast_segs[doc_i])))
        elif md_all_combs:
            for doc_i in range(doc_synth.n_docs):
                print("%s: %s" % ("GS", str(gs_segs[doc_i])))
                print("%s: %s" % ("SD", str(sd_segs[doc_i])))
                print("%s: %s\n" % ("MD", str(md_segs[doc_i])))
        else:
            for doc_i in range(doc_synth.n_docs):
                print("%s: %s" % ("GS", str(gs_segs[doc_i])))
                print("%s: %s" % ("SD", str(sd_segs[doc_i])))
                print("%s: %s\n" % ("MF", str(md_fast_segs[doc_i])))
    for time_res in time_wd_results:  
        print("%s: %s time: %f" % (time_res[0], time_res[2], time_res[1]))
    
    return single_doc_wd, multi_fast_doc_wd

def merge_docs(target_docs):
    target_docs_copy = copy.deepcopy(target_docs)
    merged_doc = target_docs_copy[0]
    all_rho = []
    all_docs_index = []
    all_d_u_wi_indexes = []
    all_U_W_counts = None
    carry_index = 0
    for doc_synth in target_docs:
        new_index = (np.array(doc_synth.docs_index)+carry_index).tolist()
        carry_index = new_index[-1]
        all_docs_index += new_index
        doc_synth.rho[-1] = 1
        all_rho += doc_synth.rho.tolist()
        if all_U_W_counts is None:
            all_U_W_counts = doc_synth.U_W_counts
        else:
            all_U_W_counts = np.vstack((all_U_W_counts, doc_synth.U_W_counts))
        all_d_u_wi_indexes.append(doc_synth.d_u_wi_indexes[0])
            
    all_rho[-1] = 0
    
    merged_doc.n_docs = len(target_docs)
    merged_doc.rho = np.array(all_rho)
    merged_doc.docs_index = all_docs_index
    merged_doc.U_W_counts = all_U_W_counts
    merged_doc.d_u_wi_indexes = all_d_u_wi_indexes
    merged_doc.isMD = True
    
    return merged_doc

def incremental_eval(doc_synth, beta):
    def grouped_bars(axes, data, group_names, group_width=None):
        if group_width is None:
            group_width=1 - 1.0 / (data.shape[1] + 1)
            
        group_left_edges = np.arange(data.shape[0], dtype="float") - (group_width / 2.0)
        bar_width = group_width / data.shape[1]
        
        marks = []
        axes.x.ticks.locator = toyplot.locator.Explicit(labels=group_names)
        for index, series in enumerate(data.T):
            left_edges = group_left_edges + (index * bar_width)
            right_edges = group_left_edges + ((index + 1) * bar_width)
            marks.append(axes.bars(left_edges, right_edges, series, opacity=0.5))
            
        return marks

    single_docs = doc_synth.get_single_docs()
    all_sd_results = []
    all_md_results = []
    for i in range(1, doc_synth.n_docs+1):
        target_docs = single_docs[:i]
        merged_doc_synth = merge_docs(target_docs)
        sd_results, md_results = single_vs_md_eval(merged_doc_synth, beta, md_all_combs=False, print_flag=True)
        all_sd_results.append(sd_results)
        all_md_results.append(md_results)
        
    final_results = []
    for sd_wds, mf_wds in zip(all_sd_results, all_md_results):
        n_ties = 0.0
        n_mf_win = 0.0
        n_mf_lost = 0.0
        for sd_wd, mf_wd in zip(sd_wds, mf_wds):
            if sd_wd == mf_wd:
                n_ties += 1.0
            elif mf_wd > sd_wd:
                n_mf_win += 1.0
            else:
                n_mf_lost += 1.0
        n_total = n_ties+n_mf_win+n_mf_lost
        n_ties_percent = n_ties/n_total
        n_mf_win_percent = n_mf_win/n_total
        n_mf_lost_percent = n_mf_lost/n_total
        final_results.append(np.array([n_ties_percent, n_mf_win_percent, n_mf_lost_percent])*100.0)
        
    group_names = list(range(1, doc_synth.n_docs+1))
    canvas = toyplot.Canvas(width=600, height=300)
    axes = canvas.cartesian()
    axes.x.label.text = "#Docs"
    axes.y.label.text = "Percentage"
    marks = grouped_bars(axes, np.array(final_results), group_names)
    canvas.legend([
    ("Tie", marks[0]),
    ("Lose", marks[1]),
    ("Win", marks[2])
    ],
    corner=("top-right", 0, 100, 50),
    );
    toyplot.pdf.render(canvas, "incremental_eval_results.pdf")
    
def dp_only_test():
    use_seed = False
    seed = 31
    if use_seed:
        np.random.seed(seed)
        
    W = 10
    beta = np.array([0.08]*W)
    n_docs = 2
    doc_len = 20
    pi = 0.2
    sent_len = 6
    n_seg = 3
    doc_synth = CVBSynDoc2(beta, pi, sent_len, n_seg, n_docs)
    data = Data(doc_synth)
    
    #incremental_eval(doc_synth, beta)
    #single_vs_md_eval(doc_synth, beta, md_all_combs=False , md_fast=True, print_flag=True)
    dp_model = dp_seg.MultiDocDPSeg(beta, data, seg_type=dp_seg.SEG_FAST)
    md_eval(doc_synth, dp_model)
    
def vi_only_test():
    use_seed = True
    seed = 26
    if use_seed:
        np.random.seed(seed)
        
    W = 12
    beta = np.array([0.08]*W)
    n_docs = 2
    pi = 0.2
    sent_len = 12
    n_seg = 3
    doc_synth = CVBSynDoc2(beta, pi, sent_len, n_seg, n_docs)
    data = Data(doc_synth)
    
    iters = 20
    vi_model = vi_seg.MultiDocVISeg(beta, data, max_topics=n_seg, n_iters=iters, log_dir="/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/logs")
    md_eval(doc_synth, [vi_model], ["VI"])
    
def dp_vs_vi():
    use_seed = True
    seed = 26
    if use_seed:
        np.random.seed(seed)
        
    W = 12
    beta = np.array([0.08]*W)
    n_docs = 2
    pi = 0.2
    sent_len = 12
    n_segs = 3
    doc_synth = CVBSynDoc2(beta, pi, sent_len, n_segs, n_docs)
    data = Data(doc_synth)
    
    iters = 20
    vi_model = vi_seg.MultiDocVISeg(beta, data, max_topics=n_segs, n_iters=iters, log_dir="../logs/", log_flag=True)
    dp_model = dp_seg.MultiDocDPSeg(beta, data, seg_type=dp_seg.SEG_FAST)
    md_eval(doc_synth, [dp_model, vi_model], ["DP", "VI"])
    
def skip_topics_test():
    use_seed = True
    seed = 56#48
    
    if use_seed:
        np.random.seed(seed)
        
    W = 10
    beta = np.array([0.3]*W)
    n_docs = 3
    pi = 0.25 
    sent_len = 6
    n_segs = 3
    n_topics = 5
    
    skip_topics_syn = CVBSynSkipTopics(beta, pi, sent_len, n_segs, n_docs, n_topics)
    data = Data(skip_topics_syn)
    
    single_docs = skip_topics_syn.get_single_docs()
    sd_model = sd_seg.SingleDocDPSeg(beta, single_docs, data)
    
    vi_dp_config = {"type": vi_seg.DP_VI_SEG, "seg_func": vi_seg.QZ_VOTING}
    vi_dp_config_v2 = {"type": vi_seg.DP_VI_SEG, "seg_func": vi_seg.QZ_VOTING_V2}
    vi_dp_qz_ll_config = {"type": vi_seg.DP_VI_SEG, "seg_func": vi_seg.QZ_LL}
    vi_config = {"type":vi_seg.VI_SEG}
    
    vi_dp_qz_voting_model = vi_seg.MultiDocVISeg(beta,\
                                                 data,\
                                                 max_topics=n_topics,\
                                                 n_iters=20,\
                                                 seg_config=vi_dp_config,\
                                                 log_dir="../logs/",\
                                                 log_flag=True)
    
    vi_dp_qz_voting_model_v2 = vi_seg.MultiDocVISeg(beta,\
                                                    data,\
                                                    max_topics=n_topics,\
                                                    n_iters=20,\
                                                    seg_config=vi_dp_config_v2,\
                                                    log_dir="../logs/",\
                                                    log_flag=True)
    
    vi_dp_qz_ll_model = vi_seg.MultiDocVISeg(beta, data, max_topics=n_topics,\
                                             seg_dur=1.0/pi,\
                                             std=3.0,\
                                             use_prior=True,
                                             n_iters=5,\
                                             seg_config=vi_dp_qz_ll_config,\
                                             log_dir="../logs/", log_flag=True)
    
    vi_model = vi_seg.MultiDocVISeg(beta,\
                                    data,\
                                    max_topics=n_topics,\
                                    n_iters=40,\
                                    seg_config=vi_config,\
                                    log_dir="../logs/",\
                                    log_flag=True)
    
    greedy_model_no_prior = greedy_seg.MultiDocGreedySeg(beta, data, max_topics=n_topics, seg_dur=1.0/pi, std=2.0, use_prior=False)
    greedy_model_std1 = greedy_seg.MultiDocGreedySeg(beta, data, max_topics=n_topics, seg_dur=1.0/pi, std=1.5, use_prior=True)
    greedy_model_std2 = greedy_seg.MultiDocGreedySeg(beta, data, max_topics=n_topics, seg_dur=1.0/pi, std=2.0, use_prior=True)
    greedy_model_std3 = greedy_seg.MultiDocGreedySeg(beta, data, max_topics=n_topics, seg_dur=1.0/pi, std=3.0, use_prior=True)
    mcmc_model = mcmc_seg.MultiDocMCMCSeg(beta, data, max_topics=n_topics, seg_dur=1.0/pi, std=3.0, use_prior=False)
    mcmc_model_std3 = mcmc_seg.MultiDocMCMCSeg(beta, data, max_topics=n_topics, seg_dur=1.0/pi, std=3.0, use_prior=True)
    dp_model = dp_seg.MultiDocDPSeg(beta, data, max_topics=n_topics, seg_type=dp_seg.SEG_SKIP_K)
    dp_model_sc = dp_seg_sc.MultiDocDPSeg(beta, data, max_topics=n_topics, seg_type=dp_seg.SEG_SKIP_K)
    md_eval(skip_topics_syn, [greedy_model_std3, mcmc_model, mcmc_model_std3, ], ["GS3", "MC ", "MCP"])
    #md_eval(skip_topics_syn, [sd_model, greedy_model_std3, mcmc_model_std3], ["SD ", "GS3", "MC "])
    
def skip_topics_incremental_test():
    use_seed = True
    seed = 229#84
    if use_seed:
        np.random.seed(seed)
        
    W = 100#100
    beta = np.array([0.3]*W)
    n_docs = 15
    pi = 0.25
    sent_len = 6
    n_segs = 5
    n_topics = 7
    
    skip_topics_syn = CVBSynSkipTopics(beta, pi, sent_len, n_segs, n_docs, n_topics)
    single_docs = skip_topics_syn.get_single_docs()
    results_dict = dict([(key, []) for key in range(n_docs)])
    
    sd_model = sd_seg.SingleDocDPSeg(beta, single_docs, Data(skip_topics_syn))
    sd_model.segment_docs()
    sd_wd_results = wd_evaluator(sd_model.get_all_segmentations(), skip_topics_syn)
        
        
    for i in range(1, skip_topics_syn.n_docs+1):
        target_docs = single_docs[:i]
        merged_doc_synth = merge_docs(target_docs)
        data = Data(merged_doc_synth)
        greedy_model = greedy_seg.MultiDocGreedySeg(beta, data, max_topics=n_topics, seg_dur=1.0/pi)
        greedy_model.segment_docs()
        wd_results = wd_evaluator(greedy_model.get_all_segmentations(), merged_doc_synth)
        for doc_i, wd_result in enumerate(wd_results):
            results_dict[doc_i].append(wd_result)
            
    print("\nSD baseline:\n")
    for doc_i, wd in enumerate(sd_wd_results):
        print("doc_%d %f" % (doc_i, wd))
        
    print("\nGMD incremental:\n")
    for doc_i in range(n_docs):
        print("doc_%d %s" % (doc_i, str(results_dict[doc_i])))
        
def real_dataset_tests():
    config_file = "/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/dataset/physics_test.json"
    with open(config_file) as data_file:    
        config = json.load(data_file)
    doc_col = MultiDocument(config)
    data = Data(doc_col)
    beta = np.array([0.3]*doc_col.W)
    pi = 0.1
    greedy_model_std3 = greedy_seg.MultiDocGreedySeg(beta, data, max_topics=3, seg_dur=1.0/pi, std=3.0, use_prior=True)
    greedy_model_std3.segment_docs()
        
    
#skip_topics_test()
real_dataset_tests()
