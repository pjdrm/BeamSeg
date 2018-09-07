'''
Created on Feb 22, 2018

@author: pjdrm
'''
from dataset.synthetic_doc_cvb import CVBSynDoc2, CVBSynSkipTopics
import time
from model.dp.segmentor import Data, SEG_TT, SEG_BL
import model.dp.multi_doc_dp_segmentor as dp_seg
import model.dp.multi_doc_dp_segmentor_single_cache as dp_seg_sc
import model.dp.multi_doc_vi_segmentor as vi_seg
import model.dp.single_doc_segmentor as sd_seg
import model.dp.multi_doc_greedy_segmentor as greedy_seg
import model.dp.multi_doc_mcmc_segmentor as mcmc_seg
import model.dp.multi_doc_mcmc_v2_segmentor as mcmc_seg_v2
import copy
import numpy as np
import toyplot
import toyplot.pdf
import toyplot.browser
import json
import operator
import sys
from eval.eval_tools import wd_evaluator, f_measure, accuracy
from dataset.real_doc import MultiDocument
import dirichlet
from itertools import chain, product
from sklearn.cluster import spectral_clustering
import os
from random import shuffle
import cProfile, pstats, io

def hyper_param_opt(data, n=10):
    dir_samples = []
    alpha = 0.01
    for i in range(0, len(data.U_W_counts), n):
        word_counts = np.sum(data.U_W_counts[i:i+n], axis=0)
        total_words = np.sum(word_counts)
        dir_samples.append((word_counts+alpha)/(total_words+alpha*data.W))
    dir_samples = np.array(dir_samples)
    prior = dirichlet.mle(dir_samples, maxiter=sys.maxsize)
    #plot_prior(prior, data)
    return prior

def hyper_param_opt_v2(doc_col):
    single_docs = doc_col.get_single_docs()
    dir_samples = []
    alpha = 0.01
    for doc_i, doc in enumerate(single_docs):
        n = int(doc_col.seg_dur_prior[doc_i][0])
        for i in range(0, len(doc.U_W_counts), n):
            word_counts = np.sum(doc.U_W_counts[i:i+n], axis=0)
            total_words = np.sum(word_counts)
            dir_samples.append((word_counts+alpha)/(total_words+alpha*doc_col.W))
    dir_samples = np.array(dir_samples)
    prior = dirichlet.mle(dir_samples)
    return prior

def get_best_prior(data):
    topic_draws = {}
    flat_gs_topics = data.doc_rho_topics
    flat_gs_topics = list(chain(*flat_gs_topics))
    for u, k in enumerate(flat_gs_topics):
        if k in topic_draws:
            topic_draws[k] += data.U_W_counts[u]
        else:
            topic_draws[k] = data.U_W_counts[u]
    
    dir_samples = []
    alpha = 0.01
    for k in topic_draws:
        total_words = np.sum(topic_draws[k])
        dir_samples.append((topic_draws[k]+alpha)/(total_words+alpha*data.W))
        
    dir_samples = np.array(dir_samples)
    prior = dirichlet.mle(dir_samples)
    
    return prior
        
def plot_topics(u_clusters, inv_vocab):
    for u_cluster in u_clusters:
        topic_plot_dict = {}
        word_counts = u_cluster.get_word_counts()
        total_words = np.sum(word_counts, axis=0)
        word_probs = word_counts/total_words
        for wi, word_prob in enumerate(word_probs):
            topic_plot_dict[wi] = word_prob
        sorted_prob = sorted(topic_plot_dict.items(), key=operator.itemgetter(1), reverse=True)
        sorted_word_probs = []
        words_sorted = []
        for wi, prob in sorted_prob:
            if prob == 0:
                break
            sorted_word_probs.append(prob)
            words_sorted.append(inv_vocab[wi])
        
        if len(words_sorted) > 70:
            h = 1200
        else:
            h = 800
        sorted_word_probs.reverse()
        words_sorted.reverse()
        title = "Topic " + str(u_cluster.k)
        canvas = toyplot.Canvas(width=500, height=h)
        axes = canvas.cartesian(label=title, margin=100)
        axes.bars(sorted_word_probs, along='y')
        axes.y.ticks.locator = toyplot.locator.Explicit(labels=words_sorted)
        axes.y.ticks.labels.angle = -90
        
        toyplot.browser.show(canvas)
    
def plot_prior(prior, data):
    topic_plot_dict = {}
    for wi, word_prob in enumerate(prior):
        topic_plot_dict[wi] = word_prob
            
    sorted_prob = sorted(topic_plot_dict.items(), key=operator.itemgetter(1), reverse=True)
    sorted_word_probs = []
    words_sorted = []
    for wi, prob in sorted_prob:
        if prob == 0:
            break
        sorted_word_probs.append(prob)
        words_sorted.append(data.inv_vocab[wi])
    
    sorted_word_probs.reverse()
    words_sorted.reverse()
    max_words = 8000
    sorted_word_probs = sorted_word_probs[:max_words]
    words_sorted = words_sorted[:max_words]
    canvas = toyplot.Canvas(width=500, height=3000)
    axes = canvas.cartesian(label="Estimated Prior", margin=100)
    axes.bars(sorted_word_probs, along='y')
    axes.y.ticks.locator = toyplot.locator.Explicit(labels=words_sorted)
    axes.y.ticks.labels.angle = -90
    
    toyplot.browser.show(canvas)

def gaussianSimGraph(cluster_elms):
    def gaussianSim(xi, xj):
        var = 100
        return np.exp((-1 * np.linalg.norm(xi - xj) ** 2) / (2 * var))
    
    sim_graph = np.zeros([cluster_elms.shape[0], cluster_elms.shape[0]])
    for i, j in np.ndindex(sim_graph.shape):
        sim = gaussianSim(cluster_elms[i], cluster_elms[j])
        if sim >= 0:
            sim_graph[i, j] = sim
    return sim_graph

def load_segs_link(seg_results_dir):
    segs_dict = {}
    for res_file in os.listdir(seg_results_dir):
        with open(seg_results_dir+res_file) as f:
            lins = f.readlines()
        domain = eval(lins[-1])[0][:-5]
        seg = eval(lins[-2].replace("Hyp Topics: ", ""))
        segs_dict[domain] = seg
    return segs_dict
    
def eval_pipeline_linking(seg_results_dir, configs_dir):
    print("Spectral Clustering link evaluation")
    all_segs = load_segs_link(seg_results_dir)
    results_dict = {}
    for config in os.listdir(configs_dir):
        
        config_file = configs_dir+config #"../dataset/physics_test.json"
        with open(config_file) as data_file:    
            config = json.load(data_file)
        domain = os.listdir(config["real_data"]["docs_dir"])[0][:-5]
        hyp_topic_seqs = all_segs[domain]
        doc_col = MultiDocument(config)
        data = Data(doc_col)
        num_clusters = doc_col.max_topics
        
        clustering_points = []
        seg_lens = []
        start = 0
        end = 0
        k_prev = hyp_topic_seqs[0]
        for k in hyp_topic_seqs:
            if k_prev != k:
                seg_lens.append(end-start)
                clustering_points.append(np.sum(doc_col.U_W_counts[start:end], axis=0))
                start = end
                k_prev = k
            end += 1
        clustering_points.append(np.sum(doc_col.U_W_counts[start:end], axis=0))
        seg_lens.append(end-start)
        if num_clusters >= len(seg_lens):
            num_clusters = len(seg_lens)-3
        if len(seg_lens) == 1:
            hyp_topics = hyp_topic_seqs
        else:
            sim_graph = gaussianSimGraph(np.array(clustering_points))
            labels = spectral_clustering(sim_graph, n_clusters=num_clusters, eigen_solver='arpack')
            hyp_topics = []
            for seg_len, label in zip(seg_lens, labels):
                hyp_topics += [label]*seg_len
            
        gs_topics = []
        for doc_i in range(data.n_docs):
            gs_topics += data.doc_rho_topics[doc_i]
            
        f1_score = f_measure(gs_topics, hyp_topics)
        acc = accuracy(gs_topics, hyp_topics)
        print("%s\n Ref:%s\nHyp:%s" % (domain, str(gs_topics), str(hyp_topics)))
        results_dict[domain] = {"F1": f1_score, "Acc": acc}
    
    ids = list(results_dict.keys())
    ids.sort()
    print("Domain\tF1\tAcc")
    for domain in ids: 
        print("%s\t%f\t%f" % (domain, results_dict[domain]["F1"], results_dict[domain]["Acc"]))
    
def gen_rnd_baseline(seg):
    shuffle(seg)
    shuffle(seg)
    shuffle(seg)
    return seg.tolist()

def get_topics_baseline(hyp_topics):
    bl_topics = []
    new_k = 0
    k_prev = hyp_topics[0]
    for k in hyp_topics:
        if k != k_prev:
            new_k += 1
        bl_topics.append(new_k)
    return bl_topics

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
    rnd_baselines = [[] for i in range(100)]
    no_segs_baseline = []
    for gs_doc in models[0].data.docs_rho_gs:
        gs_doc_copy = copy.deepcopy(gs_doc)
        gs_segs.append(gs_doc_copy)
        no_segs_baseline += [0]*len(gs_doc_copy)
        for i in range(len(rnd_baselines)):
            gs_doc_copy = copy.deepcopy(gs_doc)
            rnd_baselines[i] += gen_rnd_baseline(gs_doc_copy)
        
        
    for j, gs_seg in enumerate(gs_segs):
        print("GS:\n" + str(gs_seg.tolist()))
        for i in range(len(models)):
            print(str(models_desc[i]) + ":\n" + str(md_segs[i][j]))
        print("")
        
    for i, model in enumerate(models):
        print(str(models_desc[i])+" WD: "+str(wd_evaluator(model.get_all_segmentations(), doc_synth)))
    
    print("Baseline NO segs WD: "+str(wd_evaluator(no_segs_baseline, doc_synth)))
    avg_wd_rnd_bl = np.zeros(models[0].data.n_docs)
    for rnd_baseline in rnd_baselines:
        avg_wd_rnd_bl += np.array(wd_evaluator(rnd_baseline, doc_synth))
    
    avg_wd_rnd_bl = avg_wd_rnd_bl/len(rnd_baselines)
    print("Baseline RND segs WD: "+str(avg_wd_rnd_bl.tolist()))
    
    for i, model in enumerate(models):
        hyp_topics = []
        gs_topics = []
        hyp_topics_print = []
        gs_topics_print = []
        for doc_i in range(model.data.n_docs):
            hyp_topics += model.get_seg_with_topics(doc_i, model.best_segmentation[-1][0][1])
            hyp_topics_print.append(model.get_seg_with_topics(doc_i, model.best_segmentation[-1][0][1]))
            gs_topics += model.data.doc_rho_topics[doc_i]
            gs_topics_print.append(model.data.doc_rho_topics[doc_i])
        f1_score = f_measure(gs_topics, hyp_topics)
        acc = accuracy(gs_topics, hyp_topics)
        
        hyp_all_dif_topics = get_topics_baseline(hyp_topics)
        f1_score_bl = f_measure(gs_topics, hyp_all_dif_topics)
        acc_bl = accuracy(gs_topics, hyp_all_dif_topics)
        print("%s F1: %f F1_bl: %f Acc %f Acc_bl %f\nHyp Topics: %s\nRef Topics: %s" % (str(models_desc[i]),
                                                                                        f1_score,
                                                                                        f1_score_bl,
                                                                                        acc, acc_bl,
                                                                                        str(hyp_topics_print),
                                                                                        str(gs_topics_print)))
    
def single_vs_md_eval(doc_synth, alpha, md_all_combs=True, md_fast=True, print_flag=False):
    '''
    Print the WD results when segmenting single documents
    and all of them simultaneously (multi-doc model)
    :param doc_synth: collection of synthetic documents
    :param alpha: alpha prior vector
    :param print_flag: boolean to print or not the segmentation results
    '''
    single_docs = doc_synth.get_single_docs()
    single_doc_wd = []
    time_wd_results = []
    start = time.time()
    sd_segs = []
    for doc in single_docs:
        data = Data(doc)
        dp_model = sd_seg.SingleDocDPSeg(alpha, single_docs, data)
        dp_model.segment_docs()
        sd_segs.append(dp_model.get_final_segmentation(0))
        single_doc_wd += wd_evaluator(dp_model.get_all_segmentations(), doc)
    end = time.time()
    sd_time = (end - start)
    #single_doc_wd = ['%.3f' % wd for wd in single_doc_wd]
    time_wd_results.append(("SD", sd_time, ['%.3f' % wd for wd in single_doc_wd]))
    
    data = Data(doc_synth)
    if md_all_combs:
        dp_model = dp_seg.MultiDocDPSeg(alpha, data, seg_type=dp_seg.SEG_ALL_COMBS)
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
        dp_model = dp_seg.MultiDocDPSeg(alpha, data, seg_type=dp_seg.SEG_FAST)
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

def incremental_eval(doc_synth, alpha):
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
        sd_results, md_results = single_vs_md_eval(merged_doc_synth, alpha, md_all_combs=False, print_flag=True)
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
    alpha = np.array([0.08]*W)
    n_docs = 2
    doc_len = 20
    pi = 0.2
    sent_len = 6
    n_seg = 3
    doc_synth = CVBSynDoc2(alpha, pi, sent_len, n_seg, n_docs)
    data = Data(doc_synth)
    
    #incremental_eval(doc_synth, alpha)
    #single_vs_md_eval(doc_synth, alpha, md_all_combs=False , md_fast=True, print_flag=True)
    dp_model = dp_seg.MultiDocDPSeg(alpha, data, seg_type=dp_seg.SEG_FAST)
    md_eval(doc_synth, dp_model)
    
def vi_only_test():
    use_seed = True
    seed = 26
    if use_seed:
        np.random.seed(seed)
        
    W = 12
    alpha = np.array([0.08]*W)
    n_docs = 2
    pi = 0.2
    sent_len = 12
    n_seg = 3
    doc_synth = CVBSynDoc2(alpha, pi, sent_len, n_seg, n_docs)
    data = Data(doc_synth)
    
    iters = 20
    vi_model = vi_seg.MultiDocVISeg(alpha, data, max_topics=n_seg, n_iters=iters, log_dir="/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/logs")
    md_eval(doc_synth, [vi_model], ["VI"])
    
def dp_vs_vi():
    use_seed = True
    seed = 26
    if use_seed:
        np.random.seed(seed)
        
    W = 12
    alpha = np.array([0.08]*W)
    n_docs = 2
    pi = 0.2
    sent_len = 12
    n_segs = 3
    doc_synth = CVBSynDoc2(alpha, pi, sent_len, n_segs, n_docs)
    data = Data(doc_synth)
    
    iters = 20
    vi_model = vi_seg.MultiDocVISeg(alpha, data, max_topics=n_segs, n_iters=iters, log_dir="../logs/", log_flag=True)
    dp_model = dp_seg.MultiDocDPSeg(alpha, data, seg_type=dp_seg.SEG_FAST)
    md_eval(doc_synth, [dp_model, vi_model], ["DP", "VI"])
    
def skip_topics_test():
    use_seed = True
    seed = 56#48
    
    if use_seed:
        np.random.seed(seed)
        
    W = 10
    alpha = np.array([0.3]*W)
    n_docs = 3
    pi = 0.25 
    sent_len = 6
    n_segs = 3
    n_topics = 5
    
    skip_topics_syn = CVBSynSkipTopics(alpha, pi, sent_len, n_segs, n_docs, n_topics)
    data = Data(skip_topics_syn)
    
    single_docs = skip_topics_syn.get_single_docs()
    sd_model = sd_seg.SingleDocDPSeg(alpha, single_docs, data)
    
    vi_dp_config = {"type": vi_seg.DP_VI_SEG, "seg_func": vi_seg.QZ_VOTING}
    vi_dp_config_v2 = {"type": vi_seg.DP_VI_SEG, "seg_func": vi_seg.QZ_VOTING_V2}
    vi_dp_qz_ll_config = {"type": vi_seg.DP_VI_SEG, "seg_func": vi_seg.QZ_LL}
    vi_config = {"type":vi_seg.VI_SEG}
    
    vi_dp_qz_voting_model = vi_seg.MultiDocVISeg(alpha,\
                                                 data,\
                                                 max_topics=n_topics,\
                                                 n_iters=20,\
                                                 seg_config=vi_dp_config,\
                                                 log_dir="../logs/",\
                                                 log_flag=True)
    
    vi_dp_qz_voting_model_v2 = vi_seg.MultiDocVISeg(alpha,\
                                                    data,\
                                                    max_topics=n_topics,\
                                                    n_iters=20,\
                                                    seg_config=vi_dp_config_v2,\
                                                    log_dir="../logs/",\
                                                    log_flag=True)
    
    vi_dp_qz_ll_model = vi_seg.MultiDocVISeg(alpha, data, max_topics=n_topics,\
                                             seg_dur=1.0/pi,\
                                             std=3.0,\
                                             use_prior=True,
                                             n_iters=5,\
                                             seg_config=vi_dp_qz_ll_config,\
                                             log_dir="../logs/", log_flag=True)
    
    vi_model = vi_seg.MultiDocVISeg(alpha,\
                                    data,\
                                    max_topics=n_topics,\
                                    n_iters=40,\
                                    seg_config=vi_config,\
                                    log_dir="../logs/",\
                                    log_flag=True)
    
    greedy_model_no_prior = greedy_seg.MultiDocGreedySeg(alpha, data, max_topics=n_topics, seg_dur=1.0/pi, std=2.0, use_prior=False)
    greedy_model_std1 = greedy_seg.MultiDocGreedySeg(alpha, data, max_topics=n_topics, seg_dur=1.0/pi, std=1.5, use_prior=True)
    greedy_model_std2 = greedy_seg.MultiDocGreedySeg(alpha, data, max_topics=n_topics, seg_dur=1.0/pi, std=2.0, use_prior=True)
    greedy_model_std3 = greedy_seg.MultiDocGreedySeg(alpha, data, max_topics=n_topics, seg_dur=1.0/pi, std=3.0, use_prior=True)
    mcmc_model = mcmc_seg.MultiDocMCMCSeg(alpha, data, max_topics=n_topics, seg_dur=1.0/pi, std=3.0, use_prior=False)
    mcmc_model_v2 = mcmc_seg_v2.MultiDocMCMCSegV2(alpha, data, max_topics=n_topics, seg_dur=1.0/pi, std=3.0, use_prior=False)
    dp_model = dp_seg.MultiDocDPSeg(alpha, data, max_topics=n_topics, seg_type=dp_seg.SEG_SKIP_K)
    dp_model_sc = dp_seg_sc.MultiDocDPSeg(alpha, data, max_topics=n_topics, seg_type=dp_seg.SEG_SKIP_K)
    #md_eval(skip_topics_syn, [greedy_model_std3, mcmc_model, mcmc_model_std3], ["GS3", "MC ", "MCP"])
    md_eval(skip_topics_syn, [mcmc_model_v2], ["MC2"])
    
def skip_topics_incremental_test():
    use_seed = True
    seed = 229#84
    if use_seed:
        np.random.seed(seed)
        
    W = 100#100
    alpha = np.array([0.3]*W)
    n_docs = 15
    pi = 0.25
    sent_len = 6
    n_segs = 5
    n_topics = 7
    
    skip_topics_syn = CVBSynSkipTopics(alpha, pi, sent_len, n_segs, n_docs, n_topics)
    single_docs = skip_topics_syn.get_single_docs()
    results_dict = dict([(key, []) for key in range(n_docs)])
    
    sd_model = sd_seg.SingleDocDPSeg(alpha, single_docs, Data(skip_topics_syn))
    sd_model.segment_docs()
    sd_wd_results = wd_evaluator(sd_model.get_all_segmentations(), skip_topics_syn)
        
        
    for i in range(1, skip_topics_syn.n_docs+1):
        target_docs = single_docs[:i]
        merged_doc_synth = merge_docs(target_docs)
        data = Data(merged_doc_synth)
        greedy_model = greedy_seg.MultiDocGreedySeg(alpha, data, max_topics=n_topics, seg_dur=1.0/pi)
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
        
def profile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

def get_seg_desc(config_inst):    
    desc = "sf: "+str(config_inst["seg_func"])+\
           " mt: "+str(config_inst["max_topics"])+\
           " beta: "+str(config_inst["beta"][0])+\
           " msl: "+str(config_inst["max_seg_len"])+\
           " ws: "+str(config_inst["window_size"])+\
           " mc: "+str(config_inst["max_cache"])+\
           " cp: "+str(config_inst["cache_prune"])+\
           " fc: "+str(config_inst["flush_cache_flag"])+\
           " sf: "+str(config_inst["slack_flag"])+\
           " ts: "+str(config_inst["topic_slack"])+\
           " pc: "+'"'+str(config_inst["prior_class"])+'"'+\
           " pt: "+'"'+str(config_inst["prior_type"])+'"'
    if "config" == config_inst["prior_type"]:
        desc += " vals: "+str(config_inst["dur_prior_vals"])
    return desc

def get_all_greedy_configs(base_config):
    expand_keys = []
    expand_values = []
    all_configs = []
    for key in base_config:
        if isinstance(base_config[key],(list,)): 
            expand_keys.append(key)
            expand_values.append(base_config[key])
    
    if len(expand_keys) == 0:
        return [base_config]
    
    for val_comb in list(product(*expand_values)):
        config_instance = copy.deepcopy(base_config)
        for i, key in enumerate(expand_keys):
            config_instance[key] = val_comb[i]
        all_configs.append(config_instance)
    return all_configs

#@profile
def real_dataset_tests(data_config, greedy_seg_config):
    doc_col = MultiDocument(data_config)
    data = Data(doc_col)
    single_docs = doc_col.get_single_docs()
    alpha_tt_t0_test = [0.2, 3, 8, 10, 20, 80, 100]
    sd_seg_config = {"max_topics": doc_col.max_topics,\
                         "max_cache": 10,
                         "seg_type": None,
                         "beta": np.array([0.8]*doc_col.W),\
                         "use_dur_prior": False,
                         "seg_dur_prior": "indv",\
                         "seg_func": SEG_BL,\
                         "alpha_tt_t0": 4,\
                         "phi_log_dir": "../logs/phi"}
    all_configs = get_all_greedy_configs(greedy_seg_config)
    if greedy_seg_config["run_parallel"]:
        import ray
        ray.init(redirect_output=True)
    #single_docs = doc_col.get_single_docs()
    models = []
    models_names = []
    betas = [.8]
    windows = [50]
    run_MD = True
    run_SD = False
    for config_inst in all_configs:
        beta = config_inst["beta"]
        config_inst["max_topics"] += doc_col.max_topics
        if run_MD:
            if config_inst["seg_func"] == SEG_TT:
                config_inst["beta"] = np.array([beta]*doc_col.W)
            else:
                config_inst["beta"] = np.array([beta]*doc_col.W)#hyper_param_opt(doc_col, n=beta)
            greedy_model = greedy_seg.MultiDocGreedySeg(data, seg_config=config_inst)
            #greedy_model = dp_seg.MultiDocDPSeg(data, greedy_seg_config)
            #greedy_model = mcmc_seg_v2.MultiDocMCMCSegV2(data, greedy_seg_config)
            seg_desc = get_seg_desc(config_inst)
            models_names.append(seg_desc)
            models.append(greedy_model)
        '''
        if run_SD:
            sd_seg_config["beta"] = np.array([beta]*doc_col.W)#hyper_param_opt(doc_col, n=beta)
            sd_model = sd_seg.SingleDocDPSeg(single_docs, data, seg_config=sd_seg_config)
            models_names += [sd_model.desc+str(beta)]
            models += [sd_model]
        '''
    md_eval(doc_col, models, models_names)
    print(doc_col.doc_names)
    #plot_topics(models[1].best_segmentation[-1][0][1], doc_col.inv_vocab)
        
if __name__ == "__main__":
    if len(sys.argv) == 1:
        data_config = "../config/physics_test.json"
        greedy_seg_config = "../config/greedy_config.json"
    else:
        data_config = sys.argv[1]
        greedy_seg_config = sys.argv[2]
    with open(data_config) as data_file:    
        data_config = json.load(data_file)
        
    with open(greedy_seg_config) as seg_file:    
        greedy_seg_config = json.load(seg_file)
        
    #skip_topics_test()
    real_dataset_tests(data_config, greedy_seg_config)
    #eval_pipeline_linking("/home/pjdrm/eclipse-workspace/SegmentationScripts/all_results/", "/home/pjdrm/eclipse-workspace/SegmentationScripts/configs/")

