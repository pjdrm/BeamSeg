'''
Created on Jan 30, 2017

This is tests the initialization
of the topic tracking model.

@author: root
'''

import model.rnd_topics_segmentor as seg
import dataset.synthetic_doc as syn_doc
import debug.debug_tools as debug_tools
from dataset.real_doc import Document, MultiDocument
from multiprocessing.dummy import Pool as ThreadPool
import time
import shutil
import os
import numpy as np
import sys
import json

rnd_scan_order = "RndScanOrderModel"
rnd_tm_cache = "RndTopicsCacheModel"
rnd_tm_sd = "RndTopicsSingleDocModel"

def plot_debug_results(wd_samp_res, k_W_counts_outDir, rho1_prob_dir, log_dir):
    for wd, sampler in wd_samp_res:
        wd = "%.2f" % wd
        Sampler_id = sampler.sampler_log_file.split("/")[-1][1:-4]
        sampler_outDir = k_W_counts_outDir+\
                        "S"+Sampler_id+\
                        "_"+wd+"/"
        os.makedirs(sampler_outDir)
        rho1_model_dir = rho1_prob_dir + "SM"+str(Sampler_id) + "/"
        os.makedirs(rho1_model_dir)
        debug_tools.debug_topic_assign(sampler, sampler_outDir)
        model_log_file = log_dir + "RTModel"+str(Sampler_id)+".log"
        #plot_rho_u_prob(model_log_file, rho1_model_dir)
        
def print_wd_results(wd_samp_res):
    wd_res = ["%.2f" % wd for wd, samp in wd_samp_res]
    print(wd_res)
        
def syn_multi_doc_test(configs):
    debug_tools.clean_log()
    
    log_flag = False
    
    doc_synth_tt = syn_doc.SyntheticRndTopicMultiDoc(configs)
    doc_synth_tt.generate_docs()
    results_str = ""
    y_labels_lp = []
    md_log_file_list = []
    
    if rnd_scan_order in configs["run_models"]:
        rnd_topics_model = seg.RndScanOrderModel(configs, doc_synth_tt, log_flag)
        md_log_file = configs["logging"]["MUSE-RndScan"]
        wd_md_doc_results, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, configs, md_log_file)
        results_str += "MUSE-RndScan seg results:\nAVG: " + str(np.mean(wd_md_doc_results)) + "\nAll: " + str(wd_md_doc_results) + "\n\n"
        y_labels_lp.append("MUSE-RndScan")
        md_log_file_list.append(md_log_file)
    
    if rnd_tm_cache in configs["run_models"]:
        rnd_topics_model = seg.RndTopicsCacheModel(configs, doc_synth_tt, log_flag)
        md_log_file_cache = configs["logging"]["MUSE-C"]
        wd_md_doc_results_cache, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, configs, md_log_file_cache)
        results_str += "MUSE-Cache seg results:\nAVG: " + str(np.mean(wd_md_doc_results_cache)) + "\nAll: " + str(wd_md_doc_results_cache) + "\n\n"
        y_labels_lp.append("MUSE-C")
        md_log_file_list.append(md_log_file_cache)
    
    ind_log_file_list = []
    if rnd_tm_sd in configs["run_models"]:
        wd_indv_doc_results = []
        indv_docs = syn_doc.multi_doc_slicer(doc_synth_tt)
        for i, doc in enumerate(indv_docs):
            sampler_log_file = "logging/Sampler" + str(i) + "_indv.log"
            ind_log_file_list.append(sampler_log_file)
            rnd_topics_model = seg.RndTopicsCacheModel(configs, doc, log_flag)
            wd, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, configs, sampler_log_file)
            wd_indv_doc_results += wd
        results_str += "INDV doc seg results:\nAVG: " + str(np.mean(wd_indv_doc_results)) + "\nAll: " + str(wd_indv_doc_results)
        y_labels_lp.append("PLDA")
        
    print(results_str)
    with open(configs["logging"]["results_file"], "w+") as f:
        f.write(results_str)
        
    debug_tools.plot_log_joint_prob_md(md_log_file_list, ind_log_file_list, y_labels_lp, configs["plots"]["log_joint"])
    debug_tools.plot_iter_time(md_log_file_list + ind_log_file_list, configs["plots"]["iter_time"])
    
    
def real_multi_doc_test(configs):
    debug_tools.clean_debug()
    debug_tools.clean_log()
    if os.path.isfile(configs["logging"]["results_file"]):
        os.remove(configs["logging"]["results_file"])
    doc_col = MultiDocument(configs)
    indv_docs = syn_doc.multi_doc_slicer(doc_col)
    
    log_flag = False
    str_res = ""
    
    for K in configs["model"]["K"]:
        configs["model"]["K"] = K
        y_labels = []
        all_wd_results = []
        md_log_file_list = []
        
        if rnd_scan_order in configs["run_models"]:
            rnd_topics_model = seg.RndScanOrderModel(configs, doc_col, log_flag)
            md_log_file = configs["logging"]["MUSE-RndScan"] + "_K" + str(K) + ".log"
            wd_md_doc_results, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, configs, md_log_file)
            y_labels.append("MUSE")
            all_wd_results.append(wd_md_doc_results)
            md_log_file_list.append(md_log_file)
            
        if rnd_tm_cache in configs["run_models"]:
            rnd_topics_model = seg.RndTopicsCacheModel(configs, doc_col, log_flag)
            md_log_file_cache = configs["logging"]["MUSE-C"] + "_K" + str(K) + ".log"
            wd_md_doc_results, sampler_cache = debug_tools.run_gibbs_sampler(rnd_topics_model, configs, md_log_file_cache)
            y_labels.append("MUSE-C")
            all_wd_results.append(wd_md_doc_results)
            md_log_file_list.append(md_log_file_cache)
        
        if rnd_tm_sd in configs["run_models"]:
            wd_indv_doc_results = []
            ind_log_file_list = []
            y_labels.append("PLDA")
            for i, doc in enumerate(indv_docs):
                sampler_log_file = configs["logging"]["PLDA"] + "_" + str(i) + "_K" + str(K) + "_indv.log"
                ind_log_file_list.append(sampler_log_file)
                rnd_topics_model = seg.RndTopicsCacheModel(configs, doc, log_flag)
                wd, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, configs, sampler_log_file)
                wd_indv_doc_results += wd
            all_wd_results.append(wd_indv_doc_results)
            
        str_res += "K " + str(K) + "\n\t\t\t"
        for label in y_labels:
            str_res += label + "\t\t"
        str_res += "\n"
        for i, doc in enumerate(doc_col.doc_names):
            str_res += doc + "\t\t" 
            for j in range(len(all_wd_results)):
                str_res += str(all_wd_results[j][i]) + "\t"
            str_res += "\n"
        
        if eval(configs["plots"]["flag"]):
            debug_tools.plot_log_joint_prob_md(md_log_file_list, ind_log_file_list, y_labels, configs["plots"]["log_joint"]+"_"+str(K))
            debug_tools.plot_iter_time(md_log_file_list, configs["plots"]["iter_time"]+"_"+str(K)+".png")
            for log_file in md_log_file_list + ind_log_file_list:
                debug_tools.print_ref_hyp_plots(log_file, configs["plots"]["ref_hyp"], log_file.split("/")[-1][:-4])
    for i, label in enumerate(y_labels):
        str_res += label+" WD avg: "+str(np.average(all_wd_results[i]))+"\n"
    print(str_res)                
    with open(configs["logging"]["results_file"], "a") as f:
        f.write(str_res)
    
    '''
    k_W_counts_outDir = "./debug/rnd_topics_model/k_word_counts/"
    rho1_prob_dir = "./debug/rnd_topics_model/rho_prob/"
    log_dir = "./logging/"
    
    plot_debug_results([(np.sum(wd_md_doc_results)/len(wd_md_doc_results), sampler_cache)], k_W_counts_outDir, rho1_prob_dir, log_dir)
    '''
    
def syn_ditto_doc_test():
    debug_tools.clean_log()
    pi = 0.02
    alpha = 0.7
    alpha = 0.7
    K = 6
    W = 150
    n_sents = 300
    sentence_l = 5
    log_flag = False
    
    doc_synth_tt = syn_doc.SyntheticRndTopicPropsDoc(pi, alpha, alpha, K, W, n_sents, sentence_l)
    doc_synth_tt.generate_doc()
    
    n_copies = 4
    ditto_doc = syn_doc.SyntheticDittoDocs(doc_synth_tt, n_copies)
    
    gamma = 10
    n_iter = 3000
    burn_in = 0
    lag = 0
    
    rnd_topics_model = seg.RndTopicsCacheModel(gamma, alpha, alpha, K, ditto_doc, log_flag)
    md_log_file_cache = "logging/Sampler_MD_cache.log"
    wd_md_doc_results_cache, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, md_log_file_cache)
    
    sampler_log_file = "logging/Sampler_indv_cache.log"
    wd_indv_doc_results = []
    ind_log_file_list = []
    for i in range(n_copies):
        sampler_log_file = "logging/Sampler" + str(i) + "_indv.log"
        ind_log_file_list.append(sampler_log_file)
        rnd_topics_model = seg.RndTopicsCacheModel(gamma, alpha, alpha, K, doc_synth_tt, log_flag)
        wd, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, sampler_log_file)
        wd_indv_doc_results += wd
    
    debug_tools.plot_log_joint_prob_md(md_log_file_cache, ind_log_file_list, "./debug/rnd_topics_model/samplers_convergence")    
    print("INDV doc seg results: %s" % (str(wd_indv_doc_results)))
    print("Multi-Doc seg results: %s" % (str(wd_md_doc_results_cache)))

if __name__ == '__main__':
    if len(sys.argv) == 1:
        data_config = 'config.json'
    else:
        data_config = sys.argv[1]
    with open(data_config) as data_file:    
        config = json.load(data_file)
        
    test_func = locals()[config["test"]]
    test_func(config)

#ind_log_file_list = [os.path.join("/home/pjdrm/workspace/TopicTrackingSegmentation/debug/rnd_topics_model/10000it_50d_100sents_no_cache_test/logging/", x) for x in os.listdir("/home/pjdrm/workspace/TopicTrackingSegmentation/debug/rnd_topics_model/10000it_50d_100sents_no_cache_test/logging") if "indv" in x]
#plot_log_joint_prob_md(["/home/pjdrm/workspace/TopicTrackingSegmentation/debug/rnd_topics_model/10000it_50d_100sents_no_cache_test/logging/Sampler_MD.log", "/home/pjdrm/workspace/TopicTrackingSegmentation/debug/rnd_topics_model/10000it_50d_100sents_no_cache_test/logging/Sampler_MD_cache.log"], ind_log_file_list, "./debug/rnd_topics_model/samplers_convergence")
#plot_iter_time(["/home/pjdrm/workspace/TopicTrackingSegmentation/debug/rnd_topics_model/Sampler_MD_K10_real_NO_cache.log", "/home/pjdrm/workspace/TopicTrackingSegmentation/debug/rnd_topics_model/Sampler_MD_K10_real_cache.log"], "./debug/rnd_topics_model/time_iter_cikm.png")

#plot_topic_wd_res("/home/pjdrm/workspace/TopicTrackingSegmentation/debug/rnd_topics_model/avl_exp/50000it_avl_exps/", "/home/pjdrm/workspace/TopicTrackingSegmentation/debug/rnd_topics_model/avl_exp/50000it_avl_exps/k_wd.png")
#print_ref_hyp_plots_all_models("tmp_rot_wiki.log", "./")

#ind_log_file_list = [os.path.join("/home/pjdrm/workspace/TopicTrackingSegmentation/debug/rnd_topics_model/exp5_50d_10000it/logging", x) for x in os.listdir("/home/pjdrm/workspace/TopicTrackingSegmentation/debug/rnd_topics_model/exp5_50d_10000it/logging") ]
#plot_log_joint_prob_md(["/home/pjdrm/workspace/TopicTrackingSegmentation/debug/rnd_topics_model/exp5_50d_10000it/Sampler_MD_cache.log"], ind_log_file_list, "./debug/rnd_topics_model/samplers_convergence")
        
    