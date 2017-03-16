'''
Created on Jan 30, 2017

This is tests the initialization
of the topic tracking model.

@author: root
'''
from dataset.synthetic_doc import SyntheticRndTopicPropsDoc, SyntheticRndTopicMultiDoc, multi_doc_slicer, SyntheticDittoDocs
from model.rnd_topics_segmentor import RndTopicsModel, RndTopicsCacheModel, RndTopicsParallelModel
import debug.debug_tools as debug_tools
from debug.debug_tools import print_ref_hyp_plots, debug_topic_assign, plot_log_joint_prob, plot_log_joint_prob_md, plot_rho_u_prob, plot_iter_time, clean_debug, clean_log
from dataset.real_doc import Document, MultiDocument
from multiprocessing.dummy import Pool as ThreadPool
import time
import shutil
import os
        
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
        debug_topic_assign(sampler, sampler_outDir)
        model_log_file = log_dir + "RTModel"+str(Sampler_id)+".log"
        #plot_rho_u_prob(model_log_file, rho1_model_dir)
        
def print_wd_results(wd_samp_res):
    wd_res = ["%.2f" % wd for wd, samp in wd_samp_res]
    print(wd_res)
        
def syn_doc_test():
    pi = 0.02
    alpha = 0.7
    beta = 0.7
    K = 6
    W = 100
    n_sents = 470
    sentence_l = 7
    log_flag = False
    
    doc_synth_tt = SyntheticRndTopicPropsDoc(pi, alpha, beta, K, W, n_sents, sentence_l)
    doc_synth_tt.generate_doc()
    gamma = 10
    rnd_topics_model = RndTopicsCacheModel(gamma, alpha, beta, K, doc_synth_tt, log_flag)
    
    outFile = "debug/rnd_topics_model/rnd_topics_theta_heat_map_initial.png"
    '''
    debug_tools.test_model_state(rnd_topics_model, outFile)
    debug_tools.test_z_ui_sampling(rnd_topics_model)
    debug_tools.test_Z_sampling(rnd_topics_model, outFile)
    debug_tools.test_rho_sampling(rnd_topics_model, outFile)
    '''
    
    sampler_log_file = "logging/Sampler.log"
    n_iter = 5000
    burn_in = 0
    lag = 0
    debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, sampler_log_file)
    
    #plot_dir = "debug/rnd_topics_model/ref_hyp_plots/"
    #print_ref_hyp_plots(sampler_log_file, plot_dir, "RndTopics")
    
def syn_multi_doc_test():
    clean_log()
    pi = 0.07
    alpha = 0.07
    beta = 0.07
    K = 3
    W = 10
    n_sents = 50
    sentence_l = 20
    n_docs = 5
    log_flag = False
    
    doc_synth_tt = SyntheticRndTopicMultiDoc(pi, alpha, beta, K, W, n_sents, sentence_l, n_docs)
    doc_synth_tt.generate_docs()
    indv_docs = multi_doc_slicer(doc_synth_tt)
    gamma = 10
    
    n_iter = 10
    burn_in = 0
    lag = 0
    
    rnd_topics_model = RndTopicsModel(gamma, alpha, beta, K, doc_synth_tt, log_flag)
    md_log_file = "logging/Sampler_MD.log"
    wd_md_doc_results, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, md_log_file)
    
    rnd_topics_model = RndTopicsCacheModel(gamma, alpha, beta, K, doc_synth_tt, log_flag)
    md_log_file_cache = "logging/Sampler_MD_cache.log"
    wd_md_doc_results, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, md_log_file_cache)
    
    '''
    rt_model_parallel = RndTopicsParallelModel(gamma, alpha, beta, K, doc_synth_tt, log_flag)
    md_log_file_parallel = "logging/Sampler_MD_Parallel.log"
    wd_md_doc_results_parallel, sampler = debug_tools.run_gibbs_sampler(rt_model_parallel, n_iter, burn_in, lag, md_log_file_parallel)
    '''
    
    wd_indv_doc_results = []
    ind_log_file_list = []
    for i, doc in enumerate(indv_docs):
        sampler_log_file = "logging/Sampler" + str(i) + "_indv.log"
        ind_log_file_list.append(sampler_log_file)
        rnd_topics_model = RndTopicsCacheModel(gamma, alpha, beta, K, doc, log_flag)
        wd, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, sampler_log_file)
        wd_indv_doc_results += wd
        
    print("INDV doc seg results: %s" % (str(wd_indv_doc_results)))
    print("Multi-Doc seg results: %s" % (str(wd_md_doc_results)))
    #print("Multi-Doc-Para seg results: %s" % (str(wd_md_doc_results_parallel)))
    plot_log_joint_prob_md([md_log_file_cache], ind_log_file_list, "./debug/rnd_topics_model/samplers_convergence")
    plot_iter_time([md_log_file, md_log_file_cache] + ind_log_file_list, "./debug/rnd_topics_model/time_iter.png")
    
def real_multi_doc_test():
    clean_log()
    dc_dir = "data/avl"
    max_features = 200
    doc_col = MultiDocument(dc_dir, max_features, lemmatize=False)
    indv_docs = multi_doc_slicer(doc_col)
    
    alpha = 0.07
    beta = 0.07
    K = 3
    log_flag = False
    gamma = 10
    
    n_iter = 10
    burn_in = 0
    lag = 0
    
    '''
    rnd_topics_model = RndTopicsModel(gamma, alpha, beta, K, doc_col, log_flag)
    md_log_file = "logging/Sampler_MD.log"
    wd_md_doc_results, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, md_log_file)
    '''
    
    rnd_topics_model = RndTopicsCacheModel(gamma, alpha, beta, K, doc_col, log_flag)
    md_log_file_cache = "logging/Sampler_MD_real_cache.log"
    wd_md_doc_results, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, md_log_file_cache)
    
    wd_indv_doc_results = []
    ind_log_file_list = []
    for i, doc in enumerate(indv_docs):
        sampler_log_file = "logging/Sampler" + str(i) + "_indv.log"
        ind_log_file_list.append(sampler_log_file)
        rnd_topics_model = RndTopicsCacheModel(gamma, alpha, beta, K, doc, log_flag)
        wd, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, sampler_log_file)
        wd_indv_doc_results += wd
        
    print("INDV doc seg results: %s" % (str(wd_indv_doc_results)))
    print("Multi-Doc seg results: %s" % (str(wd_md_doc_results)))
    plot_log_joint_prob_md([md_log_file_cache], ind_log_file_list, "./debug/rnd_topics_model/samplers_convergence")
    #plot_iter_time([md_log_file, md_log_file_cache] + ind_log_file_list, "./debug/rnd_topics_model/time_iter.png")
    
def syn_ditto_doc_test():
    clean_log()
    pi = 0.02
    alpha = 0.7
    beta = 0.7
    K = 6
    W = 150
    n_sents = 300
    sentence_l = 5
    log_flag = False
    
    doc_synth_tt = SyntheticRndTopicPropsDoc(pi, alpha, beta, K, W, n_sents, sentence_l)
    doc_synth_tt.generate_doc()
    
    n_copies = 4
    ditto_doc = SyntheticDittoDocs(doc_synth_tt, n_copies)
    
    gamma = 10
    n_iter = 3000
    burn_in = 0
    lag = 0
    
    rnd_topics_model = RndTopicsCacheModel(gamma, alpha, beta, K, ditto_doc, log_flag)
    md_log_file_cache = "logging/Sampler_MD_cache.log"
    wd_md_doc_results_cache, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, md_log_file_cache)
    
    sampler_log_file = "logging/Sampler_indv_cache.log"
    wd_indv_doc_results = []
    ind_log_file_list = []
    for i in range(n_copies):
        sampler_log_file = "logging/Sampler" + str(i) + "_indv.log"
        ind_log_file_list.append(sampler_log_file)
        rnd_topics_model = RndTopicsCacheModel(gamma, alpha, beta, K, doc_synth_tt, log_flag)
        wd, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, sampler_log_file)
        wd_indv_doc_results += wd
    
    plot_log_joint_prob_md(md_log_file_cache, ind_log_file_list, "./debug/rnd_topics_model/samplers_convergence")    
    print("INDV doc seg results: %s" % (str(wd_indv_doc_results)))
    print("Multi-Doc seg results: %s" % (str(wd_md_doc_results_cache)))
    
        
def real_doc_test(doc, sampler_log_file, model_log_file, rt_args, gs_args):
    alpha = rt_args[0]
    beta = rt_args[1]
    K = rt_args[2]
    log_flag = rt_args[3]
    gamma = rt_args[4]
    rnd_topics_model = RndTopicsCacheModel(gamma, alpha, beta, K, doc, log_flag, model_log_file)
    
    n_iter = gs_args[0]
    burn_in = gs_args[1]
    lag = gs_args[2]
    
    wd, sampler = debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, sampler_log_file)
    return wd[0], sampler

def run_paralel_samplers(doc, n_samplers, log_dir, rt_args, gs_args, debug_plot_flag=True):
    pool = ThreadPool(3)
    args = []
    for s in range(n_samplers):
        sampler_log_file = log_dir + "S" + str(s) + ".log"
        model_log_file = log_dir+"RTModel"+str(s)+".log"
        args.append((doc, sampler_log_file, model_log_file, rt_args, gs_args))
    wd_samp_res = pool.starmap(real_doc_test, args)
    pool.close() 
    pool.join()
    print_wd_results(wd_samp_res)
    if debug_plot_flag:
        plot_debug_results(wd_samp_res)
        
def run_seq_sampler(doc, n_samplers, log_dir, rt_args, gs_args):
    wd_samp_res = []
    samp_log_list = []
    for i in range(n_samplers):
        sampler_log_file = log_dir + "S" + str(i) + ".log"
        samp_log_list.append(sampler_log_file)
        model_log_file = log_dir + "RTModel"+str(i)+".log"
        wd, sampler = real_doc_test(doc, sampler_log_file, model_log_file, rt_args, gs_args)
        wd_samp_res.append((wd, sampler))
    print_wd_results(wd_samp_res)
    return samp_log_list, wd_samp_res

def run_real_doc_test():
    file_path = "data/L02_vref_small.txt"
    gs_Z_fp = None #"data/L02_vref_small_Z.txt"
    max_features = 200
    doc = Document(file_path, max_features, lemmatize=False, gs_Z_file_path = gs_Z_fp)
    n_samplers = 3
    
    alpha = 0.1
    beta = 0.1
    K = 2
    log_flag = True
    gamma = 10
    rt_args = [alpha, beta, K, log_flag, gamma]
    
    n_iter = 200
    burn_in = 0
    lag = 0
    gs_args = [n_iter, burn_in, lag]
    
    clean_debug()
    clean_log()
    
    k_W_counts_outDir = "./debug/rnd_topics_model/k_word_counts/"
    rho1_prob_dir = "./debug/rnd_topics_model/rho_prob/"
    log_dir = "./logging/"
    
    samp_log_list, wd_samp_res = run_seq_sampler(doc, n_samplers, log_dir, rt_args, gs_args)
    plot_debug_results(wd_samp_res, k_W_counts_outDir, rho1_prob_dir, log_dir)
    plot_log_joint_prob(samp_log_list, "./debug/rnd_topics_model/samplers_convergence")

#run_real_doc_test()
#syn_doc_test()
#syn_multi_doc_test()
#syn_ditto_doc_test()
real_multi_doc_test()