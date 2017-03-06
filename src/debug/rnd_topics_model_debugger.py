'''
Created on Jan 30, 2017

This is tests the initialization
of the topic tracking model.

@author: root
'''
from dataset.synthetic_doc import SyntheticRndTopicPropsDoc, SyntheticRndTopicDocCollection
from model.rnd_topics_segmentor import RndTopicsModel, RndTopicsCacheModel
import debug.debug_tools as debug_tools
from debug.debug_tools import print_ref_hyp_plots, debug_topic_assign, plot_log_joint_prob, plot_rho_u_prob
from dataset.real_doc import Document
from multiprocessing.dummy import Pool as ThreadPool
import time
import shutil
import os
        
def plot_debug_results(wd_samp_res):
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
    pi = 0.2
    alpha = 0.1
    beta = 0.1
    K = 4
    W = 4
    n_sents = 20
    sentence_l = 15
    log_flag = False
    
    doc_synth_tt = SyntheticRndTopicPropsDoc(pi, alpha, beta, K, W, n_sents, sentence_l)
    doc_synth_tt.generate_doc()
    gamma = 10
    rnd_topics_model = RndTopicsModel(gamma, alpha, beta, K, doc_synth_tt, log_flag)
    
    outFile = "debug/rnd_topics_model/rnd_topics_theta_heat_map_initial.png"
    '''
    debug_tools.test_model_state(rnd_topics_model, outFile)
    debug_tools.test_z_ui_sampling(rnd_topics_model)
    debug_tools.test_Z_sampling(rnd_topics_model, outFile)
    debug_tools.test_rho_sampling(rnd_topics_model, outFile)
    '''
    
    sampler_log_file = "logging/Sampler.log"
    n_iter = 500
    burn_in = 0
    lag = 0
    debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, sampler_log_file)
    
    plot_dir = "debug/rnd_topics_model/ref_hyp_plots/"
    print_ref_hyp_plots(sampler_log_file, plot_dir, "RndTopics")
    
def syn_multi_doc_test():
    pi = 0.2
    alpha = 0.1
    beta = 0.1
    K = 4
    W = 4
    n_docs = 3
    n_sents = 20
    sentence_l = 15
    log_flag = False
    
    doc_synth_tt = SyntheticRndTopicDocCollection(pi, alpha, beta, K, W, n_sents, sentence_l, n_docs)
    doc_synth_tt.generate_docs()
    gamma = 10
    rnd_topics_model = RndTopicsModel(gamma, alpha, beta, K, doc_synth_tt, log_flag)
    
    sampler_log_file = "logging/Sampler.log"
    n_iter = 500
    burn_in = 0
    lag = 0
    debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, sampler_log_file)
        
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
    
    return debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, sampler_log_file)

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
        
def run_seq_sampler(doc, n_samplers, log_dir, rt_args, gs_args, debug_plot_flag=True):
    wd_samp_res = []
    for i in range(n_samplers):
        sampler_log_file = log_dir + "S" + str(i) + ".log"
        model_log_file = log_dir + "RTModel"+str(i)+".log"
        wd, sampler = real_doc_test(doc, sampler_log_file, model_log_file, rt_args, gs_args)
        wd_samp_res.append((wd, sampler))
    print_wd_results(wd_samp_res)
    if debug_plot_flag:
        plot_debug_results(wd_samp_res)

file_path = "data/L02_vref_small.txt"
gs_Z_fp = None #"data/L02_vref_small_Z.txt"
max_features = 200
doc = Document(file_path, max_features, lemmatize=False, gs_Z_file_path = gs_Z_fp)
n_samplers = 5

alpha = 0.1
beta = 0.1
K = 2
log_flag = True
gamma = 10
rt_args = [alpha, beta, K, log_flag, gamma]

n_iter = 1000
burn_in = 0
lag = 0
gs_args = [n_iter, burn_in, lag]

k_W_counts_outDir = "./debug/rnd_topics_model/k_word_counts/"
rho1_prob_dir = "./debug/rnd_topics_model/rho_prob/"
log_dir = "./logging/"
shutil.rmtree(k_W_counts_outDir)
shutil.rmtree(log_dir)
shutil.rmtree(rho1_prob_dir)
os.makedirs(k_W_counts_outDir)
os.makedirs(log_dir)
os.makedirs(rho1_prob_dir)

run_seq_sampler(doc, n_samplers, log_dir, rt_args, gs_args)
plot_log_joint_prob(log_dir, "./debug/rnd_topics_model/samplers_convergence")

#yn_multi_doc_test()