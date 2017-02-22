'''
Created on Jan 30, 2017

This is tests the initialization
of the topic tracking model.

@author: root
'''
from dataset.synthetic_doc import SyntheticRndTopicPropsDoc
from model.rnd_topics_segmentor import RndTopicsModel
import debug.debug_tools  as debug_tools
from debug.debug_tools import print_ref_hyp_plots, debug_topic_assign
from dataset.real_doc import Document
from multiprocessing.dummy import Pool as ThreadPool
import time
import shutil
import os
        
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
    
    log_file = "logging/Sampler.log"
    n_iter = 200
    burn_in = 0
    lag = 0
    debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, log_file)
    
    plot_dir = "debug/rnd_topics_model/ref_hyp_plots/"
    print_ref_hyp_plots(log_file, plot_dir, "RndTopics")
    
def real_doc_test(doc, log_file):
    alpha = 0.1
    beta = 0.1
    K = 3
    log_flag = False
    
    gamma = 10
    rnd_topics_model = RndTopicsModel(gamma, alpha, beta, K, doc, log_flag)
    
    n_iter = 50
    burn_in = 0
    lag = 0
    
    return debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, log_file)

def run_paralel_samplers(doc, n_samplers):
    pool = ThreadPool(3)
    args = []
    for s in range(n_samplers):
        args.append((doc, "logging/S" + str(s) + ".log"))
    results = pool.starmap(real_doc_test, args)
    pool.close() 
    pool.join()
    for r in results:
        print(str(r))

    
file_path = "data/hybrid_doc_small.txt"
log_file = "logging/Sampler.log"
max_features = 20
doc = Document(file_path, max_features, stem=True)
n_samplers = 5

'''
time1 = time.time()
run_paralel_samplers(doc, n_samplers)
time2 = time.time()
time_mt = time2-time1
'''

str_res = []
k_W_counts_outDir = "./debug/k_word_counts/"
shutil.rmtree(k_W_counts_outDir)
os.makedirs(k_W_counts_outDir)
time3 = time.time()
for i in range(n_samplers):
    wd , sampler = real_doc_test(doc, log_file)
    wd = "%.2f" % wd 
    str_res.append(wd)
    sampler_outDir = k_W_counts_outDir+"S"+str(i)+"_"+wd+"/"
    os.makedirs(sampler_outDir)
    debug_topic_assign(sampler, sampler_outDir)
print(str_res)
time4 = time.time()
time_st = time4-time3

#print('time multi-thread %0.3f secs\ntime single-thread %0.3f secs' % (time_mt, time_st))
#real_doc_test(doc, log_file)