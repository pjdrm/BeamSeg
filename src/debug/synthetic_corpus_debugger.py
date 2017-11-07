'''
Created on Jan 23, 2017

@author: root
'''
from dataset.synthetic_doc import SyntheticTopicTrackingDoc, SyntheticRndTopicPropsDoc
from debug.debug_tools import print_matrix, print_matrix_heat_map
    
def print_corpus(vocab_dic, doc_synth_tt, desc, outDir, flags):
    theta = doc_synth_tt.theta
    if flags[0]:
        print_matrix(theta)
    if flags[1]:
        print_matrix_heat_map(theta, desc + " Theta", outDir + desc + "_theta_heat_map.png")
    if flags[2]:
        print(doc_synth_tt.getText(vocab_dic))

def run():
    pi = 0.2
    alpha = 15
    beta = 0.6
    K = 10
    W = 15
    n_sents = 50
    sentence_l = 50
    vocab_dic = {}
    for w in range(W):
        vocab_dic[w] = "w" + str(w)
    outDir = "debug/synthetic_dataset/"    
    
    print_theta_flag = False
    print_heat_map_flag = True
    print_text_flag = False
    flags = [print_theta_flag, print_heat_map_flag, print_text_flag]
    
    doc_synth_tt = SyntheticTopicTrackingDoc(pi, alpha, beta, K, W, n_sents, sentence_l)
    doc_synth_tt.generate_doc()
    print_corpus(vocab_dic, doc_synth_tt, "Topic Tracking", outDir, flags)
    
    doc_synth_rnd_tp = SyntheticRndTopicPropsDoc(pi, alpha, beta, K, W, n_sents, sentence_l)
    doc_synth_rnd_tp.generate_doc()
    print_corpus(vocab_dic, doc_synth_rnd_tp, "Random Topic Proportions", outDir, flags)
