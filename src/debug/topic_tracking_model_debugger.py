'''
Created on Jan 30, 2017

This is tests the initialization
of the topic tracking model.

@author: root
'''
from dataset.corpus import SyntheticTopicTrackingDoc
from model.topic_tracking import TopicTrackingModel
import numpy as np
from debug.debug_tools import print_matrix_heat_map

def test_initialization(tt_model, outFile):
    assert np.array_equal(np.sum(tt_model.U_K_counts, axis = 0),\
                      np.sum(tt_model.W_K_counts, axis = 0)),\
                      "Topic Counts in U_K are different from W_K"
                      
    assert np.array_equal(np.sum(tt_model.U_W_counts, axis = 0),\
                          np.sum(tt_model.W_K_counts, axis = 1).T),\
                          "Word Counts in U_W are different from W_K"
                          
    assert tt_model.phi.shape == (K, W), "The shape of phi must match [W, K]"
    
    topic_counts = np.zeros(K)
    word_counts = np.zeros(W)
    for u in range(n_sents):
        for i in range(sentence_l):
            z_ui = tt_model.U_I_topics[u, i]
            w_ui = tt_model.U_I_words[u, i]
            topic_counts[z_ui] += 1
            word_counts[w_ui] += 1
            
    assert np.array_equal(topic_counts,\
                          np.array(np.sum(tt_model.W_K_counts, axis = 0))[0, :]),\
                          "Topic Counts in U_I_topics are different from W_K"
                          
    assert np.array_equal(word_counts,\
                          np.array(np.sum(tt_model.W_K_counts, axis = 1))[:, 0]),\
                          "Word Counts in U_I_words are different from W_K"
    
    '''
    Note: check the image file to see if the
    evolution of the topics seems like
    a topic tracking Model
    '''                      
    print_matrix_heat_map(tt_model.theta.toarray(), "Topic Tracking", outFile)

'''
This method tests the sampling of a z_ui variable.
The sentence u and word i are fix. u is the first
sentence of the second segment (Su_index = 0) and 
i is the fourth word (i = 3).

The methods only makes sure that the topic proportions 
from which we are going to sample z_ui sum to 1.
'''    
def test_z_ui_sampling(tt_model):
    #TODO: make a better test case that does not break if there is only 1 segment
    Su_index = 1
    i = 3
    Su_begin, Su_end = tt_model.get_Su_begin_end(Su_index)
    u = Su_begin
    w_ui = tt_model.U_I_words[u, i]
    topic_probs = []
    for k in range(tt_model.K):
        topic_probs.append(tt_model.prob_z_ui_k(w_ui, k, Su_index))
    topic_probs = topic_probs / np.sum(topic_probs)
    assert str(topic_probs.sum()) == "1.0", "The topic probabilities from which we are going to sample z_ui need to sum to 1"
    
def test_Z_sampling(tt_model, outFile):
    try:
        tt_model.sample_z()
    except:
        print("ERROR: sampling all Z variables is broken")
    test_initialization(tt_model, outFile)
    
def test_rho_u_sampling(tt_model):
    Su_index = 1
    Su_begin, Su_end = tt_model.get_Su_begin_end(Su_index)
    print(tt_model.rho)
    tt_model.sample_rho_u(Su_end-1, Su_index)
        
pi = 0.2
alpha = 15
beta = 0.6
K = 10
W = 15
n_sents = 50
sentence_l = 50

gamma = 0.4
doc_synth_tt = SyntheticTopicTrackingDoc(pi, alpha, beta, K, W, n_sents, sentence_l)
doc_synth_tt.generate_doc()
tt_model = TopicTrackingModel(gamma, alpha, beta, K, doc_synth_tt)

outFile = "debug/topic_tracking_model/topic_tracking_theta_heat_map_initial.png"
#test_initialization(tt_model, outFile)
#test_z_ui_sampling(tt_model)
test_Z_sampling(tt_model, outFile)
#test_rho_u_sampling(tt_model)



