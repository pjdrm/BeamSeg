'''
Created on Jan 30, 2017

This is tests the initialization
of the topic tracking model.

@author: root
'''
from dataset.corpus import SyntheticTopicTrackingDoc
from model.topic_tracking_segmentor import TopicTrackingModel
import debug.debug_tools as debug_tools

pi = 0.4
alpha = 15
alpha = 0.6
K = 10
W = 15
n_sents = 50
sentence_l = 50
log_flag = True

gamma = 10
doc_synth_tt = SyntheticTopicTrackingDoc(pi, alpha, alpha, K, W, n_sents, sentence_l)
doc_synth_tt.generate_doc()
rnd_topics_model = TopicTrackingModel(gamma, alpha, alpha, K, doc_synth_tt, log_flag)

outFile = "debug/topic_tracking_model/topic_tracking_theta_heat_map_initial.png"
#debug_tools.test_model_state(rnd_topics_model, outFile)
#debug_tools.test_z_ui_sampling(rnd_topics_model)
#debug_tools.test_Z_sampling(rnd_topics_model, outFile)
#debug_tools.test_rho_u_sampling(rnd_topics_model)
#debug_tools.test_rho_sampling(rnd_topics_model, outFile)

n_iter = 5
burn_in = 0
lag = 0
debug_tools.run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag)