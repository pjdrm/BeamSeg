'''
Created on Feb 9, 2017

@author: pjdrm
'''
from tqdm import trange
import numpy as np
from debug import log_tools
from eval.eval_tools import wd_evaluator, wd
from scipy import sparse
import time
np.set_printoptions(threshold=np.inf)

class SegmentationModelSampler():
    def __init__(self, segmentation_model, sampler_log_file="logging/Sampler.log"):
        self.sampler_log_file = sampler_log_file
        self.sampler_log = log_tools.log_init(sampler_log_file)
        self.seg_model = segmentation_model
        self.estimated_W_K_counts = sparse.csr_matrix((self.seg_model.W, self.seg_model.K))
        self.estimated_U_K_counts = sparse.csr_matrix(((self.seg_model.doc.n_sents, self.seg_model.K)))
        self.estimated_U_I_topics = sparse.csr_matrix((self.seg_model.doc.n_sents, max(self.seg_model.sents_len)))
        
        self.n_words = self.seg_model.doc.sents_len.sum()
        self.I_K_counts = sparse.csr_matrix((self.n_words, self.seg_model.K))
        
    def gibbs_sampler(self, n_iter, burn_in, lag):
        lag_counter = lag
        iteration = 1.0
        total_iterations = burn_in + n_iter*lag + n_iter
        t = trange(total_iterations, desc='', leave=True)
        estimated_rho = [0]*self.seg_model.n_sents
        self.estimated_W_K_counts += self.seg_model.W_K_counts
        for i in t:
            t_init = time.time()
            self.seg_model.sample_z()
            self.seg_model.sample_rho()
            t_end = time.time()
            if burn_in > 0:
                t.set_description("Burn-in iter %i" % (burn_in))
                burn_in -= 1
            else:
                if lag_counter > 0:
                    t.set_description("Lag iter %i" % (iteration))
                    lag_counter -= 1
                else:
                    self.sampler_log.info('Iteration time %s', str(t_end - t_init))
                    t.set_description("Estimate iter %i" % (iteration))
                    lag_counter = lag
                    estimated_rho += self.seg_model.rho
                    self.estimated_W_K_counts += self.seg_model.W_K_counts
                    self.estimated_U_K_counts += self.seg_model.U_K_counts
                    self.estimated_U_I_topics += self.seg_model.U_I_topics
                    #self.I_K_counts += self.get_I_K_counts(self.seg_model.U_I_topics)
                    iteration += 1.0
                    
                    #self.sampler_log.info('Hyp %s', str(self.seg_model.rho))
                    current_rho = self.estimate_rho(estimated_rho, iteration)
                    current_wd_val = wd(current_rho, self.seg_model.doc.rho)
                    self.sampler_log.info('Rho_Est %s', str(current_wd_val))
                    
                    
                    estimated_rho_eq_1 = np.append(np.nonzero(current_rho)[0], [self.seg_model.n_sents-1])
                    log_prob_joint = self.seg_model.log_prob_joint_dist(self.seg_model.gamma,\
                                                                        self.seg_model.beta,\
                                                                        self.seg_model.alpha,\
                                                                        estimated_rho_eq_1,\
                                                                        self.estimated_W_K_counts / iteration,\
                                                                        self.estimated_U_K_counts / iteration)
                    self.sampler_log.info('log_prob_joint %s', str(log_prob_joint))
                
        self.estimated_W_K_counts = self.estimated_W_K_counts / iteration
        self.estimated_U_K_counts = self.estimated_U_K_counts / iteration
        self.estimated_U_I_topics = self.estimated_U_I_topics / iteration
        self.sampler_log.info('\nestimated_U_K_counts:\n%s', self.estimated_U_K_counts)
        self.sampler_log.info('\nestimated_U_I_topics:\n%s',\
                              self.U_I_topics_to_string(self.estimated_U_I_topics,\
                                                        self.seg_model.sents_len))
        #self.sampler_log.info('\nI_K_counts:\n%s', self.I_K_counts_to_string(self.I_K_counts, self.seg_model.U_I_words, self.seg_model.doc.inv_vocab))
                              
        self.sampler_log.info('\nRho Prob %s', str(estimated_rho/iteration).replace("\n", ""))
        #self.sampler_log.info('Doc:\n%s', self.seg_model.doc.getText())
        self.sampler_log.info('Doc indexes: %s', str(self.seg_model.doc.docs_index))
        self.sampler_log.info('Doc names: %s', str(self.seg_model.doc.doc_names))
        estimated_rho = self.estimate_rho(estimated_rho, iteration) #np.rint(estimated_rho)
        self.sampler_log.info('\nMH %s', str(estimated_rho).replace("\n", "").replace(",", ""))
        self.sampler_log.info('\nGS %s', str(self.seg_model.doc.rho).replace("\n", ""))
        wd_val = wd_evaluator(estimated_rho, self.seg_model.doc)
        self.sampler_log.info('final_wd: %f', wd(estimated_rho, self.seg_model.doc.rho))
        print("\nWD %s" % (str(wd_val)))
        return wd_val
    
    def estimate_rho(self, estimated_rho, n_iters, thr = 0.8):
        return [1 if rho >= thr else 0 for rho in estimated_rho / n_iters]
    
    def U_I_topics_to_string(self, estimated_U_I_topics, sents_len):
        estimated_U_I_topics = np.rint(estimated_U_I_topics)
        str_res = ""
        for i in range(estimated_U_I_topics.shape[0]):
            for j in range(sents_len[i]):
                w_ui = self.seg_model.U_I_words[i,j]
                w_ui_str = self.seg_model.doc.inv_vocab[w_ui]
                str_res += w_ui_str + "-" + str(int(estimated_U_I_topics[i,j])) + " "
            str_res += "\n"
        return str_res
    
    def I_K_counts_to_string(self, I_K_counts, U_I_words, inv_vocab):
        str_r = ""
        index = 0
        for u in range(self.seg_model.doc.n_sents):
            for i in range(self.seg_model.doc.sents_len[u]):
                w_ui = U_I_words[u,i]
                w_ui_s = inv_vocab[w_ui]
                for k in range(self.seg_model.doc.K):
                    str_r += str(I_K_counts[index, k])[:3] + "\t\t"
                str_r += w_ui_s + "\n"
                index += 1
        return str_r
    
    def get_I_K_counts(self, U_I_topics):
        I_K_counts = sparse.csr_matrix((self.n_words, self.seg_model.K))
        index = 0
        for u in range(self.seg_model.doc.n_sents):
            for i in range(self.seg_model.doc.sents_len[u]):
                z_ui = U_I_topics[u,i]
                I_K_counts[index, z_ui] += 1
                index += 1
        return I_K_counts