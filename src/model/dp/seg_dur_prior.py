'''
Created on Sep 6, 2018

@author: root
'''
import numpy as np
from scipy.special import gammaln

class SegDurPrior(object):
    '''
    classdocs
    '''
    def __init__(self, config, data):
        self.hyper_params = None
        self.segmentation_log_prior = None
        self.dataset_len = data.total_sents
        self.n_docs = data.n_docs
        
        prior_desc = config["seg_dur_prior_config"][0].split("-")
        prior_dist = prior_desc[0]
        prior_type = prior_desc[1]
        hyper_params_raw = config["seg_dur_prior_config"][1]
        if prior_dist == "normal":
            self.segmentation_log_prior = self.segmentation_normal_log_prior
            if prior_type == "indv":
                self.hyper_params = data.seg_dur_prior_indv
            elif prior_type == "dataset":
                self.hyper_params = data.seg_dur_prior_dataset
            elif prior_type == "modality":
                self.hyper_params = data.seg_dur_prior_modality
            elif prior_type == "config":
                self.hyper_params = [hyper_params_raw]*data.n_docs
            else:
                print("ERROR: unknown prior tyoe %s"%prior_type)
        elif prior_dist == "beta_bern":
            self.segmentation_log_prior = self.segmentation_beta_bern_log_prior
            self.hyper_params = hyper_params_raw
        elif prior_dist == "gamma_poisson":
            self.segmentation_log_prior = self.segmentation_gamma_poisson_log_prior
            self.hyper_params = hyper_params_raw
                
    def normal_log_prior(self, seg_size, doc_i):
        mean = self.hyper_params[doc_i][0]
        std = self.hyper_params[doc_i][1]
        norm_logpdf = -np.log((np.sqrt(2*np.pi*(std**2))))-(seg_size-mean)**2/(2*(std**2))
        return norm_logpdf
        
    def segmentation_normal_log_prior(self, u_clusters):
        log_prior = 0.0
        for u_cluster in u_clusters:
            for doc_i in u_cluster.get_docs():
                u_begin, u_end = u_cluster.get_segment(doc_i)
                seg_size = u_end-u_begin+1
                log_prior += self.normal_log_prior(seg_size, doc_i)
        return log_prior
    
    def segmentation_beta_bern_log_prior(self, u_clusters):
        f1 = np.zeros(self.n_docs)
        f2 = np.zeros(self.n_docs)
        denom = np.zeros(self.n_docs)
        for u_cluster in u_clusters:
            for doc_i in u_cluster.get_docs():
                u_begin, u_end = u_cluster.get_segment(doc_i)
                seg_len = u_end-u_begin+1
                f1[doc_i] += 1.0
                f2[doc_i] += seg_len
                denom[doc_i] += seg_len
                
        f2 -= f1
        f1 += self.hyper_params
        f2 += self.hyper_params
        denom += 2*self.hyper_params
        log_prior = np.sum(gammaln(f1)+gammaln(f2)-gammaln(denom))
        return log_prior
    
    def segmentation_gamma_poisson_log_prior(self, u_clusters):
        #[alpha, beta, lambda_hp, interval]
        doc_lens = np.zeros(self.n_docs)
        n_rho1 = np.zeros(self.n_docs)
        for u_cluster in u_clusters:
            for doc_i in u_cluster.get_docs():
                u_begin, u_end = u_cluster.get_segment(doc_i)
                seg_len = u_end-u_begin+1
                doc_lens[doc_i] += seg_len
                n_rho1[doc_i] += 1.0
                
        alpha = self.hyper_params[0]
        beta = self.hyper_params[1]
        lambda_hp = np.array([self.hyper_params[2]]*self.n_docs)
        interval = np.array([self.hyper_params[3]]*self.n_docs)
        n = 1
        
        lambda_adjusted = doc_lens*lambda_hp/interval
        f1 = (n_rho1+alpha-1)*np.log(lambda_adjusted)
        f1[f1 == np.NINF] = 0.0 #in the greedy algorithm we might not have reached a document yet, which results in doc:len 0
        f2 = -lambda_adjusted*(n+beta)
        log_prior = np.sum(f1+f2)
        return log_prior
                