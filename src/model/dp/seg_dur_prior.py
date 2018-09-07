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
            lambda_hp = hyper_params_raw[0]
            interval = hyper_params_raw[1]
            alpha = hyper_params_raw[2]
            beta = hyper_params_raw[3]
            lambda_adjusted = float(self.dataset_len)*float(lambda_hp)/float(interval)
            self.hyper_params = [alpha, beta, lambda_adjusted]
                
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
    
    def segmentation_gamma_poisson_log_prior(self, u_clusters): #TODO: I need to rescale everytime. I forgot segmentation is incremental
        n_rho1 = 0
        for u_cluster in u_clusters:
            n_rho1 += len(u_cluster.get_docs())
            
        alpha = self.hyper_params[0]
        beta = self.hyper_params[1]
        lambda_hp = self.hyper_params[2]
        n = 1
         
        log_prior = (n_rho1+alpha-1.0)*np.log(lambda_hp)-(n+beta)*lambda_hp
        return log_prior
                