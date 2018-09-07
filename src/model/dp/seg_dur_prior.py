'''
Created on Sep 6, 2018

@author: root
'''
import numpy as np

class SegDurPrior(object):
    '''
    classdocs
    '''
    def __init__(self, config, data):
        self.hyper_params = None
        self.segmentation_log_prior = None
        self.dataset_len = data.total_sents
        
        if config["prior_class"] == "normal":
            self.segmentation_log_prior = self.segmentation_normal_log_prior
            prior_type = config["prior_type"]
            if prior_type == "indv":
                self.hyper_params = data.seg_dur_prior_indv
            elif prior_type == "dataset":
                self.hyper_params = data.seg_dur_prior_dataset
            elif prior_type == "modality":
                self.hyper_params = data.seg_dur_prior_modality
            elif prior_type == "config":
                self.hyper_params = [config["dur_prior_vals"]]*data.n_docs
            else:
                print("ERROR: unknown prior tyoe %s"%prior_type)
        elif config["prior_class"] == "beta_bern":
            self.segmentation_log_prior = self.segmentation_beta_bern_log_prior
            self.hyper_params = config["dur_prior_vals"]
        elif config["prior_class"] == "gamma_poisson":
            self.segmentation_log_prior = self.segmentation_gamma_poisson_log_prior
            raw_params = config["dur_prior_vals"]
            lambda_hp = raw_params[0]
            interval = raw_params[1]
            alpha = raw_params[2]
            beta = raw_params[3]
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
        n_rho1 = 0
        n_rho0 = 0
        for u_cluster in u_clusters:
            n_rho1 += len(u_cluster.get_docs())
            n_rho0 += u_cluster.get_n_sents()
        n_rho0 = n_rho0-n_rho1
        log_prior = np.log(self.hyper_params**n_rho1)+np.log((1.0-self.hyper_params))*n_rho0
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
                