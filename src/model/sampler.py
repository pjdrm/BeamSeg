'''
Created on Feb 9, 2017

@author: pjdrm
'''
from tqdm import trange
import numpy as np
import logging
from eval.eval_tools import wd
from scipy import sparse

class SegmentationModelSampler():
    def __init__(self, segmentation_model, log_file="logging/Sampler.log"):
        self.log_init(log_file)
        self.seg_model = segmentation_model
        self.estimated_W_K_counts = sparse.csr_matrix((self.seg_model.W, self.seg_model.K))
     
    def log_init(self, log_file):
        self.sampler_log = logging.getLogger("sampler_logger")
        formatter = logging.Formatter('%(levelname)s:%(message)s')
        fileHandler = logging.FileHandler(log_file, mode='w')
        fileHandler.setFormatter(formatter)
    
        self.sampler_log.setLevel(logging.INFO)
        self.sampler_log.addHandler(fileHandler)
        
    def gibbs_sampler(self, n_iter, burn_in, lag):
        lag_counter = lag
        iteration = 1.0
        total_iterations = burn_in + n_iter*lag + n_iter
        t = trange(total_iterations, desc='', leave=True)
        estimated_rho = [0]*self.seg_model.n_sents
        self.estimated_W_K_counts += self.seg_model.W_K_counts
        self.sampler_log.info('Ref %s', str(self.seg_model.doc.rho))
        self.sampler_log.info('Hyp %s', str(self.seg_model.rho))
        for i in t:
            self.seg_model.sample_z()
            self.seg_model.sample_rho()
            if burn_in > 0:
                t.set_description("Burn-in iter %i n_segs %d" % (burn_in, self.seg_model.n_segs))
                burn_in -= 1
            else:
                if lag_counter > 0:
                    t.set_description("Lag iter %i\tn_segs %d" % (iteration, self.seg_model.n_segs))
                    lag_counter -= 1
                else:
                    t.set_description("Estimate iter %i\tn_segs %d" % (iteration, self.seg_model.n_segs))
                    lag_counter = lag
                    estimated_rho += self.seg_model.rho
                    self.estimated_W_K_counts += self.seg_model.W_K_counts
                    self.sampler_log.info('Hyp %s', str(self.seg_model.rho))
                    iteration += 1.0
                
        estimated_rho = estimated_rho / iteration
        estimated_rho = np.rint(estimated_rho)
        estimated_rho = estimated_rho.astype(int)
        self.estimated_W_K_counts = self.estimated_W_K_counts / iteration
        
        self.sampler_log.info('\nMH %s', str(estimated_rho).replace("\n", ""))
        self.sampler_log.info('\nGS %s', str(self.seg_model.doc.rho).replace(",", ""))
        wd_val = wd(estimated_rho, self.seg_model.doc.rho)
        self.sampler_log.info('wd: %f', wd_val)
        print("\nWD %s" % (str(wd_val)))
        return wd_val
