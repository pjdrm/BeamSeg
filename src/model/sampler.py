'''
Created on Feb 9, 2017

@author: root
'''
from tqdm import trange
import numpy as np
import logging
from eval.eval_tools import wd

class SegmentationModelSampler():
    def __init__(self, segmentation_model, log_file="logging/Sampler.log"):
        self.log_init(log_file)
        self.seg_model = segmentation_model
     
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
        self.sampler_log.info('Ref %s', str(self.seg_model.doc.rho))
        self.sampler_log.info('Hyp %s', str(self.seg_model.rho))
        for i in t:
            self.seg_model.sample_z()
            self.seg_model.sample_rho()
            if burn_in > 0:
                t.set_description("Burn-in iter %i rho = 1 %d" % (burn_in, self.seg_model.n_segs))
                burn_in -= 1
            else:
                if lag_counter > 0:
                    t.set_description("Lag iter %i\trho = 1 %d" % (iteration, self.seg_model.n_segs))
                    lag_counter -= 1
                else:
                    t.set_description("Estimate iter %i\trho = 1 %d" % (iteration, self.seg_model.n_segs))
                    lag_counter = lag
                    estimated_rho += self.seg_model.rho
                        
                    hyp_seg = np.rint(estimated_rho / iteration).astype(int)
                    self.sampler_log.info('wd: %f', wd(hyp_seg, self.seg_model.doc.rho))
                    self.sampler_log.info('Hyp %s', str(hyp_seg))
                    iteration += 1.0
                
        estimated_rho = estimated_rho / iteration
        estimated_rho = np.rint(estimated_rho)
        estimated_rho = estimated_rho.astype(int)
        wd_val = wd(estimated_rho, self.seg_model.doc.rho)
        self.sampler_log.info('wd: %f', wd_val)
        print("\nWD %s" % (str(wd_val)))