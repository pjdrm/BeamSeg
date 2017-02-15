'''
Created on Feb 9, 2017

@author: root
'''
from tqdm import trange
import numpy as np
import logging
from eval.eval_tools import wd
import debug.debug_tools
from scipy import sparse

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
        
    def update_z_history(self, U_I_K_history, U_I_topics):
        for i in range(8):
            z_ui = U_I_topics[0, i]
            U_I_K_history[i, z_ui] += 1
            
    def gibbs_sampler(self, n_iter, burn_in, lag):
        lag_counter = lag
        iteration = 1.0
        total_iterations = burn_in + n_iter*lag + n_iter
        t = trange(total_iterations, desc='', leave=True)
        estimated_Z = sparse.csr_matrix((self.seg_model.doc.n_sents, max(self.seg_model.sents_len)))
        estimated_Z += self.seg_model.U_I_topics
        estimated_rho = [0]*self.seg_model.n_sents
        self.sampler_log.info('Ref %s', str(self.seg_model.doc.rho))
        self.sampler_log.info('Hyp %s', str(self.seg_model.rho))
        U_I_K_history = np.zeros((8, self.seg_model.K))
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
                    estimated_Z += self.seg_model.U_I_topics
                    self.update_z_history(U_I_K_history, self.seg_model.U_I_topics)
                    hyp_seg = np.rint(estimated_rho / iteration).astype(int)
                    #self.sampler_log.info('wd: %f', wd(hyp_seg, self.seg_model.doc.rho))
                    self.sampler_log.info('Hyp %s', str(self.seg_model.rho))
                    #Su_begin, Su_end = self.seg_model.get_Su_begin_end(0)
                    #self.sampler_log.info("\nmodel U_I_topics:\n%s" ,\
                    #                       debug.debug_tools.print_matrix(self.seg_model.U_I_topics[Su_begin:Su_end, 0:8]))
                    iteration += 1.0
                
        estimated_rho = estimated_rho / iteration
        print("estimated_rho %s" % (str(estimated_rho)))
        print('Ref %s' % (str(self.seg_model.doc.rho).replace(" ", ", ")))
        estimated_rho = np.rint(estimated_rho)
        estimated_rho = estimated_rho.astype(int)
        print("Hyp %s" % (str(estimated_rho).replace(" ", ", ")))
        
        estimated_Z = estimated_Z / iteration
        
        wd_val = wd(estimated_rho, self.seg_model.doc.rho)
        self.sampler_log.info('wd: %f', wd_val)
        print("\nWD %s" % (str(wd_val)))
        #print("model W_K_counts:\n%s\ndoc_synth W_K_counts:\n%s" % (self.seg_model.W_K_counts.toarray(), self.seg_model.doc.W_K_counts.toarray()))
        '''
        Su_begin, Su_end = self.seg_model.get_Su_begin_end(0)
        print("doc_synth U_I_topics:\n%s\nmodel U_I_topics:\n%s" % (debug.debug_tools.print_matrix(self.seg_model.doc.U_I_topics[Su_begin:Su_end, 0:8]),\
                                                                    debug.debug_tools.print_matrix(estimated_Z[Su_begin:Su_end, 0:8])))
        print("UI_Z_history:\n%s" % (debug.debug_tools.print_matrix(U_I_K_history)))
        '''
