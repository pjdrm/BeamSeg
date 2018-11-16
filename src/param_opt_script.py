'''
Created on Nov 6, 2018

@author: root
'''
import json
import sys
import numpy as np
from model.dp.segmentor import Data, SEG_TT
import model.dp.multi_doc_greedy_segmentor as greedy_seg
from eval.eval_tools import wd_evaluator
from dataset.real_doc import MultiDocument
from test_scripts import get_all_greedy_configs
import copy
import os
import time

def opt_step(data, doc_col, best_cfg, param_key, param_key2, param_index, learning_rate, sign):
    new_cfg = copy.deepcopy(best_cfg)
    if param_key == "beta":
        new_cfg[param_key] += learning_rate*sign
        if new_cfg[param_key][0] <= 0:
            return None, None
    elif param_key2 is not None:
        new_cfg[param_key][1][param_key2][param_index] += learning_rate*sign
        if new_cfg[param_key][1][param_key2][param_index] < 0:
            return None, None
    else:
        new_cfg[param_key][1][param_index] += learning_rate*sign
        if new_cfg[param_key][1][param_index] <= 0:
            return None, None
        
    seg_model = greedy_seg.MultiDocGreedySeg(data, seg_config=new_cfg)
    seg_model.segment_docs()
    wd = wd_evaluator(seg_model.get_all_segmentations(), doc_col)
    return wd, new_cfg
    
def opt_param(data,
              doc_col,
              best_wd,
              config,
              param_key,
              param_key2,
              param_index,
              learning_rate,
              log_file_path):
    improved = False
    best_cfg = config
    sign = 1.0
    best_wd_avg = np.average(best_wd)
    while True: #going in the positive direction of search
        wd, new_cfg = opt_step(data, doc_col, best_cfg, param_key, param_key2, param_index, learning_rate, sign)
        wd_avg = np.average(wd)
        with open(log_file_path, "a+") as log_f:
            print("wd_avg: %f  wd: %s seg_dur_prior: %s beta_prior: %f" % (wd_avg,
                                                                           wd,
                                                                           new_cfg["seg_dur_prior_config"][1],
                                                                           new_cfg["beta"][0]), file=log_f)
        if wd_avg < best_wd_avg:
            improved = True
            best_wd = wd
            best_wd_avg = wd_avg
            best_cfg = new_cfg
        else:
            break
        
    if not improved:
        sign = -1.0
        while True: #going in the negative direction of search
            wd, new_cfg = opt_step(data, doc_col, best_cfg, param_key, param_key2, param_index, learning_rate, sign)
            if wd is None:#means we were testing a negative value for a parameter
                break
            wd_avg = np.average(wd)
            with open(log_file_path, "a+") as log_f:
                print("wd_avg: %f  wd: %s seg_dur_prior: %s beta_prior: %f" % (wd_avg,
                                                                               wd,
                                                                               new_cfg["seg_dur_prior_config"][1],
                                                                               new_cfg["beta"][0]), file=log_f)
            if wd_avg < best_wd_avg:
                improved = True
                best_wd = wd
                best_wd_avg = wd_avg
                best_cfg = new_cfg
            else:
                break
    return improved, best_cfg, best_wd
    
def get_params_opt(cfg):
    params_list = []
    if type(cfg["seg_dur_prior_config"][1]) is dict:
        for key in cfg["seg_dur_prior_config"][1]:
            for i in range(len(cfg["seg_dur_prior_config"][1][key])):
                params_list.append(["seg_dur_prior_config", key, i])
    else:
        params_list += [["seg_dur_prior_config", None, 0], ["seg_dur_prior_config", None, 1]]
    return params_list
    
def opt_model_params(data_config, greedy_seg_config, log_file_path):
    if os.path.exists(log_file_path): os.remove(log_file_path)
    doc_col = MultiDocument(data_config)
    data = Data(doc_col)
    all_configs = get_all_greedy_configs(greedy_seg_config)
    config_inst = all_configs[0]
    if len(all_configs) > 1:
        print("ERROR: more than one base config was found")
        return
    
    beta = config_inst["beta"]
    config_inst["max_topics"] += doc_col.max_topics
    if config_inst["seg_func"] == SEG_TT:
        config_inst["beta"] = np.array([beta]*doc_col.W)
    else:
        config_inst["beta"] = np.array([beta]*doc_col.W)
    
    best_cfg = config_inst
    param_list = [["beta", None, None]]
    param_list += get_params_opt(best_cfg) #[["seg_dur_prior_config", "html", 0], ["seg_dur_prior_config", "html", 1]]
    seg_model = greedy_seg.MultiDocGreedySeg(data, seg_config=config_inst)
    seg_model.segment_docs()
    wd_begin = wd_evaluator(seg_model.get_all_segmentations(), doc_col)
    best_wd = wd_begin
    beta_base_learning_rate = 0.1
    beta_learning_rate = beta_base_learning_rate
    beta_learning_rate_step = 0.5
    
    base_learning_rate = 0.5
    learning_rate = base_learning_rate
    learning_rate_step = 3.0
    any_impr = False
    n_loops = 999999999
    max_time = 259200
    cum_time = 0
    times_up = False
    while True:
        if n_loops == 0:
            break
        for param_key, param_key2, param_index in param_list:
            if param_key == "beta":
                lr = beta_learning_rate
            else:
                lr = learning_rate
            start = time.time()
            improved, new_cfg, new_wd = opt_param(data,
                                                    doc_col,
                                                    best_wd,
                                                    best_cfg,
                                                    param_key,
                                                    param_key2,
                                                    param_index,
                                                    lr,
                                                    log_file_path)
            end = time.time()
            run_time = (end - start)
            cum_time += run_time
            if improved:
                best_wd = new_wd
                best_cfg = new_cfg
                any_impr = True
        with open(log_file_path, "a+") as log_f:
            print("best wd_avg: %f\nbest cfg: %s\nbeta: %f\n----Loop %d end------" % ( np.average(best_wd),
                                                                                       best_cfg["seg_dur_prior_config"],
                                                                                       best_cfg["beta"][0],
                                                                                       n_loops), file=log_f)
        if cum_time > max_time:
            print("Time's up!")
            times_up = False
            break
        n_loops -= 1
        if times_up:
            break
        
        if not any_impr:
            learning_rate += learning_rate_step
            beta_learning_rate += beta_learning_rate_step
        else:
            learning_rate = base_learning_rate
            beta_learning_rate = beta_base_learning_rate
        any_impr = False
            
    best_cfg["beta"] = best_cfg["beta"][0]
    with open(log_file_path, "a+") as log_f:
        print("wd start: %s wd end: %s\nbest cfg:\n%s" % (str(wd_begin), str(best_wd), best_cfg), file=log_f)
    
if __name__ == "__main__":
    if len(sys.argv) == 1:
        data_config = "../config/physics_test.json"
        greedy_seg_config = "../config/greedy_config.json"
        log_file_path = "../logs/opt_params_log.txt"
    else:
        data_config = sys.argv[1]
        greedy_seg_config = sys.argv[2]
        log_file_path = sys.argv[3]
        
    with open(data_config) as data_file:    
        data_config = json.load(data_file)
        
    with open(greedy_seg_config) as seg_file:
        greedy_seg_config = json.load(seg_file)

    opt_model_params(data_config, greedy_seg_config, log_file_path)