'''
Created on Mar 6, 2019

@author: pjdrm
'''
import json
import os
import copy

def get_topic_labels(utt_topic_labels):
    prev_label = -1
    seg_labels = []
    for utt_label in utt_topic_labels:
        if utt_label != prev_label:
            seg_labels.append(utt_label)
        prev_label = utt_label
    return seg_labels
        
        
def gen_dataset(root_dir, results_file):
    with open(results_file) as res_f:
        lins = res_f.readlines()
        docs_list = eval(lins[-1])
        ref_topics = eval(lins[-2].replace("Ref Topics: ", ""))
        all_doc_txt = ""
        all_segment_label = []
        for doc_name, doc_topics in zip(docs_list, ref_topics):
            all_segment_label += get_topic_labels(doc_topics)
            with open(root_dir+"/"+doc_name) as f:
                all_doc_txt += f.read()
    all_doc_txt = all_doc_txt.replace("====================", "==========")
    return all_segment_label, all_doc_txt
    
def find_best_params(results_file):
    def get_val(lin, split_str):
        return lin.split(split_str)[1].split(" ")[0]
    
    best_params = {"walktraps": {"best_ari": -1,
                                 "steps": None,
                                 "weight": None,
                                 "score_func": None,
                                 "n": None},
                    "Affinity": {"best_ari": -1,
                                 "damping": None,
                                 "preference": None},
                    "Agglomerative": {"best_ari": -1,
                                     "metric": None,
                                     "linkage": None,
                                     "var": None},
                    "dbscan": {"best_ari": -1,
                             "minPts": None,
                             "eps": None,
                             "metric": None},
                    "Spectral": {"best_ari": -1,
                             "var": None,
                             "metric": None},
                    "Mean": {"best_ari": -1,
                             "bandwidth": None}}
    with open(results_file) as f:
        lins = f.readlines()
    for l in lins:
        algo = l.split(" ")[0]
        ari = float(get_val(l, "ARI: "))
        best_ari = best_params[algo]["best_ari"]
        if ari <= best_ari:
            continue
        if algo == "walktraps":
            steps = get_val(l, "steps ")
            weight = get_val(l, "weight: ")
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["best_ari"] = ari
            best_params[algo]["steps"] = steps
            best_params[algo]["weight"] = weight
            best_params[algo]["score_func"] = score_func
            best_params[algo]["n"] = n
        elif algo == "Affinity":
            damping = get_val(l, "damping: ")
            preference = get_val(l, "preference: ")
            best_params[algo]["damping"] = damping
            best_params[algo]["preference"] = preference
        elif algo == "Agglomerative":
            linkage = get_val(l, "linkage: ")
            metric = get_val(l, "metric: ")
            var = get_val(l, "var: ")
            best_params[algo]["best_ari"] = ari
            best_params[algo]["linkage"] = linkage
            best_params[algo]["metric"] = metric
            best_params[algo]["vars"] = var
        elif algo == "Spectral":
            var = get_val(l, "var: ")
            metric = get_val(l, "metric: ")
            best_params[algo]["best_ari"] = ari
            best_params[algo]["var"] = var
            best_params[algo]["metric"] = metric
        elif algo == "kmeans":
            best_params[algo]["best_ari"] = ari
        elif algo == "Mean":
            bandwidth = get_val(l, "bandwidth: ")
            best_params[algo]["best_ari"] = ari
            best_params[algo]["bandwidth"] = bandwidth
    return best_params

def gen_config(best_params, cfg_template_path):
    with open(cfg_template_path) as f:
        cfg_template = json.load(f)
        
    for algo in best_params:
        algo_params = best_params[algo]
        if algo == "walktraps":
            algo_cfg = cfg_template["algorithms"]["slicing_walktraps"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["weight_schemes"] = [algo_params["weight"]]
            algo_cfg["score_function"] = [algo_params["score_func"]]
            algo_cfg["steps"] = '['+algo_params["steps"]+']'
        elif algo == "Affinity":
            algo_cfg = cfg_template["algorithms"]["slicing_affinity_propagation"]
            algo_cfg["damping"] = '['+algo_params["damping"]+']'
            algo_cfg["preference"] = '['+algo_params["preference"]+']'
        elif algo == "Agglomerative":
            algo_cfg = cfg_template["algorithms"]["slicing_agglomerative_clustering"]
            algo_cfg["metric"] = [algo_params["metric"]]
            algo_cfg["linkage"] = [algo_params["linkage"]]
            algo_cfg["vars"] = '['+algo_params["vars"]+']'
        elif algo == "Spectral":
            algo_cfg = cfg_template["algorithms"]["slicing_spectral"]
            algo_cfg["metric"] = [algo_params["metric"]]
            algo_cfg["vars"] = '['+algo_params["vars"]+']'
        elif algo == "Mean":
            algo_cfg = cfg_template["algorithms"]["slicing_mean_shift"]
            algo_cfg["bandwidth"] = '['+algo_params["bandwidth"]+']'
    return cfg_template

def gen_experiment(algs_cfg, yaml_fp, domains_dir, results_dir, domain_desc, out_dir):
    def get_res_fp(domain_id, domain_desc, results_dir):
        for r_fp in os.listdir(results_dir):
            if domain_id in r_fp and domain_desc in r_fp:
                return r_fp
            
    def change_doc_in_dir(yalm_cfg, out_dir):
        final_yaml = ""
        flag = True
        for l in yalm_cfg.split("\n"):
            if flag and "input_directory" in l:
                l = l.split(":")[0]
                l += ": "+out_dir
                flag = False
            final_yaml += l+"\n"
        return final_yaml
    
    os.mkdir(out_dir+"/docs")
    os.mkdir(out_dir+"/configs")
    os.mkdir(out_dir+"/yalms")
    
    with open(yaml_fp) as f:
        yalm_template = f.read()
        
    for domain_dir in os.listdir(domains_dir):
        if "domain" in domain_dir:
            domain_id = domain_dir.replace("omain", "")+"_"
        result_fp = get_res_fp(domain_id, domain_desc, results_dir)
        all_segment_label, all_doc_txt = gen_dataset(domains_dir+domain_dir+"/doc_segs", results_dir+result_fp)
        cfg = copy.deepcopy(algs_cfg)
        cfg["slicing_true_labels"] =  str(all_segment_label)[1:-1]
        cfg_fp = domain_desc+"_"+domain_dir+".json"
        with open(out_dir+"/configs/"+cfg_fp, "w+") as cfg_f:
            cfg_f.write(json.dumps(cfg, indent=1))
        
        os.mkdir(out_dir+"docs/"+domain_id[:-1])
        docs_fp = domain_desc+"_docs_"+domain_dir+".txt"
        with open(out_dir+"docs/"+domain_id[:-1]+"/"+docs_fp, "w+") as docs_f:
            docs_f.write(all_doc_txt)
            
        yalm_cfg = copy.deepcopy(yalm_template)
        yalm_cfg = change_doc_in_dir(yalm_cfg, out_dir+"docs/"+domain_id[:-1])
        yalm_fp = domain_desc+"_"+domain_dir+".yaml"
        with open(out_dir+"/yalms/"+yalm_fp, "w+") as docs_f:
            docs_f.write(yalm_cfg)
        
        
best_params = find_best_params("/home/pjdrm/Dropbox/results_AVL_common_topics.txt")
cfg_final = gen_config(best_params, "/home/pjdrm/Dropbox/tests_bio_d0.json")
gen_experiment(cfg_final,
               "/home/pjdrm/Dropbox/bio_d0.yaml",
               "/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/dataset/Biography/",
               "/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/beamseg/",
               "bio",
               "/home/pjdrm/Desktop/gcm_exps/Biography/")
#gen_dataset("/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/dataset/Biography/domain0/doc_segs", "/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/beamseg/bio_d0_segbl_modality_bb.txt")
