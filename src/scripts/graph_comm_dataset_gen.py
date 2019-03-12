'''
Created on Mar 6, 2019

@author: pjdrm
'''
import json
import os
import copy

def get_topic_labels(topic_root_dir, file_name):
    seg_labels = []
    i = 1
    done_flag = False
    while not done_flag:
        done_flag = True
        for topic_dir in os.listdir(topic_root_dir):
            for doc in os.listdir(topic_root_dir+topic_dir):
                if file_name+"seg"+str(i) in doc:
                    t_label = int(topic_dir.replace("topic", ""))
                    seg_labels.append(t_label)
                    i += 1
                    done_flag = False
                    break
    return seg_labels                  
        
def gen_dataset(root_dir, results_file):
    all_doc_txt = ""
    all_segment_label = []
    for doc_name in os.listdir(root_dir+"/doc_segs"):
        with open(root_dir+"/doc_segs/"+doc_name) as f:
            all_doc_txt += f.read()
        all_segment_label += get_topic_labels(root_dir+"/doc_rels/", doc_name)
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
                   "louvain": {"best_ari": -1,
                                 "weight": None,
                                 "score_func": None,
                                 "n": None},
                   "cnm": {"best_ari": -1,
                                 "score_func": None,
                                 "n": None},
                   "bigclam": {"best_ari": -1,
                                 "score_func": None,
                                 "n": None},
                   "clique_precolation": {"best_ari": -1,
                                 "score_func": None,
                                 "n": None},
                   "fast_greedy": {"best_ari": -1,
                                   "weight": None,
                                 "score_func": None,
                                 "n": None},
                   "edge_betweeness": {"best_ari": -1,
                                   "weight": None,
                                 "score_func": None,
                                 "n": None},
                   "girvan_newman": {"best_ari": -1,
                                 "score_func": None,
                                 "n": None},
                   "label_propagation": {"best_ari": -1,
                                   "weight": None,
                                 "score_func": None,
                                 "n": None},
                   "leading_eigenvector": {"best_ari": -1,
                                   "weight": None,
                                 "score_func": None,
                                 "n": None},
                   "spinglass": {"best_ari": -1,
                                   "weight": None,
                                 "score_func": None,
                                 "n": None},
                   "LDA": {"best_ari": -1,
                                   "wordsPerTopic": None,
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
                             "bandwidth": None},
                   "kmeans": {"best_ari": -1}}
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
        elif algo == "louvain":
            weight = get_val(l, "weight: ")
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["best_ari"] = ari
            best_params[algo]["weight"] = weight
            best_params[algo]["score_func"] = score_func
            best_params[algo]["n"] = n
        elif algo == "cnm":
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["best_ari"] = ari
            best_params[algo]["score_func"] = score_func
            best_params[algo]["n"] = n
        elif algo == "bigclam":
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["best_ari"] = ari
            best_params[algo]["score_func"] = score_func
            best_params[algo]["n"] = n
        elif algo == "clique_precolation":
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["best_ari"] = ari
            best_params[algo]["score_func"] = score_func
            best_params[algo]["n"] = n
        elif algo == "fast_greedy":
            weight = get_val(l, "weight: ")
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["weight"] = weight
            best_params[algo]["best_ari"] = ari
            best_params[algo]["score_func"] = score_func
            best_params[algo]["n"] = n
        elif algo == "edge_betweeness":
            weight = get_val(l, "weight: ")
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["weight"] = weight
            best_params[algo]["best_ari"] = ari
            best_params[algo]["score_func"] = score_func
            best_params[algo]["n"] = n
        elif algo == "girvan_newman":
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["best_ari"] = ari
            best_params[algo]["score_func"] = score_func
            best_params[algo]["n"] = n
        elif algo == "label_propagation":
            weight = get_val(l, "weight: ")
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["weight"] = weight
            best_params[algo]["best_ari"] = ari
            best_params[algo]["score_func"] = score_func
            best_params[algo]["n"] = n
        elif algo == "leading_eigenvector":
            weight = get_val(l, "weight: ")
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["weight"] = weight
            best_params[algo]["best_ari"] = ari
            best_params[algo]["score_func"] = score_func
            best_params[algo]["n"] = n
        elif algo == "spinglass":
            weight = get_val(l, "weight: ")
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["weight"] = weight
            best_params[algo]["best_ari"] = ari
            best_params[algo]["score_func"] = score_func
            best_params[algo]["n"] = n
        elif algo == "LDA":
            wordsPerTopic = get_val(l, "wordsPerTopic: ")
            score_func = get_val(l, "score_func: ")
            n = get_val(l, "top-N ")
            best_params[algo]["wordsPerTopic"] = wordsPerTopic
            best_params[algo]["best_ari"] = ari
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
        elif algo == "dbscan":
            minPts = get_val(l, "minPts: ")
            eps = get_val(l, "eps: ")
            metric = get_val(l, "metric: ")
            best_params[algo]["best_ari"] = ari
            best_params[algo]["minPts"] = minPts
            best_params[algo]["eps"] = eps
            best_params[algo]["metric"] = metric
            best_params[algo]["vars"] = var
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
        elif algo == "louvain":
            algo_cfg = cfg_template["algorithms"]["slicing_louvain"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["weight_schemes"] = [algo_params["weight"]]
            algo_cfg["score_function"] = [algo_params["score_func"]]
        elif algo == "cnm":
            algo_cfg = cfg_template["algorithms"]["slicing_cnm"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["score_function"] = [algo_params["score_func"]]
        elif algo == "bigclam":
            algo_cfg = cfg_template["algorithms"]["slicing_bigclam"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["score_function"] = [algo_params["score_func"]]
        elif algo == "clique_precolation":
            algo_cfg = cfg_template["algorithms"]["slicing_clique_precolation"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["score_function"] = [algo_params["score_func"]]
        elif algo == "fast_greedy":
            algo_cfg = cfg_template["algorithms"]["slicing_community_fastgreedy"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["weight_schemes"] = [algo_params["weight"]]
            algo_cfg["score_function"] = [algo_params["score_func"]]
        elif algo == "edge_betweeness":
            algo_cfg = cfg_template["algorithms"]["slicing_edge_betweenness"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["weight_schemes"] = [algo_params["weight"]]
            algo_cfg["score_function"] = [algo_params["score_func"]]
        elif algo == "girvan_newman":
            algo_cfg = cfg_template["algorithms"]["slicing_girvan_newman"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["score_function"] = [algo_params["score_func"]]
        elif algo == "label_propagation":
            algo_cfg = cfg_template["algorithms"]["slicing_label_propagation"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["weight_schemes"] = [algo_params["weight"]]
            algo_cfg["score_function"] = [algo_params["score_func"]]
        elif algo == "leading_eigenvector":
            algo_cfg = cfg_template["algorithms"]["slicing_leading_eigenvector"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["weight_schemes"] = [algo_params["weight"]]
            algo_cfg["score_function"] = [algo_params["score_func"]]
        elif algo == "spinglass":
            algo_cfg = cfg_template["algorithms"]["slicing_spinglass"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["weight_schemes"] = [algo_params["weight"]]
            algo_cfg["score_function"] = [algo_params["score_func"]]
        elif algo == "LDA":
            algo_cfg = cfg_template["algorithms"]["slicing_lda"]
            algo_cfg["n"] = '['+algo_params["n"]+']'
            algo_cfg["score_function"] = [algo_params["score_func"]]
            algo_cfg["wordsPerTopic"] = '['+algo_params["wordsPerTopic"]+']'
        elif algo == "Affinity":
            algo_cfg = cfg_template["algorithms"]["slicing_affinity_propagation"]
            algo_cfg["damping"] = '['+algo_params["damping"]+']'
            algo_cfg["preference"] = '['+algo_params["preference"]+']'
        elif algo == "dbscan":
            algo_cfg = cfg_template["algorithms"]["slicing_dbscan"]
            algo_cfg["metric"] = [algo_params["metric"]]
            algo_cfg["eps"] = '['+algo_params["eps"]+']'
            algo_cfg["minPts"] = '['+algo_params["minPts"]+']'
            algo_cfg["vars"] = '['+algo_params["vars"]+']'
        elif algo == "Agglomerative":
            algo_cfg = cfg_template["algorithms"]["slicing_agglomerative_clustering"]
            algo_cfg["metric"] = [algo_params["metric"]]
            algo_cfg["linkage"] = [algo_params["linkage"]]
            algo_cfg["vars"] = '['+algo_params["vars"]+']'
        elif algo == "Spectral":
            algo_cfg = cfg_template["algorithms"]["slicing_spectral"]
            algo_cfg["metric"] = [algo_params["metric"]]
            algo_cfg["vars"] = '['+algo_params["var"]+']'
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
    
    run_script = ""
    for domain_dir in os.listdir(domains_dir):
        if "domain" in domain_dir:
            domain_id = domain_dir.replace("omain", "")+"_"
        else:
            domain_id = domain_dir+"_"
        result_fp = get_res_fp(domain_id, domain_desc, results_dir)
        all_segment_label, all_doc_txt = gen_dataset(domains_dir+domain_dir, result_fp)
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
        yalm_cfg = change_doc_in_dir(yalm_cfg, "resources/"+domain_desc+"/docs/"+domain_id[:-1])
        yalm_fp = domain_desc+"_"+domain_dir+".yaml"
        with open(out_dir+"/yalms/"+yalm_fp, "w+") as docs_f:
            docs_f.write(yalm_cfg)
            
        run_script += "python tests/slice_tester.py resources/"+domain_desc+"/yalms/"+yalm_fp+" "+"resources/"+domain_desc+"/configs/"+cfg_fp+"\n"
    with open(out_dir+"run_tests_"+domain_desc+".sh", "w+") as f:
        f.write(run_script)
        
        
best_params = find_best_params("/home/pjdrm/Dropbox/results_AVL_common_topics.txt")
cfg_final = gen_config(best_params, "/home/pjdrm/Dropbox/tests_bio_d0.json")
gen_experiment(cfg_final,
               "/home/pjdrm/Dropbox/bio_d0.yaml",
               "/home/pjdrm/workspace/TopicTrackingSegmentation/dataset/Lectures/",
               "/home/pjdrm/workspace/TopicTrackingSegmentation/thesis_exp/beamseg/",
               "lectures",
               "/home/pjdrm/Desktop/gcm_exps/Lectures/")
#gen_dataset("/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/dataset/Biography/domain0/doc_segs", "/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/thesis_exp/beamseg/bio_d0_segbl_modality_bb.txt")
