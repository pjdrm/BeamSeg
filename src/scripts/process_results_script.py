'''
Created on Dec 10, 2018

@author: root
'''
import os
import json
import numpy as np
import copy

def merge_results(r1, r2):
    for subdomain in r2:
        for seg_model in r2[subdomain]:
            r1[subdomain][seg_model] = r2[subdomain][seg_model]
    return r1 
    
def get_results(file_path):
    resuls_dict = {}
    with open(file_path) as f:
        lins = f.readlines()
            
        for l in lins:
            if "Baseline NO segs" in l:
                resuls_dict["wd_bl_no_segs"] = eval(l.split("WD: ")[1])
            elif "Baseline RND segs" in l:
                resuls_dict["wd_rnd_segs"] = eval(l.split("WD: ")[1])
            elif "WD: " in l:
                resuls_dict["wd"] = eval(l.split("WD: ")[1])
            elif "F1" in l:
                f1 = float(l.split("F1: ")[1].split(" ")[0])
                f1_bl = float(l.split("F1_bl: ")[1].split(" ")[0])
                acc = float(l.split("Acc ")[1].split(" ")[0])
                acc_bl = float(l.split("Acc_bl ")[1])
                resuls_dict["ti"] = {"f1": f1, "f1_bl": f1_bl, "acc": acc, "acc_bl": acc_bl}
        if len(lins[-1]) > 1:
            resuls_dict["doc_names"] = eval(lins[-1])
    return resuls_dict

def get_bayesseg_subdomain(res_fp, lin, beamseg_res):
    def search_sub_domain(doc_name, beamseg_res):
        for bs_subdomain in beamseg_res:
            seg_type = list(beamseg_res[bs_subdomain].keys())[0]
            prior = list(beamseg_res[bs_subdomain][seg_type].keys())[0]
            beamseg_doc_names = beamseg_res[bs_subdomain][seg_type][prior]["doc_names"]
            if doc_name in beamseg_doc_names:
                return bs_subdomain
        
    if "lectures" in res_fp:
        doc_name = lin.replace("docId: ", "").split(" ")[0]
        sub_domain = search_sub_domain(doc_name, beamseg_res)
    elif res_fp.startswith("results_l"):
        doc_name = lin.replace("docId: ", "").split(" ")[0]
        sub_domain = doc_name.split("_")[0]
    elif "news" in res_fp:
        doc_name = lin.replace("docId: ", "").split(" ")[0].replace(".ref", "").replace(".", "")+".txt"
        sub_domain = search_sub_domain(doc_name, beamseg_res)
    elif "bio" in res_fp:
        doc_name = lin.replace("docId: ", "").split(" ")[0]
        sub_domain = search_sub_domain(doc_name, beamseg_res)
    elif "avl" in res_fp:
        doc_name = lin.replace("docId: ", "").split(" ")[0]
        sub_domain = "avl_trees"
        
    return sub_domain, doc_name
    
def get_bayesseg_results(root_dir, domain, beamseg_res, segtype_filter=["ui", "aps"]):
    results_dict = {}
    for dir in os.listdir(root_dir):
        if dir in segtype_filter: #APS results are just wrong. UI results output a boundary on the first sentence always.
            continue
        segmentor = dir
        dir = root_dir+dir+"/"+domain
        for res_fp in os.listdir(dir):
            with open(dir+"/"+res_fp) as res_f:
                lins = res_f.readlines()
            for lin in lins:
                if "docId" in lin:
                    sub_domain, doc_name = get_bayesseg_subdomain(res_fp, lin, beamseg_res)
                    str_split = lin.replace("docId: ", "").split(" ")
                    wd = float(str_split[-1])
                    if sub_domain not in results_dict:
                        results_dict[sub_domain] = {}
                    if segmentor not in results_dict[sub_domain]:
                        results_dict[sub_domain][segmentor] = {"": {"doc_names": [], "wd": [], "wd_bl_no_segs": [], "wd_rnd_segs": []}}
                    results_dict[sub_domain][segmentor][""]["doc_names"].append(doc_name)
                    results_dict[sub_domain][segmentor][""]["wd"].append(wd)
                    
                    seg_type = list(beamseg_res[sub_domain].keys())[0]
                    prior = list(beamseg_res[sub_domain][seg_type].keys())[0]
                    beamseg_doc_names = beamseg_res[sub_domain][seg_type][prior]["doc_names"]
                    rnd_bl = beamseg_res[sub_domain][seg_type][prior]["wd_rnd_segs"]
                    no_segs_bl = beamseg_res[sub_domain][seg_type][prior]["wd_bl_no_segs"]
                    
                    i = beamseg_doc_names.index(doc_name)
                    results_dict[sub_domain][segmentor][""]["wd_bl_no_segs"].append(no_segs_bl[i])
                    results_dict[sub_domain][segmentor][""]["wd_rnd_segs"].append(rnd_bl[i])
    return results_dict
                
def get_beamseg_results(dir, domain):
    results_dict = {}
    for res_f in os.listdir(dir):
        if domain in res_f:
            domain_id = res_f.split("_seg")[0]
            seg_type = "seg"+res_f.split("_seg")[1][0:2]
            if "modality_" in res_f:
                split_str = "modality_"
            else:
                split_str = "dataset_"
            prior_type = res_f.split(split_str)[1].replace(".txt", "")
            
            if domain_id not in results_dict:
                results_dict[domain_id] = {"segbl_modality": {}, "segbl_dataset": {}, "segtt_modality": {}, "segtt_dataset": {}}
            seg_type += "_"+split_str[:-1]
            results_dict[domain_id][seg_type][prior_type] = get_results(dir+"/"+res_f)
    res_cpy = copy.deepcopy(results_dict)
    for domain in res_cpy:
        for model in res_cpy[domain]:
            if len(results_dict[domain][model].keys()) == 0:
                results_dict[domain].pop(model)
    return results_dict

def get_results_summary(results_dict):
    all_docs_results = {}
    domain_results = {}
    for domain in results_dict:
        for seg_priorapp in results_dict[domain]:
            if seg_priorapp not in all_docs_results:
                all_docs_results[seg_priorapp] = {}
                domain_results[seg_priorapp] = {}
            for prior_type in results_dict[domain][seg_priorapp]:
                all_docs_results[seg_priorapp][prior_type] = 0
                if prior_type not in domain_results[seg_priorapp]:
                    domain_results[seg_priorapp][prior_type] = {}
                domain_results[seg_priorapp][prior_type][domain] = 0
    
    domain_results_processed = copy.deepcopy(all_docs_results)
    avg_wd_results = copy.deepcopy(all_docs_results)
    bl_avg_wd_results = {"rnd": copy.deepcopy(all_docs_results), "no_segs": copy.deepcopy(all_docs_results)}
    baseline_nosegs_results = copy.deepcopy(all_docs_results)
    baseline_nosegs_ties_results = copy.deepcopy(all_docs_results)
    baseline_rnd_results = copy.deepcopy(all_docs_results)
    
    for domain in results_dict:
        max_docs = -1
        for seg_priorapp in results_dict[domain]:
                for prior_type in results_dict[domain][seg_priorapp]:
                    docs = results_dict[domain][seg_priorapp][prior_type]["doc_names"]
                    if len(docs) > max_docs:
                        max_docs = len(docs)
                        doc_names = docs
                    if avg_wd_results[seg_priorapp][prior_type] == 0:
                        avg_wd_results[seg_priorapp][prior_type] = []
                        bl_avg_wd_results["rnd"][seg_priorapp][prior_type] = []
                        bl_avg_wd_results["no_segs"][seg_priorapp][prior_type] = []
                    avg_wd_results[seg_priorapp][prior_type] += results_dict[domain][seg_priorapp][prior_type]["wd"]
                    bl_avg_wd_results["rnd"][seg_priorapp][prior_type] += results_dict[domain][seg_priorapp][prior_type]["wd_rnd_segs"]
                    bl_avg_wd_results["no_segs"][seg_priorapp][prior_type] += results_dict[domain][seg_priorapp][prior_type]["wd_bl_no_segs"]
        
        for doc_name in doc_names:
            best_models = None
            best_wd = 1.1
            for seg_priorapp in results_dict[domain]:
                for prior_type in results_dict[domain][seg_priorapp]:
                    if doc_name in results_dict[domain][seg_priorapp][prior_type]["doc_names"]:
                        i = results_dict[domain][seg_priorapp][prior_type]["doc_names"].index(doc_name)
                        wd = results_dict[domain][seg_priorapp][prior_type]["wd"][i]
                        wd_no_segs_bl = results_dict[domain][seg_priorapp][prior_type]["wd_bl_no_segs"][i]
                        wd_rnd_bl = results_dict[domain][seg_priorapp][prior_type]["wd_rnd_segs"][i]
                        
                        if wd < wd_no_segs_bl:
                            baseline_nosegs_results[seg_priorapp][prior_type] += 1
                        
                        if wd == wd_no_segs_bl:
                            baseline_nosegs_ties_results[seg_priorapp][prior_type] += 1
                            
                        if wd < wd_rnd_bl:
                            baseline_rnd_results[seg_priorapp][prior_type] += 1
                            
                        if wd == best_wd:
                            best_models.append([seg_priorapp, prior_type])
                            
                        if wd < best_wd:
                            best_wd = wd
                            best_models = [[seg_priorapp, prior_type]]
            for best_model in best_models:
                all_docs_results[best_model[0]][best_model[1]] += 1
                domain_results[best_model[0]][best_model[1]][domain] += 1
    
    domain_results_counts = {}
    for seg_priorapp in domain_results:
        for prior_type in domain_results[seg_priorapp]:
            for domain in domain_results[seg_priorapp][prior_type]:
                c = domain_results[seg_priorapp][prior_type][domain]
                if domain not in domain_results_counts:
                    domain_results_counts[domain] = [[c, [seg_priorapp, prior_type]]]
                else:
                    best_c = domain_results_counts[domain][0][0]
                    if c == best_c:
                        domain_results_counts[domain].append([c, [seg_priorapp, prior_type]])
                    if c > best_c:
                        domain_results_counts[domain] = [[c, [seg_priorapp, prior_type]]]
                        
    for domain in domain_results_counts:
        for res in domain_results_counts[domain]:
            domain_results_processed[res[1][0]][res[1][1]] += 1
            
    for seg_priorapp in results_dict[domain]:
        if seg_priorapp == "cvs":
            print()
        for prior_type in results_dict[domain][seg_priorapp]:
            avg_wd = np.average(avg_wd_results[seg_priorapp][prior_type])
            std_wd = np.std(avg_wd_results[seg_priorapp][prior_type])
            avg_wd_results[seg_priorapp][prior_type] = str(avg_wd)[0:5]+"+-"+str(std_wd)[0:5]
            
            avg_wd = np.average(bl_avg_wd_results["rnd"][seg_priorapp][prior_type])
            std_wd = np.std(bl_avg_wd_results["rnd"][seg_priorapp][prior_type])
            bl_avg_wd_results["rnd"][seg_priorapp][prior_type] = str(avg_wd)[0:5]+"+-"+str(std_wd)[0:5]
            
            avg_wd = np.average(bl_avg_wd_results["no_segs"][seg_priorapp][prior_type])
            std_wd = np.std(bl_avg_wd_results["no_segs"][seg_priorapp][prior_type])
            bl_avg_wd_results["no_segs"][seg_priorapp][prior_type] = str(avg_wd)[0:5]+"+-"+str(std_wd)[0:5]
        
    return all_docs_results, domain_results_processed, avg_wd_results, baseline_nosegs_results, baseline_nosegs_ties_results, baseline_rnd_results, bl_avg_wd_results

def get_subdomain_doc_names(subdomain_dict):
    max_len = -1
    for seg_type in subdomain_dict:
        for prior_type in subdomain_dict[seg_type]:
            docs = subdomain_dict[seg_type][prior_type]["doc_names"]
            if len(docs) > max_len:
                max_len = len(docs)
                doc_names = docs
    return doc_names

def print_domain_results(results_dict):
    incomplete_domains = []
    print_str = ""
    header = True
    domains_sort = sorted(results_dict.keys())
    for sub_domain in domains_sort:
        doc_names = get_subdomain_doc_names(results_dict[sub_domain])
        if header: #TODO: make sure we get the full list of documents
            seg_type = list(results_dict[sub_domain].keys())[0]
            prior_type = list(results_dict[sub_domain][seg_type].keys())[0]
            headers = results_dict[sub_domain][seg_type][prior_type]
            print_str += sub_domain+"\t"
            for doc in doc_names:
                print_str += doc+"\t"
            print_str += "Average\nRND Segs\t"
            for wd in headers["wd_rnd_segs"]:
                print_str += str(wd)+"\t"
            print_str += str(np.average(headers["wd_rnd_segs"]))
            
            print_str += "\nNO segs\t"
            for wd in headers["wd_bl_no_segs"]:
                print_str += str(wd)+"\t"
            print_str += str(np.average(headers["wd_bl_no_segs"]))
            print_str += "\n\n"
            header = False
        
        seg_types_sorted = sorted(results_dict[sub_domain].keys())
        for seg_type in seg_types_sorted:
            prior_type_order = sorted(list(results_dict[sub_domain][seg_type]))
            print_str += seg_type+"\n"
            for prior_type in prior_type_order:
                print_str += prior_type + "\t"
                for doc in doc_names:
                    if doc in results_dict[sub_domain][seg_type][prior_type]["doc_names"]:
                        i = results_dict[sub_domain][seg_type][prior_type]["doc_names"].index(doc)
                        wd = results_dict[sub_domain][seg_type][prior_type]["wd"][i]
                        print_str += str(wd)+"\t"
                    else:
                        print_str += "\t"
                wds = results_dict[sub_domain][seg_type][prior_type]["wd"]
                print_str += str(np.average(wds))
                print_str += "\n"
            print_str += "\n"
            
        print_str += "\n"
        header = True
    res_summary_alldocs, res_summary_domain, avg_wd_results, baseline_nosegs_results, baseline_nosegs_ties_results, baseline_rnd_results, bl_avg_wd_results = get_results_summary(results_dict)
    n_docs = 0
    for domain in results_dict:
        for seg_type in results_dict[domain]:
            for prior_type in results_dict[domain][seg_type]:
                n_docs += len(results_dict[domain][seg_type][prior_type]["wd"])
                break
            break
    print_str += "\nResults Summary\n\n"
    for seg_type in res_summary_alldocs:
        prior_type_order = sorted(list(results_dict[sub_domain][seg_type]))
        print_str += seg_type+"\nPrior Type\t#Best Results (all docs)\t#Best Results (domain)\tWD avg (all docs)\t#Wins vs BL no segs\t#Ties vs BL no segs\t#Wins vs BL rnd segs\n"
        for prior_type in prior_type_order:
            print_str += prior_type+"\t"+str(res_summary_alldocs[seg_type][prior_type])+"\t"
            print_str += str(res_summary_domain[seg_type][prior_type])+"\t"
            print_str += str(avg_wd_results[seg_type][prior_type])+"\t"
            print_str += str(baseline_nosegs_results[seg_type][prior_type])+"/"+str(n_docs)+"\t"
            print_str += str(baseline_nosegs_ties_results[seg_type][prior_type])+"/"+str(n_docs)+"\t"
            print_str += str(baseline_rnd_results[seg_type][prior_type])+"/"+str(n_docs)+"\n"
        print_str += "\n"
    print_str += "\nBaseline WD avg results\nRND\tNo segs\n"
        
    for seg_type in bl_avg_wd_results["rnd"]:
        prior_type_order = sorted(list(results_dict[sub_domain][seg_type]))
        for prior_type in prior_type_order:
            print_str += str(bl_avg_wd_results["rnd"][seg_type][prior_type])+"\t"
            print_str += str(bl_avg_wd_results["no_segs"][seg_type][prior_type])
            break
        break
             
    print(print_str)
    print(set(incomplete_domains))

segtype_filer = ["beamseg", "aps", "ui"]
results_beamseg = get_beamseg_results("/home/pjdrm/workspace/TopicTrackingSegmentation/thesis_exp/beamseg", "news")
results_bayesseg =  get_bayesseg_results("/home/pjdrm/workspace/TopicTrackingSegmentation/thesis_exp/", "mw_news", results_beamseg, segtype_filer)
merged_results = merge_results(results_beamseg, results_bayesseg)
print_domain_results(results_bayesseg)
#print(json.dumps(results_dict, sort_keys=True, indent=4))