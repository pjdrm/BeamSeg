'''
Created on Dec 10, 2018

@author: root
'''
import os
import json
import numpy as np
import copy

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

def get_domain_results(dir, domain):
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
    
    for domain in results_dict:
        for seg_priorapp in results_dict[domain]:
                for prior_type in results_dict[domain][seg_priorapp]:
                    n_docs = len(results_dict[domain][seg_priorapp][prior_type]["wd"])
                    break
                break
            
        for i in range(n_docs):
            best_model = None
            best_wd = 1.1
            for seg_priorapp in results_dict[domain]:
                for prior_type in results_dict[domain][seg_priorapp]:
                    wd = results_dict[domain][seg_priorapp][prior_type]["wd"][i]
                    if wd < best_wd:
                        best_wd = wd
                        best_model = [seg_priorapp, prior_type]
            all_docs_results[best_model[0]][best_model[1]] += 1
            domain_results[best_model[0]][best_model[1]][domain] += 1
    
    domain_results_counts = {}
    for seg_priorapp in domain_results:
        for prior_type in domain_results[seg_priorapp]:
            for domain in domain_results[seg_priorapp][prior_type]:
                c = domain_results[seg_priorapp][prior_type][domain]
                if domain not in domain_results_counts:
                    domain_results_counts[domain] = [c, [seg_priorapp, prior_type]]
                else:
                    best_c = domain_results_counts[domain][0]
                    if c > best_c:
                        domain_results_counts[domain] = [c, [seg_priorapp, prior_type]]
                        
    for domain in domain_results_counts:
        res = domain_results_counts[domain]
        domain_results_processed[res[1][0]][res[1][1]] += res[0]
        
    return all_docs_results, domain_results_processed

def print_domain_results(results_dict):
    incomplete_domains = []
    print_str = ""
    prior_type_order = ["norm", "bb", "gp"]
    header = True
    domains_sort = sorted(results_dict.keys())
    if "segtt_dataset" in results_dict[domains_sort[0]]:
        headers = results_dict[domains_sort[0]]["segtt_dataset"]["gp"]
    else:
        headers = results_dict[domains_sort[0]]["segtt_modality"]["gp"]
    for sub_domain in domains_sort:
        if header:
            print_str += sub_domain+"\t"
            for doc in headers["doc_names"]:
                print_str += doc+"\t"
            print_str += "Average\nRND Segs\t"
            for wd in headers["wd_rnd_segs"]:
                print_str += str(wd)+"\t"
            print_str += "\nNO segs\t"
            for wd in headers["wd_bl_no_segs"]:
                print_str += str(wd)+"\t"
            print_str += "\n\n"
            header = False
        
        if "segbl_modality" in results_dict[sub_domain]:
            print_str += "segbl_modality\n"
            for prior_type in prior_type_order:
                print_str += prior_type + "\t"
                if "wd" in results_dict[sub_domain]["segbl_modality"][prior_type]:
                    wds = results_dict[sub_domain]["segbl_modality"][prior_type]["wd"]
                    for wd in wds:
                        print_str += str(wd)+"\t"
                    print_str += str(np.average(wds))
                else:
                    incomplete_domains.append(sub_domain)
                print_str += "\n"
            print_str += "\n"
            
            print_str += "segtt_modality\n"
            for prior_type in prior_type_order:
                print_str += prior_type + "\t"
                if "wd" in results_dict[sub_domain]["segtt_modality"][prior_type]:
                    wds = results_dict[sub_domain]["segtt_modality"][prior_type]["wd"]
                    for wd in wds:
                        print_str += str(wd)+"\t"
                    print_str += str(np.average(wds))
                else:
                    incomplete_domains.append(sub_domain)
                print_str += "\n"
            
        if "segtt_dataset" in results_dict[sub_domain]:
            print_str += "\nsegbl_dataset\n"
            for prior_type in prior_type_order:
                print_str += prior_type + "\t"
                if "wd" in results_dict[sub_domain]["segbl_dataset"][prior_type]:
                    wds = results_dict[sub_domain]["segbl_dataset"][prior_type]["wd"]
                    for wd in wds:
                        print_str += str(wd)+"\t"
                    print_str += str(np.average(wds))
                else:
                    incomplete_domains.append(sub_domain)
                print_str += "\n"
            print_str += "\n"
            
            print_str += "segtt_dataset\n"
            for prior_type in prior_type_order:
                print_str += prior_type + "\t"
                if "wd" in results_dict[sub_domain]["segtt_dataset"][prior_type]:
                    wds = results_dict[sub_domain]["segtt_dataset"][prior_type]["wd"]
                    for wd in wds:
                        print_str += str(wd)+"\t"
                    print_str += str(np.average(wds))
                else:
                    incomplete_domains.append(sub_domain)
                print_str += "\n"        
        
        print_str += "\n"
        header = True
    res_summary_alldocs, res_summary_domain = get_results_summary(results_dict)
    print_str += "\nResults Summary\n\n"
    for seg_type in res_summary_alldocs:
        print_str += seg_type+"\nPrior Type\t#Best Results (all docs)\t#Best Results (domain)\n"
        for prior_type in res_summary_alldocs[seg_type]:
            print_str += prior_type+"\t"+str(res_summary_alldocs[seg_type][prior_type])+"\t"
            print_str += str(res_summary_domain[seg_type][prior_type])+"\n"
        print_str += "\n"
    print(print_str)
    print(set(incomplete_domains))
    
results_dict = get_domain_results("/home/pjdrm/eclipse-workspace/TopicTrackingSegmentation/final_results", "news")
print_domain_results(results_dict)
#print(json.dumps(results_dict, sort_keys=True, indent=4))