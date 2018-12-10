'''
Created on Dec 10, 2018

@author: root
'''
import os
import json
#bio_d0_segbl_modality_bb
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
            prior_type = res_f.split("modality_")[1].replace(".txt", "")
            
            if domain_id not in results_dict:
                results_dict[domain_id] = {"segbl": {}, "segtt": {}}
                
            results_dict[domain_id][seg_type][prior_type] = get_results(dir+"/"+res_f)
    return results_dict

def print_domain_results(results_dict):
    incomplete_domains = []
    print_str = ""
    prior_type_order = ["norm", "bb", "gp"]
    header = True
    domains_sort = sorted(results_dict.keys())
    for sub_domain in domains_sort:
        if header:
            print_str += sub_domain+"\t"
            for doc in results_dict[domains_sort[0]]["segtt"]["gp"]["doc_names"]:
                print_str += doc+"\t"
            print_str += "\nRND Segs\t"
            for wd in results_dict[domains_sort[1]]["segtt"]["gp"]["wd_rnd_segs"]:
                print_str += str(wd)+"\t"
            print_str += "\nNO segs\t"
            for wd in results_dict[domains_sort[1]]["segtt"]["gp"]["wd_bl_no_segs"]:
                print_str += str(wd)+"\t"
            print_str += "\n\n"
            header = False
    
        print_str += "segbl\n"
        for prior_type in prior_type_order:
            print_str += prior_type + "\t"
            if "wd" in results_dict[sub_domain]["segbl"][prior_type]:
                for wd in results_dict[sub_domain]["segbl"][prior_type]["wd"]:
                    print_str += str(wd)+"\t"
            else:
                incomplete_domains.append(sub_domain+" segbl")
            print_str += "\n"
        print_str += "\n"
        
        print_str += "segtt\n"
        for prior_type in prior_type_order:
            print_str += prior_type + "\t"
            if "wd" in results_dict[sub_domain]["segtt"][prior_type]:
                for wd in results_dict[sub_domain]["segtt"][prior_type]["wd"]:
                    print_str += str(wd)+"\t"
            else:
                incomplete_domains.append(sub_domain+" segtt")
            print_str += "\n"
        print_str += "\n"
        header = True
    print(print_str)
    print(set(incomplete_domains))
    
results_dict = get_domain_results("/home/pjdrm/Desktop/final_results", "bio_d11")
print_domain_results(results_dict)
#print(json.dumps(results_dict, sort_keys=True, indent=4))