import numpy as np
from matplotlib import pyplot as plt
from dataset.real_doc import MultiDocument
from model.dp.segmentor import Data
import json
from random import shuffle
import os
import itertools as it

def process_data(docs, bottoms):
    plot_data_dic = {}
    for doc_i, doc in zip(bottoms, docs):
        begin = 0
        seg_len = 0
        k_prev = doc[0]
        for k in doc:
            if k != k_prev:
                end = begin+seg_len-1
                if k_prev not in plot_data_dic:
                    plot_data_dic[k_prev] = [[], [], []]
                plot_data_dic[k_prev][0].append(begin)
                plot_data_dic[k_prev][1].append(seg_len)
                plot_data_dic[k_prev][2].append(doc_i)
                k_prev = k
                begin = end+1
                seg_len = 1
            else:
                seg_len += 1
                
        end = begin+seg_len
        if k_prev not in plot_data_dic:
            plot_data_dic[k_prev] = [[], [], []]
        plot_data_dic[k_prev][0].append(begin)
        plot_data_dic[k_prev][1].append(seg_len)
        plot_data_dic[k_prev][2].append(doc_i)
    return plot_data_dic

def plot_topic_identification(plot_data_dic, colors, bars_y_pos, doc_names):
    map_color_indx = {}
    for i, k in enumerate(plot_data_dic):
        map_color_indx[k] = i
        
    for k in plot_data_dic:
        value = np.array(plot_data_dic[k][1])
        left = np.array(plot_data_dic[k][0])
        bottoms = np.array(plot_data_dic[k][2])
        color_i = map_color_indx[k]
        plt.bar(left=left, height=0.02, width=value, bottom=bottoms, color=colors[color_i], orientation="horizontal", label=k)
    
    plt.yticks(bars_y_pos+0.01, doc_names)
        
def load_topic_identification_results(log_file):
    with open(log_file) as f:
        lins = f.readlines()
    
    hyp_ti = None
    ref_ti = None
    for lin in lins:
        if lin.startswith("Hyp Topics: "):
            hyp_ti = eval(lin.split("Hyp Topics: ")[1][:-1])
            
        if lin.startswith("Ref Topics: "):
            ref_ti = eval(lin.split("Ref Topics: ")[1][:-1])
    doc_names = eval(lins[-1])
    for i in range(len(doc_names)):
        doc_names[i] = doc_names[i].replace("_processed_annotated", "").replace("_cap_man_processed_annotated", "")[:-4]
    return hyp_ti, ref_ti, doc_names

def rebuild_topic_identification_old_logs(log_file, config_file, target_order, outfile):
    with open(config_file) as data_file:    
        config = json.load(data_file)
    doc_col = MultiDocument(config)
    data = Data(doc_col)
    gs_topics_dict = {}
    for doc_i in range(data.n_docs):
        gs_topics_dict[doc_col.doc_names[doc_i]] = data.doc_rho_topics[doc_i]
        
    gs_topics = []
    for doc_name in target_order:
        gs_topics.append(gs_topics_dict[doc_name])
    
    with open(log_file) as f:
        lins = f.readlines()
    
    hyp_topics = None
    for lin in lins:
        if lin.startswith("Hyp Topics: "):
            hyp_topics = eval(lin.split("Hyp Topics: ")[1][:-1])
            
    hyp_ti = []
    begin = 0
    for gs_doc in gs_topics:
        l = len(gs_doc)
        end = begin+l
        hyp_doc = hyp_topics[begin:end]
        hyp_ti.append(hyp_doc)
        begin = end
        
    h_l = len(hyp_topics)
    gs_l = 0
    for gs_doc in gs_topics:
        gs_l += len(gs_doc)
    print("hyp len: %d ref len: %d"%(h_l, gs_l))
    str_rebuild_log = "Hyp Topics: "+str(hyp_ti)+"\nRef Topics: "+str(gs_topics)+"\n"+str(target_order)
    with open(outfile, "w+") as f:
        f.write(str_rebuild_log)
    #print(hyp_ti)
    #print(gs_topics)
    #print(target_order)
    
def rebuild_logs():   
    rebuild_topic_identification_old_logs("/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/logs_conll/l02_results_topics_log.txt", \
                                          "/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/configs/l02_config.json",\
                                          ['L02_14_processed_annotated_html.txt', 'L02_8_processed_annotated_html.txt', 'L02_19_processed_annotated_html.txt', 'L02_0_processed_annotated_html.txt', 'L02_39_processed_annotated_html.txt', 'L02_86_processed_annotated_html.txt', 'L02_422_processed_annotated_ppt.txt', 'L02_v0_cap_man_processed_annotated.txt', 'L02_v19_cap_man_processed_annotated.txt', 'L02_vref_cap_man_processed_annotated.txt'],\
                                          "/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/l02_conll_rebuild.txt")
    
    rebuild_topic_identification_old_logs("/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/logs_conll/l03_results_topics_log.txt", \
                                          "/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/configs/l03_config.json",\
                                          ['L03_7_processed_annotated_html.txt',
                                            'L03_342_processed_annotated_pdf.txt',
                                            'L03_48_processed_annotated_html.txt',
                                            'L03_365_processed_annotated_pdf.txt',
                                            'L03_185_processed_annotated_html.txt',
                                            'L03_402_processed_annotated_ppt.txt',
                                            'L03_v19_cap_man_processed_annotated.txt',
                                            'L03_239_processed_annotated_html.txt',
                                            'L03_213_processed_annotated_html.txt',
                                            'L03_vref_cap_man_processed_annotated.txt'],
                                          "/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/l03_conll_rebuild.txt")
    
    rebuild_topic_identification_old_logs("/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/logs_conll/l06_results_topics_log.txt", \
                                          "/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/configs/l06_config.json",\
                                          ['L06_433_processed_annotated_ppt.txt',
                                            'L06_33_processed_annotated_html.txt',
                                            'L06_v11_cap_man_processed_annotated.txt',
                                            'L06_153_processed_annotated_html.txt',
                                            'L06_422_processed_annotated_ppt.txt',
                                            'L06_421_processed_annotated_ppt.txt',
                                            'L06_435_processed_annotated_ppt.txt',
                                            'L06_77_processed_annotated_html.txt',
                                            'L06_111_processed_annotated_html.txt',
                                            'L06_vref_cap_man_processed_annotated.txt'],
                                          "/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/l06_conll_rebuild.txt")
    
    rebuild_topic_identification_old_logs("/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/logs_conll/l08_results_topics_log.txt", \
                                          "/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/configs/l08_config.json",\
                                          ['L08_vref_processed_annotated.txt',
                                            'L08_v2_processed_annotated.txt',
                                            'L08_v102_processed_annotated.txt',
                                            'L08_167_processed_annotated_html.txt',
                                            'L08_404_processed_annotated_ppt.txt',
                                            'L08_406_processed_annotated_ppt.txt',
                                            'L08_407_processed_annotated_ppt.txt',
                                            'L08_86_processed_annotated_html.txt',
                                            'L08_92_processed_annotated_html.txt',
                                            'L08_283_processed_annotated_html.txt',
                                            'L08_284_processed_annotated_html.txt'],
                                          "/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/l08_conll_rebuild.txt")
     
    rebuild_topic_identification_old_logs("/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/logs_conll/l10_results_topics_log.txt", \
                                          "/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/configs/l10_config.json",\
                                          ['L10_181_processed_annotated_html.txt',
                                            'L10_320_processed_annotated_pdf.txt',
                                            'L10_413_processed_annotated_ppt.txt',
                                            'L10_124_processed_annotated_html.txt',
                                            'L10_v21_processed_annotated.txt',
                                            'L10_31_processed_annotated_html.txt',
                                            'L10_v0_processed_annotated.txt',
                                            'L10_192_processed_annotated_html.txt',
                                            'L10_174_processed_annotated_html.txt',
                                            'L10_vref_processed_annotated.txt'],
                                          "/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/l10_conll_rebuild.txt")
    
def plot_ti_logs(log_dir, out_dir):
    for log_file in os.listdir(log_dir):
        log_file_path = log_dir+log_file
        if os.path.isdir(log_file_path):
            continue
        hyp_ti, ref_ti, doc_names = load_topic_identification_results(log_file_path)
        total_right, total_possible = stats_cross_doc_topics(ref_ti, hyp_ti, doc_names)
        
        bar_dist = 0.06
        bottoms = [bar_dist]*len(hyp_ti)
        bottoms[0] = 0.02
        bottoms = np.cumsum(bottoms, axis=0)
        plot_data_hyp = process_data(hyp_ti, bottoms)
        plot_data_ref = process_data(ref_ti, bottoms)
        
        colors_ref = ['#86E0A1', '#76D9A9', '#6AD1AF', '#61C9B5', '#5DC1B8', '#5DB8BA', '#62AFBB', '#68A5B9', '#709BB5', '#7892AF', '#7F88A7', '#857E9E', '#897493', '#8B6B88', '#8C627C', '#8B5A6F', '#885363', '#834C56', '#7D464B', '#764140']
        colors_hyp = ['#7fbc41', '#4d9221', '#dfc27d', '#80cdc1', '#c51b7d', '#de77ae', '#e6f5d0', '#35978f', '#01665e', '#2166ac', '#4d4d4d', '#f6e8c3', '#b8e186', '#fde0ef', '#bf812d', '#c7eae5', '#f5f5f5', '#f1b6da', '#8c510a']
        colors_ref = colors_hyp
        
        x_max = len(max(hyp_ti, key=lambda x: len(x)))
        fig = plt.figure(1)
        ax = plt.subplot(211)
        ax.set_xlim(0, x_max)
        ax.tick_params(axis=u'y', which=u'both',length=0)
        
        plot_topic_identification(plot_data_ref, colors_ref, bottoms, doc_names)
        plt.title('Reference')
        
        ax1 = plt.subplot(212)
        ax1.set_xlim(0, x_max)
        ax1.tick_params(axis=u'y', which=u'both',length=0)
        plot_topic_identification(plot_data_hyp, colors_hyp, bottoms, doc_names)
        plt.title('Hypothesis\n'+str(total_right)+"/"+str(total_possible)+" correct topic identifications")
        
        plt.subplots_adjust(hspace=0.4)
        fig.savefig(out_dir+log_file.split("_")[0]+"ti_plot.pdf", bbox_inches='tight')
        fig.clf()

def count_cross_doc_topics(ref_d1, ref_d2, hyp_d1, hyp_d2):
    total = 0
    prev_k = hyp_d1[0]
    found_match = False
    for i in range(len(hyp_d1)):
        ref_d1_k = ref_d1[i]
        hyp_d1_k = hyp_d1[i]
        if found_match and hyp_d1_k == prev_k:
            continue
        else:
            found_match = False
        for j in range(len(hyp_d2)):
            ref_d2_k = ref_d2[j]
            hyp_d2_k = hyp_d2[j]
            if ref_d1_k == ref_d2_k and hyp_d1_k == hyp_d2_k:
                total += 1
                found_match = True
                break
        prev_k = hyp_d1_k
    return total
    
def stats_cross_doc_topics(ref_topics, hyp_topics, doc_names):
    doc_combs = list(it.combinations(range(len(ref_topics)), 2))
    total_possible = 0
    total_right = 0
    for doc_i, doc_j in doc_combs:
        ref_d1 = ref_topics[doc_i]
        ref_d2 = ref_topics[doc_j]
        hyp_d1 = hyp_topics[doc_i]
        hyp_d2 = hyp_topics[doc_j]
        topic_rel_count = count_cross_doc_topics(ref_d1, ref_d2, hyp_d1, hyp_d2)
        total_right += topic_rel_count
        total_possible += count_cross_doc_topics(ref_d1, ref_d2, ref_d1, ref_d2)
        if topic_rel_count > 0:
            print("%s %s: %d"%(doc_names[doc_i], doc_names[doc_j], topic_rel_count))
    print("%d/%d correct topic identifications" % (total_right, total_possible))
    return total_right, total_possible
    
rebuild_logs()
plot_ti_logs("/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/", "/home/pjdrm/workspace/TopicTrackingSegmentation/logs/rebuild_logs/plots/")