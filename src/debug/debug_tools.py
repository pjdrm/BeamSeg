'''
Created on Jan 30, 2017

@author: root
'''
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from model.sampler import SegmentationModelSampler
import matplotlib as mpl
import pandas as pd
import operator
import os
plt.style.use('ggplot')
plt.rcParams["figure.figsize"] = [16.0, 3.0]

def plot_ref_hyp_seg(ref_seg, hyp_seg, outFile, seg_dec):
    df2 = pd.DataFrame(np.array([ref_seg, hyp_seg]).T, columns=["Reference", seg_dec])
    ax = df2.plot.bar(stacked=True, width=1)
    ax.yaxis.set_ticks_position('left')
    ax.axhline(y = 0, linewidth=1.5, color='k')
    ax.axhline(y = 1.995, linewidth=1.3, color='k')
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='off',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='off') 
    plt.yticks(np.arange(0, 3, 1))
    plt.setp(ax.get_xticklabels(), visible=False)
    plt.xlabel('Sentences', fontsize=14, color="k")
    plt.ylabel('#Boundaries', fontsize=14, color="k")
    plt.legend(loc='upper left', fancybox=None, ncol=1, fontsize = 14)
    plt.savefig(outFile, bbox_inches='tight')
    
def print_ref_hyp_plots(logFile, outDir, seg_desc):
    with open(logFile) as f:
        lins = f.readlines()
        i = 0
        for j, lin in enumerate(lins):
            if lin.startswith("INFO:Ref "):
                ref_seg = getSeg(lins, j)
            
            if lin.startswith("INFO:Hyp "):
                hyp_seg = getSeg(lins, j)
                plot_ref_hyp_seg(ref_seg, hyp_seg, outDir + "gs_iter" + str(i) + ".png", seg_desc)
                i += 1
                
def getSeg(lins, j):
    seg_str = "[" + lins[j].split("[")[1]
    j += 1
    while not seg_str.endswith("]\n"):
        seg_str += lins[j]
        j += 1
    return eval(seg_str.strip().replace(" ", ", "))

def test_model_state(segmentation_model, outFile):
    assert np.array_equal(np.sum(segmentation_model.U_K_counts, axis = 0),\
                      np.sum(segmentation_model.W_K_counts, axis = 0)),\
                      "Topic Counts in U_K are different from W_K"
                      
    assert np.array_equal(np.sum(segmentation_model.U_W_counts, axis = 0),\
                          np.sum(segmentation_model.W_K_counts, axis = 1).T),\
                          "Word Counts in U_W are different from W_K"
                          
    assert segmentation_model.phi.shape == (segmentation_model.K, segmentation_model.W), "The shape of phi must match [W, K]"
    
    topic_counts = np.zeros(segmentation_model.K)
    word_counts = np.zeros(segmentation_model.W)
    for u in range(segmentation_model.n_sents):
        for i in range(segmentation_model.sents_len[u]):
            z_ui = segmentation_model.U_I_topics[u, i]
            w_ui = segmentation_model.U_I_words[u, i]
            topic_counts[z_ui] += 1
            word_counts[w_ui] += 1
            
    assert np.array_equal(topic_counts,\
                          np.array(np.sum(segmentation_model.W_K_counts, axis = 0))[0, :]),\
                          "Topic Counts in U_I_topics are different from W_K"
                          
    assert np.array_equal(word_counts,\
                          np.array(np.sum(segmentation_model.W_K_counts, axis = 1))[:, 0]),\
                          "Word Counts in U_I_words are different from W_K"
                          
    assert segmentation_model.n_segs == (len(segmentation_model.rho_eq_1)),\
           "Number of segments does not match rho_eq_1 len"
    
    '''
    Note: check the image file to see if the
    evolution of the topics seems like
    a topic tracking Model
    '''                      
    print_matrix_heat_map(segmentation_model.theta.toarray(), "Topic Tracking", outFile)
    
'''
This method tests the sampling of a z_ui variable.
The sentence u and word i are fix. u is the first
sentence of the second segment (Su_index = 0) and 
i is the fourth word (i = 3).

The methods only makes sure that the topic proportions 
from which we are going to sample z_ui sum to 1.
'''    
def test_z_ui_sampling(segmentation_model):
    #TODO: make a better test case that does not break if there is only 1 segment
    Su_index = 1
    i = 3
    Su_begin, Su_end = segmentation_model.get_Su_begin_end(Su_index)
    u = Su_begin
    w_ui = segmentation_model.U_I_words[u, i]
    topic_probs = []
    for k in range(segmentation_model.K):
        topic_probs.append(segmentation_model.prob_z_ui_k(w_ui, k, Su_index))
    topic_probs = topic_probs / np.sum(topic_probs)
    assert str(topic_probs.sum()) == "1.0", "The topic probabilities from which we are going to sample z_ui need to sum to 1"
    
def test_Z_sampling(segmentation_model, outFile):
    '''
    Just trying to see if the code runs and
    then check the consistency of the state
    after sampling Z 
    '''
    segmentation_model.sample_z()
    test_model_state(segmentation_model, outFile)
    
def test_rho_u_sampling(segmentation_model):
    Su_index = 1
    Su_begin, Su_end = segmentation_model.get_Su_begin_end(Su_index)
    print(segmentation_model.rho)
    segmentation_model.sample_rho_u(Su_end-1, Su_index)
    
def test_rho_sampling(rnd_topics_model, outFile):
    '''
    Just trying to see if the code runs and
    then check the consistency of the state
    after sampling rho
    '''
    rnd_topics_model.sample_rho()
    test_model_state(rnd_topics_model, outFile)
    
def run_gibbs_sampler(rnd_topics_model, n_iter, burn_in, lag, sampler_log_file="logging/Sampler.log"):
    sampler = SegmentationModelSampler(rnd_topics_model, sampler_log_file)
    wd = sampler.gibbs_sampler(n_iter, burn_in, lag)
    return wd, sampler

def print_matrix_heat_map(matrix, title, outFile):
    ax = plt.axes()
    sns.heatmap(matrix, ax = ax, cmap='RdYlGn_r')
    ax.set_title(title)
    plt.xlabel('Topics', fontsize=14)
    plt.ylabel('Segments', fontsize=14)
    plt.savefig(outFile, bbox_inches='tight')
    plt.clf()
    
def print_matrix(matrix):
    str_final = ""
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            str_val = str(matrix[i, j])
            str_split = str_val.split("e")
            if len(str_split[0]) > 4:
                str_r = str_split[0][:4]
            else:
                str_r = str_split[0]
            str_final += str_r
            
            if len(str_split) == 2:
                str_final += "e" + str_split[1]
            else:
                str_final += "\t"
            str_final += "\t"
        str_final += "\n"
    return str_final

def plot_topic_assign(k_counts, inv_vocab, n, color, out_file):
    plt.clf()
    f, axarr = plt.subplots(1, squeeze=False)
    labelsVals = {}
    for i, val in enumerate(k_counts):
        labelsVals[inv_vocab[i]] = val
    sorted_dic = sorted(labelsVals.items(), key=operator.itemgetter(1))
    yLabels = [label for label, val in sorted_dic[:n]]
    y_pos = np.arange(len(yLabels))/5.
    ax_i = 0
    axarr[ax_i, 0].barh(y_pos, [labelsVals[y] for y in yLabels], align='center', color=color, height=0.2)
    axarr[ax_i, 0].set_yticks(y_pos)
    axarr[ax_i, 0].set_yticklabels(yLabels)

    plt.xlabel('Counts')
    plt.savefig(out_file)
    
def debug_topic_assign(sampler, outDir):
    inv_vocab =  {v: k for k, v in sampler.seg_model.doc.vocab.items()}
    n = len(sampler.seg_model.doc.vocab)
    for k in range(sampler.seg_model.K):
        k_counts = sampler.estimated_W_K_counts[:, k]
        plot_topic_assign(k_counts.toarray().T[0], inv_vocab, n, 'g', outDir + "k_w_counts" + str(k) + ".png")
        
def plot_log_joint_prob(log_dir, outFile):
    plt.clf()
    y_labels = []
    conv_fig = plt.figure(1)
    ax_lp = conv_fig.add_subplot(211)
    ax_lp.set_ylabel('log prob joint')
    ax_lp.set_xlabel('Iteration')

    ax_wd = conv_fig.add_subplot(212)
    ax_wd.set_ylabel('WD')
    ax_wd.set_xlabel('Iteration')
    
    for sampler_log_file in os.listdir(log_dir):
        if not sampler_log_file.startswith("S"):
            continue
        with open(log_dir+sampler_log_file) as r_file:
            y_lp = []
            y_wd = []
            lins = r_file.readlines()
            for lin in lins:
                if lin.startswith("INFO:log_prob_joint"):
                    y_lp.append(float(lin.split("INFO:log_prob_joint ")[1]))
                if lin.startswith("INFO:final_wd"):
                    final_wd = "%.2f" % float(lin.split("INFO:final_wd: ")[1])
                    y_labels.append(sampler_log_file[:-4] + " " + final_wd)
                if lin.startswith("INFO:Rho_Est"):
                    current_iter_wd = float(lin.split("INFO:Rho_Est ")[1])
                    y_wd.append(current_iter_wd)
            
            x = range(len(y_lp))
                
            ax_lp.plot(x, y_lp)
            ax_wd.plot(x, y_wd)
    ax_lp.legend(y_labels, loc='lower right')
    ax_wd.legend(y_labels, loc='upper right')
    
    conv_fig.savefig(outFile+".png")
    
def plot_rho_u_prob(sampler_log_file, outDir):
    y_vals_dic = {}
    with open(sampler_log_file) as r_file:
        lins = r_file.readlines()
        for lin in lins:
            if lin.startswith("INFO:sample_rho_u:"):
                u = lin.split(" u ")[1].split(" ")[0]
                if u not in y_vals_dic:
                    y_vals_dic[u] = []
                y_vals_dic[u].append(float(lin.split(" prob_1 ")[1]))
    
    n_iters = len(y_vals_dic["0"])
    for u in y_vals_dic:
        plt.clf()
        rho_fig = plt.figure(1)
        ax_rho = rho_fig.add_subplot(211)
        ax_rho.set_ylabel('Prob rho = 1')
        ax_rho.set_xlabel('Iteration')
        x = range(n_iters)
        ax_rho.plot(x, y_vals_dic[u])
        rho_fig.savefig(outDir + "rho_u"+u+"_prob.png")
