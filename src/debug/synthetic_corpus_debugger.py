'''
Created on Jan 23, 2017

@author: root
'''
from dataset.corpus import SyntheticTopicTrackingDoc, SyntheticRndTopicPropsDoc
import seaborn as sns
import matplotlib.pyplot as plt

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
    print(str_final)
        
    
def debug(vocab_dic, doc_synth_tt, desc, outDir, flags):
    theta = doc_synth_tt.theta.toarray()
    if flags[0]:
        print_matrix(theta)
    if flags[1]:
        print_matrix_heat_map(theta, desc + " Theta", outDir + desc + "_theta_heat_map.png")
    if flags[2]:
        print(doc_synth_tt.getText(vocab_dic))
        
pi = 0.2
alpha = 10
beta = 0.6
K = 10
W = 15
n_sents = 50
sentence_l = 50
vocab_dic = {}
for w in range(W):
    vocab_dic[w] = "w" + str(w)
outDir = "debug/"    

print_theta_flag = False
print_heat_map_flag = True
print_text_flag = False
flags = [print_theta_flag, print_heat_map_flag, print_text_flag]

doc_synth_tt = SyntheticTopicTrackingDoc(pi, alpha, beta, K, W, n_sents, sentence_l)
doc_synth_tt.generate_doc()
debug(vocab_dic, doc_synth_tt, "Topic Tracking", outDir, flags)

doc_synth_rnd_tp = SyntheticRndTopicPropsDoc(pi, alpha, beta, K, W, n_sents, sentence_l)
doc_synth_rnd_tp.generate_doc()
debug(vocab_dic, doc_synth_rnd_tp, "Random Topic Proportions", outDir, flags)
