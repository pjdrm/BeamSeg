'''
Created on Jan 30, 2017

@author: root
'''
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
