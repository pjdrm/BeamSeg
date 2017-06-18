'''
Created on Feb 16, 2017

@author: root
'''
import numpy as np
from scipy import sparse, int32
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk.stem
import os

#add_to_stop_words = ["object", "time", "zero"]
#add_to_stop_words = ["object", "time", "want", "one", "velocity", "would"]
#add_to_stop_words = ["object", "time", "want", "one", "velocity", "would", "positive", "negative"] -config for 3 seg experiment k = 2
class Document(object):
    def __init__(self, doc_path, configs):
        max_features = configs["real_data"]["max_features"]
        lemmatize = eval(configs["real_data"]["lemmatize"])
        min_tf = configs["real_data"]["min_tf"]
        max_w_percent = configs["real_data"]["max_w_percent"]
        max_dispersion = configs["real_data"]["max_dispersion"]
        filter_words_flag = eval(configs["real_data"]["filter_words_flag"])
        self.remove_from_stop_words = configs["real_data"]["remove_from_stop_words"]
        self.add_to_stop_words = configs["real_data"]["add_to_stop_words"]
        
        self.isMD = False
        self.K = 2
        
        #These matrixes are here for debug compatibility
        self.U_K_counts = sparse.csr_matrix((1, 1), dtype=int32)
        self.U_I_topics = sparse.csr_matrix((1, 1), dtype=int32)
        self.W_K_counts = sparse.csr_matrix((1, 1), dtype=int32)
        
        self.my_stopwords = self.load_sw(doc_path, lemmatize, min_tf)
        self.load_doc(doc_path, max_features, lemmatize)
        
        if filter_words_flag:    
            self.filter_words(max_w_percent, max_dispersion)
            #Talking about ineficiency...
            '''
            filter_words adds to the global variables to_stop_words.
            Basically I have to reload the corpus all over again with the
            new list of stopwords.
            
            The thing is that to calculate the words to be filtered I need
            the matrixes obtained by loading the corpus.
            '''
            self.load_doc(doc_path, max_features, lemmatize, min_tf)
        self.del_ghost_lines() 
    
    def process_doc(self, doc_path):
        rho = []
        sents = []
        with open(doc_path) as doc_file:
            lins = doc_file.readlines()[1:-1]
            for lin in lins:
                if lin == "==========\n":
                    rho[-1] = 1
                else:
                    rho.append(0)
                    sents.append(lin)
        return rho, sents
                       
    def load_doc(self, doc_path, max_features, lemmatize):
        self.rho, sents = self.process_doc(doc_path)
        self.n_sents = len(sents)
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
        self.n_segs = len(self.rho_eq_1)
        
        if lemmatize:
            vectorizer = ENLemmatizerCountVectorizer(self.my_stopwords, max_features=max_features)
        else:
            vectorizer = CountVectorizer(analyzer = "word",\
                                         strip_accents = "unicode",\
                                         stop_words = self.my_stopwords,\
                                         max_features=max_features)
            
        self.U_W_counts = vectorizer.fit_transform(sents).toarray()
        self.vocab = vectorizer.vocabulary_
        self.inv_vocab =  {v: k for k, v in self.vocab.items()}
        self.W = len(self.vocab)
        self.sents_len = np.sum(self.U_W_counts, axis = 1)
        self.U_I_words = np.zeros((self.n_sents, max(self.sents_len)), dtype=int32)
        '''
        This part is not efficient, but I have figure out a way to use CountVectorizer
        and obtain the word sequence that I need for the U_I_words variable.
        
        I am assuming that by using build_analyzer I obtain the same
        preprocessing used in CountVectorizer. Then, filtering by vocab
        indicates which words were discarded.
        '''
        analyzer = vectorizer.build_analyzer()
        for u_index, u in enumerate(sents):
            u_I = analyzer(u)
            i = 0
            for w_ui in u_I:
                if w_ui in self.vocab:
                    self.U_I_words[u_index, i] = self.vocab[w_ui]
                    i += 1
                        
    def load_sw(self, doc_path, lemmatize, min_tf):
        sw_list = stopwords.words("english")
        sw_list += self.add_to_stop_words
        
        rho, sents = self.process_doc(doc_path)
        #It seems that lemmatization takes place after sw removal
        #we need to specify sw in the unlematized form
        vectorizer = CountVectorizer(analyzer = "word",\
                                         strip_accents = "unicode")
        U_W_counts = vectorizer.fit_transform(sents)
        vocab = vectorizer.vocabulary_
        inv_vocab =  {v: k for k, v in vocab.items()}
        w_total_counts = np.sum(U_W_counts, axis=0).A1
        filter_words = []
        
        for w in range(U_W_counts.shape[1]):
            if inv_vocab[w] == "le":
                print()
            if w_total_counts[w] < min_tf:
                filter_words.append(inv_vocab[w])
        sw_list += filter_words
        sw_list = [sw for sw in sw_list if sw not in self.remove_from_stop_words]
        return sw_list
        
    def filter_words(self, max_w_percent, max_dispersion):
        word_chains_dic = {}
        for u in range(self.n_sents):
            for i in range(self.sents_len[u]):
                w_ui = self.U_I_words[u,i]
                if w_ui not in word_chains_dic:
                    word_chains_dic[w_ui] = []
                word_chains_dic[w_ui].append(u)
                
        self.disp_scores_dic = {}
        self.disp_scores_dic2 = {}
        self.n_w_percent_dic = {}
        self.n_w_percent_dic2 = {}
        w_to_filter = []
        n_words = self.sents_len.sum()
        for w in word_chains_dic.keys():
            num = 0.0
            w_i_list = word_chains_dic[w]
            for j in range(len(w_i_list)-1):
                i = w_i_list[j]
                i_plus_1 = w_i_list[j+1]
                num += i_plus_1 - i
            w_str = self.inv_vocab[w]
            disp_score = num / (len(w_i_list)-1)
            self.disp_scores_dic[w] = disp_score
            self.disp_scores_dic2[w_str] = disp_score
            if disp_score > max_dispersion:
                w_to_filter.append(w_str)
            
            n_w_percent = len(w_i_list) / float(n_words)
            self.n_w_percent_dic[w] = n_w_percent
            self.n_w_percent_dic2[w_str] = n_w_percent
            if n_w_percent > max_w_percent:
                w_to_filter.append(w_str)
        self.add_to_stop_words += w_to_filter
            
    '''
    Boundary ghost lines are lines with all word counts equal to 0.
    I found these particular lines to badly affect inference, thus,
    I print a warning if I find them.
    '''                    
    def del_ghost_lines(self):
        self.ghost_lines = np.where(~self.U_W_counts.any(axis=1))[0]
        boundary_ghost_lines = np.intersect1d(self.rho_eq_1, self.ghost_lines)
        if len(boundary_ghost_lines) > 0:
            print("WARNING: the following ghost lines match a boundary: %s" % (str(boundary_ghost_lines)))
            '''
            The current fix to ghost lines is to consider
            the previous line as boundary instead.
            '''
            for b_gl in boundary_ghost_lines:
                if b_gl-1 in boundary_ghost_lines:
                    print("WARNING: Oh no another boundary ghost line...")
                self.rho[b_gl-1] = 1
        
        self.n_sents -= len(self.ghost_lines)
        self.U_W_counts = np.delete(self.U_W_counts, self.ghost_lines, axis=0)
        self.U_I_words = np.delete(self.U_I_words, self.ghost_lines, axis=0)
        self.rho = np.delete(self.rho, self.ghost_lines, axis=0)
        self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
        self.n_segs = len(self.rho_eq_1)
        self.sents_len = np.sum(self.U_W_counts, axis = 1)
                
class MultiDocument(Document):
    def __init__(self, configs):
        self.doc_names = []
        self.docs_index =[]
        doc_path = "tmp_docs.txt"
        self.prepare_multi_doc(configs["real_data"]["docs_dir"], doc_path)
        Document.__init__(self, doc_path, configs)
        self.update_doc_index()
        os.remove(doc_path)
        self.isMD = True
        
    def prepare_multi_doc(self, doc_dir, doc_tmp_path):
        str_cat_files = ""
        doc_offset = 0
        for doc in os.listdir(doc_dir):
            self.doc_names.append(doc)
            with open(os.path.join(doc_dir, doc), encoding="utf-8", errors='ignore') as f:
                str_doc = f.read()
                str_cat_files += str_doc[:-10]
                doc_len = (str_doc.count("\n")+1) - str_doc.count("==========")
                doc_offset += doc_len
                self.docs_index.append(doc_offset)
        self.n_docs = len(self.docs_index)
        str_cat_files += "=========="
        with open(doc_tmp_path, "w+") as f_out:
            f_out.write(str_cat_files)
    
    #TODO: REALLY CHECK THIS IS CORRECT
    def update_doc_index(self):
        updated_doc_index = []
        c = 0
        doc_index = self.docs_index.pop(0)
        checked_indexes = []
        for gl in self.ghost_lines:
            if gl > doc_index - 1:
                checked_indexes.append(doc_index)
                updated_doc_index.append(doc_index-c)
                doc_index = self.docs_index.pop(0)
            c += 1
        if not doc_index in checked_indexes:
            updated_doc_index.append(doc_index-c)
        for doc_index in self.docs_index:
            updated_doc_index.append(doc_index-c)
        self.docs_index = updated_doc_index
                        
class ENLemmatizerCountVectorizer(CountVectorizer):
    def __init__(self, stopwords_list=None, max_features=None):
        CountVectorizer.__init__(self,analyzer = "word",\
                                 strip_accents = "unicode",\
                                 stop_words = stopwords_list,\
                                 max_features=max_features)
        self.en_lemmatizer = nltk.stem.WordNetLemmatizer()
        
    def build_analyzer(self):
        analyzer = super(ENLemmatizerCountVectorizer, self).build_analyzer()
        return lambda doc: ([self.en_lemmatizer.lemmatize(w) for w in analyzer(doc)])