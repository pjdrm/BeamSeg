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
import dataset.synthetic_doc as syn_doc
import copy
import operator
from audioop import reverse

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
        self.W_I_words = []
        self.d_u_wi_indexes = []
        '''
        This part is not efficient, but I have figure out a way to use CountVectorizer
        and obtain the word sequence that I need for the U_I_words variable.
        
        I am assuming that by using build_analyzer I obtain the same
        preprocessing used in CountVectorizer. Then, filtering by vocab
        indicates which words were discarded.
        '''
        analyzer = vectorizer.build_analyzer()
        word_count = 0
        doc_i_u = []
        for u_index, u in enumerate(sents):
            u_w_indexes = []
            u_I = analyzer(u)
            i = 0
            for w_ui in u_I:
                if w_ui in self.vocab:
                    u_w_indexes.append(word_count)
                    word_count += 1
                    vocab_index = self.vocab[w_ui]
                    self.U_I_words[u_index, i] = vocab_index
                    self.W_I_words.append(vocab_index)
                    i += 1
                    
            if len(u_w_indexes) > 0:
                doc_i_u.append(u_w_indexes)
            if u_index+1 in self.docs_index:
                self.d_u_wi_indexes.append(doc_i_u)
                doc_i_u = []
        self.W_I_words = np.array(self.W_I_words)
                        
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
        w_total_counts = np.array(np.sum(U_W_counts, axis=0))[0]
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
        self.doc_topic_seq, self.doc_rho_topics, self.max_topics = self.load_doc_topic_seq(configs["real_data"]["doc_links_dir"])
        self.seg_dur_prior = self.get_prior()
        self.print_processed_docs(configs["real_data"]["docs_processed_dir"])
        
    def print_processed_docs(self, out_dir):
        u_i = 0
        for doc_i, doc_name in enumerate(self.doc_names):
            doc_i_str = "==========\n"
            for u in self.d_u_wi_indexes[doc_i]:
                for wi in u:
                    word = self.inv_vocab[self.W_I_words[wi]]
                    doc_i_str += word + " "
                doc_i_str += "\n"
                if u_i in self.rho_eq_1:
                    doc_i_str += "==========\n"
                u_i += 1
            with open(out_dir+doc_name, "w+") as f:
                f.write(doc_i_str)
        
    def get_prior(self):
        prior_docs = []
        seg_lens = []
        seg_len = 0
        cp_rho = copy.deepcopy(self.rho)
        cp_rho[-1] = 1
        for u, rho in enumerate(cp_rho):
            seg_len += 1
            if rho == 1:
                seg_lens.append(seg_len)
                seg_len = 0
                
            if u+1 in self.docs_index:
                prior_docs.append([np.average(seg_lens), np.std(seg_lens)])
                seg_lens = []
        return prior_docs
        
    def find_target_docs(self, doc_names, docs_topic_seq):
        for i, doc_name in enumerate(doc_names):
            if "ref" in doc_name:
                doc_ref = i
                break
        
        overlap_dict = {}
        for i, doc_name in enumerate(doc_names):
            if i == doc_ref:
                continue
            overlap_dict[doc_name] = 0
            for k in docs_topic_seq[i]:
                if k in docs_topic_seq[doc_ref]:
                    overlap_dict[doc_name] += 1
        sorted_docs = sorted(overlap_dict.items(), key=operator.itemgetter(1), reverse=True)
        for doc_name, count in sorted_docs:
            print("%s %s" % (doc_name, count))
        
    def load_doc_topic_seq(self, links_dir):
        topic_dict = {}
        docs_topic_seq = []
        for i, dir_name in enumerate(os.listdir(links_dir)):
            topic_dict[dir_name] = i
            
        for doc_name in self.doc_names:
            if "processed" in doc_name:
                doc_name_split = doc_name.split("_")
                doc_name = doc_name_split[0]+"_"+doc_name_split[1]+"_"
            else:
                doc_name = doc_name[:-4]
            topic_seq = []
            topic_seq_dict = {}
            for dir_name in os.listdir(links_dir):
                for doc_seg in os.listdir(links_dir+"/"+dir_name):
                    if doc_name in doc_seg:
                        i = int(doc_seg.split("_seg")[1].split(".txt")[0])
                        topic_seq_dict[i] = topic_dict[dir_name]
            for i in range(1, len(topic_seq_dict.keys())+1):
                topic_seq.append(topic_seq_dict[i])
            docs_topic_seq.append(topic_seq)
        
        #self.find_target_docs(self.doc_names, docs_topic_seq)
        doc_i = 0
        doc_rho_topics = []
        doc_i_rho_topics = []
        i = 0
        for u, rho_u in enumerate(self.rho):
            doc_i_rho_topics.append(docs_topic_seq[doc_i][i])
            if rho_u == 1:
                i += 1
            if u+1 in self.docs_index:
                i = 0
                doc_i += 1
                doc_rho_topics.append(doc_i_rho_topics)
                doc_i_rho_topics = []
        
        topics_set = set()
        for doc_i_rho_topics in doc_rho_topics:
            for topic in doc_i_rho_topics:
                topics_set.add(topic)
        max_topics = len(topics_set)
        
        return docs_topic_seq, doc_rho_topics, max_topics
    
    def prepare_multi_doc(self, doc_dir, doc_tmp_path):
        str_cat_files = ""
        doc_offset = 0
        docs_file_names = ['L03_7_processed_annotated_html.txt',
                            'L03_342_processed_annotated_pdf.txt',
                            'L03_48_processed_annotated_html.txt',
                            'L03_365_processed_annotated_pdf.txt',
                            'L03_185_processed_annotated_html.txt',
                            'L03_402_processed_annotated_ppt.txt',
                            'L03_v19_cap_man_processed_annotated.txt',
                            'L03_239_processed_annotated_html.txt',
                            'L03_213_processed_annotated_html.txt',
                            'L03_vref_cap_man_processed_annotated.txt']
        #sorted(docs_file_names)
        for doc in docs_file_names:
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
        carry = 0
        gl_index = 0
        for doc_index in self.docs_index:
            updated_index = doc_index-carry
            for i in range(gl_index, len(self.ghost_lines)):
                if self.ghost_lines[i] < doc_index:
                    updated_index -= 1
                    carry += 1
                    gl_index += 1
                    if gl_index == len(self.ghost_lines):
                        break
            updated_doc_index.append(updated_index)
        self.docs_index = updated_doc_index
        
    def get_single_docs(self):
        indv_docs = syn_doc.multi_doc_slicer(self)
        return indv_docs
                        
class ENLemmatizerCountVectorizer(CountVectorizer):
    def __init__(self, stopwords_list=None, max_features=None):
        CountVectorizer.__init__(self,analyzer="word",\
                                 strip_accents="unicode",\
                                 stop_words=stopwords_list,\
                                 max_features=max_features)
        self.en_lemmatizer = nltk.stem.WordNetLemmatizer()
        
    def build_analyzer(self):
        analyzer = super(ENLemmatizerCountVectorizer, self).build_analyzer()
        return lambda doc: ([self.en_lemmatizer.lemmatize(w) for w in analyzer(doc)])