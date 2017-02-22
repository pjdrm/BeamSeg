'''
Created on Feb 16, 2017

@author: root
'''
import numpy as np
from scipy import sparse, int32
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
import nltk.stem

class Document(object):
    def __init__(self, doc_path, max_features, stem=False, min_tf = 3):
        #These matrixes are here for debug compatibility
        self.U_K_counts = sparse.csr_matrix((1, 1), dtype=int32)
        self.U_I_topics = sparse.csr_matrix((1, 1), dtype=int32)
        self.W_K_counts = sparse.csr_matrix((1, 1), dtype=int32)
        self.load_doc(doc_path, max_features, stem, min_tf)
        
    def load_doc(self, doc_path, max_features, stem, min_tf):
        self.rho = []
        sents = []
        with open(doc_path) as doc_file:
            lins = doc_file.readlines()[1:-1]
            for lin in lins:
                if lin == "==========\n":
                    self.rho[-1] = 1
                else:
                    self.rho.append(0)
                    sents.append(lin)
        
            self.n_sents = len(sents)
            self.rho_eq_1 = np.append(np.nonzero(self.rho)[0], [self.n_sents-1])
            self.n_segs = len(self.rho_eq_1)
            
            if stem:
                vectorizer = ENStemmedCountVectorizer(min_tf, max_features=max_features)
            else:
                vectorizer = CountVectorizer(analyzer = "word",\
                                             strip_accents = "unicode",\
                                             stop_words = stopwords.words("english"),\
                                             max_features=max_features, min_df = min_tf)
                
            self.U_W_counts = vectorizer.fit_transform(sents)
            self.vocab = vectorizer.vocabulary_
            self.W = len(self.vocab)
            self.sents_len = np.sum(self.U_W_counts, axis = 1).A1
            self.U_I_words = sparse.csr_matrix((self.n_sents, max(self.sents_len)), dtype=int32)
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
                        
class ENStemmedCountVectorizer(CountVectorizer):
    def __init__(self, min_tf, max_features):
        CountVectorizer.__init__(self,analyzer = "word",\
                                 strip_accents = "unicode",\
                                 stop_words = stopwords.words("english"),\
                                 max_features=max_features,
                                 min_df = min_tf)
        self.en_stemmer = nltk.stem.SnowballStemmer('english')
        
    def build_analyzer(self):
        analyzer = super(ENStemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([self.en_stemmer.stem(w) for w in analyzer(doc)])