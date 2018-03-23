#!/usr/bin/env python3
import numpy as np
import gensim
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def loadGlove():
    f = open("GloVe/glove.6B.300d.txt")
    model = {}
    for l in f:
        splitLine = l.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding

    return model

def loadWord2Vec():
    path = "Word2Vec/GoogleNews-vectors-negative300-SLIM.bin"
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    return model
