#!/usr/bin/env python3
import numpy as np
import gensim

def loadGlove():
    f = open("")
    model = {}
    for l in f:
        splitLine = l.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding

    return model

def loadWord2Vec():
    path = ""
    model = gensim.models.Word2Vec.load_word2vec_format(path, binary=True)
    return model
