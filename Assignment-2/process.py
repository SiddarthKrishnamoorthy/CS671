#!/usr/bin/env python3
import numpy as np
import os
import re
import _pickle as pkl
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

def stemmer(data):
    stop_words = ['<', '>', '/', 'br']
    tokens = word_tokenize(data)
    stop_word_set = stopwords.words('english')
    output = [x.lower() for x in tokens if x not in stop_words and stop_word_set]
    return output

train = {}
test = {}
train[1] = {}
train[0] = {}
for filename in os.listdir('aclImdb/train/neg/'):
    if filename.endswith('.txt'):
        #print(filename)
        regex = re.search('([0-9]+)_([0-9]+).txt', filename)
        f = str(regex.group(1))
        data = open('aclImdb/train/neg/'+filename, 'r').read()
        train[0][f] = stemmer(data)

for filename in os.listdir('aclImdb/train/pos/'):
    if filename.endswith('.txt'):
        #print(filename)
        regex = re.search('([0-9]+)_([0-9]+).txt', filename)
        f = str(regex.group(1))
        data = open('aclImdb/train/pos/'+filename, 'r').read()
        train[1][f] = stemmer(data)

test[0] = {}
test[1] = {}
for filename in os.listdir('aclImdb/test/neg/'):
    if filename.endswith('.txt'):
        #print(filename)
        regex = re.search('([0-9]+)_([0-9]+).txt', filename)
        f = str(regex.group(1))
        data = open('aclImdb/test/neg/'+filename, 'r').read()
        test[0][f] = stemmer(data)

for filename in os.listdir('aclImdb/test/pos/'):
    if filename.endswith('.txt'):
        #print(filename)
        regex = re.search('([0-9]+)_([0-9]+).txt', filename)
        f = str(regex.group(1))
        data = open('aclImdb/test/pos/'+filename, 'r').read()
        test[1][f] = stemmer(data)

data = [test, train]
out_file = open('data.pkl', 'wb')
pkl.dump(data, out_file)
out_file.close()
