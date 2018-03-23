#!/usr/bin/env python3
import numpy as np
import gensim
from gensim.models.doc2vec import LabeledSentence
import os

docs = [f for f in os.listdir('aclImdb/train/neg/') if f.endswith('.txt')]
data = []
for doc in docs:
    f = open('aclImdb/train/neg/'+doc, 'r')
    data.append(f.read())
    f.close()

tmp = []
for f in os.listdir('aclImdb/train/pos/'):
    if f.endswith('.txt'):
        docs.append(f)
        tmp.append(f)

for doc in tmp:
    f = open('aclImdb/train/pos/'+doc, 'r')
    data.append(f.read())
    f.close()

tmp = []
for f in os.listdir('aclImdb/test/pos/'):
    if f.endswith('.txt'):
        docs.append(f)
        tmp.append(f)
for doc in tmp:
    f = open('aclImdb/test/pos/'+doc, 'r')
    data.append(f.read())
    f.close()

tmp = []
for f in os.listdir('aclImdb/test/neg/'):
    if f.endswith('.txt'):
        docs.append(f)
        tmp.append(f)
for doc in tmp:
    f = open('aclImdb/test/neg/'+doc, 'r')
    data.append(f.read())
    f.close()


class LabelledLineSentence(object):
    def __init__(self, doc_list, labels_list):
        self.labels_list = labels_list
        self.doc_list = doc_list

    def __iter__(self):
        for idx, doc in enumerate(self.doc_list):
            yield LabeledSentence(words=doc.split(), tags=[self.labels_list[idx]])

it = LabelledLineSentence(data, docs)
model = gensim.models.Doc2Vec(vector_size=300, window=10, min_count=5, workers=11, alpha=0.025, min_alpha=0.025) #Use fixed learning rate
model.build_vocab(it)

# Train
for epoch in range(20):
    model.train(it, total_examples=model.corpus_count, epochs=model.epochs)

model.save('doc2vec.model')
