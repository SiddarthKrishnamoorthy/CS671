#!/usr/bin/env python3

import numpy as np    
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from sklearn.neural_network import MLPClassifier
from gensim.scripts.glove2word2vec import glove2word2vec
from loader import load_data, rdep, feature_create, trans_feature
from loader import get_feature, ldep, dependency_relations

def get_conf(sentence, parse):
    configuration = []
    transition = []
    for i in range(len(sentence)):
        config = []
        transit = []

        stack = [0]
        buff = [i for i in range(1,len(parse[i])+1)]
        edge = []
        dg = dependency_relations(parse[i])
        dg.sort()
        terminal_conf = [[0], [], dg]
        while [stack, buff, edge] != terminal_conf:
            config.append([stack, buff, edge])
            if len(stack)>0 and len(buff)>0:
                if [buff[0], stack[-1]] in dg:
                    transit.append('la')
                    edge.append([buff[0], stack[-1]])
                    stack = stack[:-1]        # pop topmost element
                    buff = buff

                elif [stack[-1], buff[0]] in dg:        # right-arc transition
                    flag = 0
                    for w in range(1,len(parse[i])+1):
                        if [buff[0], w] in dg:
                            if [buff[0], w] not in edge:
                                flag = 1
                                break
                    if flag == 1:
                        stack.append(buff[0])    # push top of buff to stack
                        buff = buff[1:]
                        edge = edge
                        transit.append('shift')
                    else:
                        transit.append('ra')
                        edge.append([stack[-1], buff[0]])
                        buff[0] = stack[-1]        # replace top of buff with top of stack
                        stack = stack[:-1]

                else:
                    transit.append('shift')
                    stack.append(buff[0])        # push top of buff to stack
                    buff = buff[1:]
                    edge = edge
                
            elif len(buff) > 0:
                transit.append('shift')
                stack.append(buff[0])        # push top of buff to stack
                buff = buff[1:]
                edge = edge
            else:            # dg may be non-projective
                config = config[:-1]
                break
            edge.sort()
        configuration.append(config)
        transition.append(transit)

    return configuration, transition

glove2word2vec(glove_input_file='glove.6B.50d.txt', word2vec_output_file='gensim_glove_vectors.txt')
model = KeyedVectors.load_word2vec_format('gensim_glove_vectors.txt',binary=False)

POSTAG = {'ADJ':1, 'ADP':2, 'ADV':3, 'AUX':4, 'CCONJ':5, 'DET':6, 'INTJ':7,
        'NOUN':8, 'NUM':9, 'PART':10, 'PRON':11, 'PROPN':12, 'PUNCT':13,
        'SCONJ':14, 'SYM':15, 'VERB':16, 'X':17}

sent_train, parse_train = load_data('en_ewt-ud-train.conllu')
sent_test, parse_test = load_data('en_ewt-ud-test.conllu')

config_train, transit_train = get_conf(sent_train, parse_train)
config_test, transit_test = get_conf(sent_test, parse_test)

X_train = feature_create(config_train,sent_train, parse_train)
y_train = trans_feature(transit_train)

X_test = feature_create(config_test, sent_test, parse_test)
y_test = trans_feature(transit_test)

model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam')
model.fit(X_train, y_train)
model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_val, y_val), verbose=1)

print(model.score(X_test, y_test))
