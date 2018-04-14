#!/usr/bin/env python3
import numpy as np
import random
import _pickle as pkl
from classifiers import SVM, LR, NB, NN, nultinomialNB
from data_loader import loadGlove, loadWord2Vec
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import sys
from gensim.models.doc2vec import Doc2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.preprocessing import sequence

if(sys.argv[1] == 'glove'):
    glove = loadGlove()
    f = open('data.pkl', 'rb')
    [test, train] = pkl.load(f)
    f.close()

    train_set = []
    for key, value in train.items():
        for f_no, words in value.items():
            avg = np.zeros(300)
            #ctr = len(words)
            for i in range(0, len(words)):
                try:
                    avg += glove[train[key][f_no][i]]
                except:
                    continue
            avg = avg/len(words)
            train_set.append([key, avg])

    random.shuffle(train_set)
    X_train = [x[1] for x in train_set]
    Y_train = [x[0] for x in train_set]
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    test_set = []
    for key, value in test.items():
        for f_no, words in value.items():
            avg = np.zeros(300)
            #ctr = len(words)
            for i in range(0, len(words)):
                try:
                    avg += glove[test[key][f_no][i]]
                except:
                    continue
            avg = avg/len(words)
            test_set.append([key, avg])

    random.shuffle(test_set)
    X_test = [x[1] for x in test_set]
    Y_test = [x[0] for x in test_set]
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

if sys.argv[1] == 'paragraph':
    train_set = []
    for f in os.listdir('aclImdb/train/neg/'):
        if f.endswith('.txt'):
            train_set.append([0, f])
    for f in os.listdir('aclImdb/train/pos/'):
        if f.endswith('.txt'):
            train_set.append([1,f])

    random.shuffle(train_set)

    model = Doc2Vec.load('doc2vec.model')
    X_train = [model[x[1]] for x in train_set]
    Y_train = [x[0] for x in train_set]
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    test_set = []
    for f in os.listdir('aclImdb/test/neg/'):
        if f.endswith('.txt'):
            test_set.append([0, f])
    for f in os.listdir('aclImdb/test/pos/'):
        if f.endswith('.txt'):
            test_set.append([1,f])

    random.shuffle(test_set)

    model = Doc2Vec.load('doc2vec.model')
    X_test= [model[x[1]] for x in test_set]
    Y_test= [x[0] for x in test_set]
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)
    #print('tset')

if sys.argv[1] == 'bbow':
    print('test')
    f = open('aclImdb/imdb.vocab').read()
    stop_words = ['<', '>', '/', 'br']
    tokens = word_tokenize(f)
    print('test')
    stop_word_set = stopwords.words('english')
    vocab = [x.lower() for x in tokens if x not in stop_words and stop_word_set]
    print('test')
    vocab = set(vocab)
    vocab = list(vocab)
    print('test')
    vectorizer = CountVectorizer(vocabulary=vocab, binary=True)
    print('test')

    train_set = load_files('aclImdb/train/', categories=['pos', 'neg'], shuffle=True)
    print('test')
    test_set = load_files('aclImdb/test/', categories=['pos', 'neg'], shuffle=True)
    print('test')
    Y_train = train_set.target
    print('test')
    Y_test= test_set.target
    print('test')
    X_train = vectorizer.transform(train_set.data)
    #X_train = X_train.toarray()
    print('test')
    X_test = vectorizer.transform(test_set.data)
    #X_test = X_test.toarray()
    print('test')

if sys.argv[1] == 'tf':
    print('test')
    f = open('aclImdb/imdb.vocab').read()
    stop_words = ['<', '>', '/', 'br']
    tokens = word_tokenize(f)
    print('test')
    stop_word_set = stopwords.words('english')
    vocab = [x.lower() for x in tokens if x not in stop_words and stop_word_set]
    print('test')
    vocab = set(vocab)
    vocab = list(vocab)
    print('test')
    vectorizer = TfidfVectorizer(vocabulary=vocab, norm='l1', use_idf=False)
    print('test')

    train_set = load_files('aclImdb/train/', categories=['pos', 'neg'], shuffle=True)
    print('test')
    test_set = load_files('aclImdb/test/', categories=['pos', 'neg'], shuffle=True)
    print('test')
    Y_train = train_set.target
    print('test')
    Y_test= test_set.target
    print('test')
    X_train = vectorizer.transform(train_set.data)
    print('test')
    X_test = vectorizer.transform(test_set.data)
    print('test')

if sys.argv[1] == 'tfidf':
    print('test')
    f = open('aclImdb/imdb.vocab').read()
    stop_words = ['<', '>', '/', 'br']
    tokens = word_tokenize(f)
    print('test')
    stop_word_set = stopwords.words('english')
    vocab = [x.lower() for x in tokens if x not in stop_words and stop_word_set]
    print('test')
    vocab = set(vocab)
    vocab = list(vocab)
    print('test')
    vectorizer = TfidfVectorizer(vocabulary=vocab, norm='l1', use_idf=True)
    print('test')

    train_set = load_files('aclImdb/train/', categories=['pos', 'neg'], shuffle=True)
    print('test')
    test_set = load_files('aclImdb/test/', categories=['pos', 'neg'], shuffle=True)
    print('test')
    Y_train = train_set.target
    print('test')
    Y_test= test_set.target
    print('test')
    X_train = vectorizer.fit_transform(train_set.data)
    print('test')
    X_test = vectorizer.fit_transform(test_set.data)
    print('test')

if sys.argv[1] == 'word2vec':
    glove = loadWord2Vec()
    f = open('data.pkl', 'rb')
    [test, train] = pkl.load(f)
    f.close()

    train_set = []
    for key, value in train.items():
        for f_no, words in value.items():
            avg = np.zeros(300)
            #ctr = len(words)
            for i in range(0, len(words)):
                try:
                    avg += glove[train[key][f_no][i]]
                except:
                    continue
            avg = avg/len(words)
            train_set.append([key, avg])

    random.shuffle(train_set)
    X_train = [x[1] for x in train_set]
    Y_train = [x[0] for x in train_set]
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    test_set = []
    for key, value in test.items():
        for f_no, words in value.items():
            avg = np.zeros(300)
            #ctr = len(words)
            for i in range(0, len(words)):
                try:
                    avg += glove[test[key][f_no][i]]
                except:
                    continue
            avg = avg/len(words)
            test_set.append([key, avg])

    random.shuffle(test_set)
    X_test = [x[1] for x in test_set]
    Y_test = [x[0] for x in test_set]
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

if sys.argv[1] == 'tfidf_word2vec':
    glove = loadWord2Vec()
    f = open('data.pkl', 'rb')
    [test, train] = pkl.load(f)
    f.close()

    train_set = []
    for key, value in train.items():
        for f_no, words in value.items():
            avg = np.zeros(300)
            #ctr = len(words)
            for i in range(0, len(words)):
                try:
                    avg += glove[train[key][f_no][i]]
                except:
                    continue
            avg = avg/len(words)
            train_set.append([key, avg])

    random.shuffle(train_set)
    X_train = [x[1] for x in train_set]
    Y_train = [x[0] for x in train_set]
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)

    test_set = []
    for key, value in test.items():
        for f_no, words in value.items():
            avg = np.zeros(300)
            #ctr = len(words)
            for i in range(0, len(words)):
                try:
                    avg += glove[test[key][f_no][i]]
                except:
                    continue
            avg = avg/len(words)
            test_set.append([key, avg])

    random.shuffle(test_set)
    X_test = [x[1] for x in test_set]
    Y_test = [x[0] for x in test_set]
    X_test = np.array(X_test)
    Y_test = np.array(Y_test)

if sys.argv[1] == 'rnn':
    max_features = 20000
    maxlen = 80
    batch_size = 32

    (X_train, Y_train), (X_test, Y_test) = imdb.load_data(num_words=max_features)

    X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    X_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    model = Sequential()
    model.add(Embedding(max_features, 128))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])

    print("Train...")
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=15)

    score, accuracy = model.evaluate(X_test, Y_test, batch_size=batch_size)

    print('Test score: {0}'.format(score))
    print('Test accuracy: {0}'.format(accuracy))

if sys.argv[1] != 'rnn':
    print("SVM: {0}".format(SVM(X_train, Y_train, X_test, Y_test)))
    print("Naive Bayes: {0}".format(nultinomialNB(X_train, Y_train, X_test, Y_test)))
    print("Logistic Regression: {0}".format(LR(X_train, Y_train, X_test, Y_test)))
    print("Fully connected Neural Net: {0}".format(NN(X_train, Y_train, X_test, Y_test)))
