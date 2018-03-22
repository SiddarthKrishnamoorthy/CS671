#!/usr/bin/env python3
import sys
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier

def SVM(X_train, Y_train, X_test, Y_test):
    model = LinearSVC()
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)

def LR(X_train, Y_train, X_test, Y_test):
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)

def NB(X_train, Y_train, X_test, Y_test):
    model = GaussianNB()
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)

def NN(X_train, Y_train, X_test, Y_test):
    model = MLPClassifier()
    model.fit(X_train, Y_train)
    return model.score(X_test, Y_test)

# TODO: Write model for RNN
