#!/usr/bin/env python3
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re

raw_data = open('data_full')
raw_data = raw_data.read()

vocab = set(raw_data)
# Data
X = []
y = []

r1 = re.compile("(.{1,5})(\.|\?|!|\])'?</s>\s*<s>'?(.{1,5})")
match1 = re.findall(r1, raw_data)

r2 = re.compile("(.{1,5})(\.|\?|!|\])([^<]{1,5})")
match2 = re.findall(r2, raw_data)
#print(match2)

for tup in match1:
    idx = 0
    s = ""
    s = tup[0]+tup[2]
    s = list(s)
    x = np.zeros(len(vocab))
    for v in vocab:
        if v in s:
            x[idx] = 1
        idx += 1
    X.append(x)
    y.append(0)

for tup in match2:
    idx = 0
    s = ""
    s = tup[0]+tup[2]
    s = list(s)
    x = np.zeros(len(vocab))
    for v in vocab:
        if v in s:
            x[idx] = 1
        idx += 1
    X.append(x)
    y.append(1)

y = np.asarray(y)
X = np.asarray(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 69)

model = LogisticRegression()
model.fit(X_train, y_train)

print("Accuracy: {0}".format(model.score(X_test, y_test)))
