# -*- coding: utf-8 -*-
from numpy import *
from sklearn.datasets import fetch_20newsgroups
from gensim.parsing import preprocess_string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from time import time

random.seed(0)

df = fetch_20newsgroups()
N = len(df.data) # 訓練文書数

# 前処理 (Porterのステマーなど)
corpus = map(lambda s: " ".join(preprocess_string(s)), df.data)

# 全体の9割を訓練に使い，残りの1割のクラスを正しく決定出来るか検証
indices = arange(N)
random.shuffle(indices)
train_indices = indices[:9*N/10]
test_indices  = indices[9*N/10:]

# コーパスを読み込み，特徴ベクトル変換器を構築
vec = CountVectorizer()
vec.fit([corpus[i] for i in train_indices])

# 全文書を特徴ベクトルに変換
X = vec.transform(corpus)

# 単純ベイズ(ベルヌーイ)
start = time()
clf = BernoulliNB()
clf.fit(X[train_indices], df.target[train_indices])
learning_time = time() - start

# 推定精度
start = time()
predicted = clf.predict(X[test_indices])
predict_time = time() - start
print "BernoulliNB: accuracy=%.3f, learning time=%.3fsec, predict time=%.3fsec" % (average(predicted == df.target[test_indices]), learning_time, predict_time)

# 単純ベイズ(多項)
start = time()
clf = MultinomialNB()
clf.fit(X[train_indices], df.target[train_indices])
learning_time = time() - start

# 推定精度
start = time()
predicted = clf.predict(X[test_indices])
predict_time = time() - start
print "MultinomialNB: accuracy=%.3f, learning time=%.3fsec, predict time=%.3fsec" % (average(predicted == df.target[test_indices]), learning_time, predict_time)
