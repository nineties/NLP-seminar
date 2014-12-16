# -*- coding: utf-8 -*-
from numpy import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

df = fetch_20newsgroups()

# Tfidf表現に変換し，K-meansでクラスタリング
K = 10
vec = TfidfVectorizer()
data = vec.fit_transform(df.data)
cls = KMeans(n_clusters=K)
labels = cls.fit_predict(data)

# 各クラスの重心において，寄与度の高い単語上位10個を表示
for k in range(K):
    print "class %d:" % k,
    center = cls.cluster_centers_[k]
    words = argsort(center)[-10:]
    print " ".join(reversed([vec.get_feature_names()[w] for w in words]))
