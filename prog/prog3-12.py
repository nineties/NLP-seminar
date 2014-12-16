# -*- coding: utf-8 -*-
from numpy import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

df = fetch_20newsgroups()

# Tfidf表現に変換し，K-meansでクラスタリング
K = 10
vec = TfidfVectorizer(max_df=0.01, min_df=50)
data = vec.fit_transform(df.data).toarray()
cls = AgglomerativeClustering(n_clusters=K, linkage="ward") 
labels = cls.fit_predict(data)

# 各クラスの重心において，寄与度の高い単語上位10個を表示
for k in range(K):
    print "class %d:" % k,
    center = data[labels == k].mean(axis=1)
    words = argsort(center)[-10:]
    print " ".join(reversed([vec.get_feature_names()[w] for w in words]))
