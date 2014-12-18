# -*- coding: utf-8 -*-
from numpy import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering

df = fetch_20newsgroups()

sel = random.choice(arange(len(df.data)), 2000)

# Tfidf表現に変換し，凝集型クラスタリング
K = 10
vec = TfidfVectorizer(max_df=0.1, min_df=1)
data = vec.fit_transform([df.data[i] for i in sel]).toarray()
print len(vec.get_feature_names())
cls = AgglomerativeClustering(n_clusters=K, linkage="complete") 
labels = cls.fit_predict(data)

# 各クラスの重心において，寄与度の高い単語上位10個を表示
for k in range(K):
    print "class %d:" % k,
    center = data[labels == k].mean(axis=0)
    words = argsort(center)[-10:]
    print " ".join(reversed([vec.get_feature_names()[w] for w in words]))
