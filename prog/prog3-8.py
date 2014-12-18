# -*- coding: utf-8 -*-
from numpy import *
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.mixture import GMM

df = fetch_20newsgroups()

# Tfidf表現に変換し，混合分布モデルでクラスタリング
K = 10
vec = TfidfVectorizer(max_df=0.05, min_df=50)
data = vec.fit_transform(df.data).toarray()
print len(vec.get_feature_names())
cls = GMM(n_components=K, covariance_type='spherical')
cls.fit(data)
labels = cls.predict(data)

# 各クラスの重心において，寄与度の高い単語上位10個を表示
for k in range(K):
    print "class %d:" % k,
    center = cls.means_[k]
    words = argsort(center)[-10:]
    print " ".join(reversed([vec.get_feature_names()[w] for w in words]))
    print "variance:", cls.covars_[k][0]
