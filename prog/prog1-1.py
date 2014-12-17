# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
import re
from sklearn.datasets import fetch_20newsgroups
from gensim.models import Word2Vec

# 簡単のため前処理として数字・特殊記号の除去, 小文字への統一のみ
corpus = map(lambda s: re.sub('[0-9!"#$%&\'()*+,\-./:;<=>?@\\\[\]^_`{|}~]',' ',s.lower()).split(), fetch_20newsgroups().data)

# 前後5単語を窓として使い，二次元空間に単語をマッピングする
model = Word2Vec(corpus, size=2, window=5)

words = model.vocab.keys()
for w in words:
    v = model[w]
    text(v[0], v[1], w)
show()
