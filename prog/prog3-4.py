# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

# ‘å‚«‚³‚Ì‘S‚­ˆÙ‚È‚é2•Ï”‚ğg‚Á‚½ê‡

random.seed(0)

N = 300

x = r_[random.normal(70, 10, N/2), random.normal(100, 10, N/2)]
y = r_[random.normal(7, 1, N/2), random.normal(10, 1, N/2)]

scatter(x, y, s=50, cmap=cm.rainbow)
show()

# ³‹K‰»‚ğs‚¤
X = c_[x, y]
X = scale(X, axis=0)

scatter(X[:,0], X[:,1], s=50, cmap=cm.rainbow)
show()

cls = KMeans(n_clusters=2)
labels = cls.fit_predict(X)

scatter(x, y, s=50, c=labels, cmap=cm.rainbow)
show()
