# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
from sklearn.cluster import KMeans

# 集団サイズが大きく異る場合

random.seed(0)

N = 300

x = r_[random.normal(5, 0.1, N/10), random.normal(6, 0.4, 9*N/10)]
y = r_[random.normal(5, 0.1, N/10), random.normal(6, 0.4, 9*N/10)]

scatter(x, y, s=50, cmap=cm.rainbow)
show()

cls = KMeans(n_clusters=2)
labels = cls.fit_predict(c_[x, y])

scatter(x, y, s=50, c=labels, cmap=cm.rainbow)
show()
