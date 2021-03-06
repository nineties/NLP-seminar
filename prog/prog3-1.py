# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
from sklearn.cluster import KMeans

# ユークリッド距離に基づくクラスタリング

random.seed(0)

N = 300

x = random.uniform(-1, 1, N)
y = random.uniform(-1, 1, N)

xlim(-1, 1)
ylim(-1, 1)
scatter(x, y, s=50, cmap=cm.rainbow)
show()

cls = KMeans(n_clusters = 10)
labels = cls.fit_predict(c_[x, y])

xlim(-1, 1)
ylim(-1, 1)
scatter(x, y, s=50, c=labels, cmap=cm.rainbow)
show()
