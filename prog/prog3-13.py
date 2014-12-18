# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
from sklearn.cluster import KMeans, DBSCAN

random.seed(0)

N = 300

theta = random.uniform(0, 2*pi, N)
r = r_[zeros(N/6), ones(2*N/6), 2*ones(3*N/6)]
r = random.normal(r, 0.1)

x = r * cos(theta)
y = r * sin(theta)

scatter(x, y, s=50, cmap=cm.rainbow)
show()

kmeans = KMeans(n_clusters = 3)
labels = kmeans.fit_predict(c_[x, y])
scatter(x, y, s=50, c=labels, cmap=cm.rainbow)
show()

dbscan = DBSCAN(eps=0.4)
labels = dbscan.fit_predict(c_[x, y])
scatter(x, y, s=50, c=labels, cmap=cm.rainbow)
show()
