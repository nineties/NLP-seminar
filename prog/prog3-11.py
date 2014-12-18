# -*- coding: utf-8 -*-
from numpy import *
from matplotlib.pyplot import *
from sklearn.cluster import AgglomerativeClustering

# 凝集型クラスタリング

random.seed(0)

N = 300

x = random.uniform(-1, 1, N)
y = random.normal(x**2-1, 0.3)

scatter(x, y, s=50, cmap=cm.rainbow)
show()

for linkage in ["ward", "complete", "average"]:
    cls = AgglomerativeClustering(n_clusters = 3, linkage = linkage)
    labels = cls.fit_predict(c_[x, y])

    scatter(x, y, s=50, c=labels, cmap=cm.rainbow)
    title("linkage = %s" % linkage)
    show()
