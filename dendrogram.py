import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f = pd.read_csv("Mall_Customers.csv")

X = f.iloc[:,[3,4]].values

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))

plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distances')
plt.show()
