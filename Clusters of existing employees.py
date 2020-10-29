import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

dataset = pd.ExcelFile('C:/Users/USER/Hash-Analytic-Python-Analytics-Problem-case-study-1 (1).xlsx')
existing_employees = dataset.parse('Existing employees')

from sklearn.cluster import KMeans
X = existing_employees.iloc[:, [1,2]].values
kmeans = KMeans(n_clusters = 4, init = 'k-means++', n_init = 10, max_iter = 300, random_state=0)
Y_kmeans = kmeans.fit_predict(X)

plt.scatter(X[Y_kmeans==0,0], X[Y_kmeans==0,1], s=100, c='red', label = 'Cluster1')
plt.scatter(X[Y_kmeans==1,0], X[Y_kmeans==1,1], s=100, c='blue', label = 'Cluster2')
plt.scatter(X[Y_kmeans==2,0], X[Y_kmeans==2,1], s=100, c='green', label = 'Cluster3')
plt.scatter(X[Y_kmeans==3,0], X[Y_kmeans==3,1], s=100, c='cyan', label = 'Cluster4')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=200, c='yellow', label = 'Centroids')

plt.title('Clusters of Satisfaction level vs Last Evaluation')
plt.xlabel('Satisfaction level')
plt.ylabel('Last Evaluation')
plt.savefig('Clusters of existing employees.png')
plt.legend()
plt.show()
