# Clustering using k-Means
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# read wholesale data
df = pd.read_csv('wholesale_customers_data.csv', sep=',')
data = scale(df)

# TODO reduce dataset to two principal axes (for 2D plotting)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

# TODO run k-Means in a loop with increasing number of clusters to find the optimal one
inertias = []
inertias_no_pca = []
# TODO K should be [2,15]
K = list(range(2,16))
PCAS = list(range(2,5))
for k in K:
    # TODO run k-Means with selected number of clusters
    kmeans = KMeans(n_clusters=k).fit(reduced_data)
    inertias.append(kmeans.inertia_)
    kmeans = KMeans(n_clusters=k).fit(data)
    inertias_no_pca.append(kmeans.inertia_)

plt.plot(K, inertias, label="PCA", marker="x")
plt.plot(K, inertias_no_pca, label="NO PCA", marker="x")
plt.grid()
plt.legend()
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow method for optimal k')
plt.show()
