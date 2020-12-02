# Clustering using k-Means
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

# read wholesale data
df = pd.read_csv('wholesale_customers_data.csv', sep=',')
data = scale(df)

# reduce dataset to two principal axes (for 2D plotting)
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(data)

# run k-Means with increasing number of clusters to find the optimal one
inertias = []
K = range(1, 15)
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans = kmeans.fit(reduced_data)
    inertias.append(kmeans.inertia_)

plt.plot(K, inertias, 'bx-')
plt.xlabel('k')
plt.ylabel('Inertia')
plt.title('Elbow method for optimal k')
plt.show()
