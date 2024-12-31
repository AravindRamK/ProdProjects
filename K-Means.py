import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


file_path = 'C:/Users/karav_867vu4n/OneDrive/Desktop/Mall_Customers.csv'
data = pd.read_csv(file_path)


features = data[['Annual Income (k$)', 'Spending Score (1-100)']]


inertia = []
range_clusters = range(1, 11)

for k in range_clusters:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(range_clusters, inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.xticks(range_clusters)
plt.grid()
plt.show()


k = 5  
kmeans = KMeans(n_clusters=k, random_state=42)
data['Cluster'] = kmeans.fit_predict(features)

silhouette_avg = silhouette_score(features, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg:.2f}')
print(f'Inertia (Within-Cluster Sum of Squares): {kmeans.inertia_:.2f}')


plt.figure(figsize=(10, 6))
for i in range(k):
    cluster_points = data[data['Cluster'] == i]
    plt.scatter(cluster_points['Annual Income (k$)'],
                cluster_points['Spending Score (1-100)'], label=f'Cluster {i+1}')


centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], s=200, c='black', marker='X', label='Centroids')

plt.title('K-means Clustering Results')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()

