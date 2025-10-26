import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

data = pd.read_csv("/home/apsit/Downloads/Mall_Customers.csv")

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

X_scaled = StandardScaler().fit_transform(X)

dbscan = DBSCAN(eps=0.3, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)

data['Cluster'] = clusters

print("Cluster label distribution:\n", data['Cluster'].value_counts())

plt.figure(figsize=(8,6))
plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=data['Cluster'], cmap='viridis', s=50)
plt.title('DBSCAN Clustering - Mall Customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.colorbar(label='Cluster Label')
plt.show()

print(data.head())
