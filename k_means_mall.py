from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


data = pd.read_csv('/home/apsit/Downloads/mall_customers.csv')


X = data[['Annual Income (₹000)', 'Spending Score (1–100)']]


wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()


kmeans = KMeans(n_clusters=5, random_state=42)
X['Cluster'] = kmeans.fit_predict(X)


sns.scatterplot(data=X, x='Annual Income (₹000)', y='Spending Score (1–100)',
                hue='Cluster', palette='viridis')
plt.title('Mall Customer Segments')
plt.show()


print(X.groupby('Cluster').mean())
