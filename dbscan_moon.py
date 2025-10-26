from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

X, y = make_moons(n_samples=300, noise=0.05, random_state=42)

dbscan = DBSCAN(eps=0.2, min_samples=5) 
clusters = dbscan.fit_predict(X)

df = pd.DataFrame(X, columns=['x', 'y'])
df['Cluster'] = clusters

plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='x', y='y', hue='Cluster', palette='viridis', legend='full')
plt.title('DBSCAN Clustering on Make Moons Dataset')
plt.show()

print("Cluster labels found:", set(clusters))
print(f"Number of clusters (excluding noise): {len(set(clusters)) - (1 if -1 in clusters else 0)}")
print(f"Number of noise points: {sum(clusters == -1)}")
