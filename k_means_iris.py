from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


iris = load_iris()
X = iris.data
y = iris.target

kmeans_iris = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans_iris.fit(X)


cm = confusion_matrix(y, kmeans_iris.labels_)
accuracy = np.max(cm.sum(axis=0)) / len(y)

print("Confusion Matrix:\n", cm)
print(f"Approximate Clustering Accuracy: {accuracy:.2f}")

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X[:, 2], y=X[:, 3], hue=kmeans_iris.labels_, palette='viridis', legend='full')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('K-Means Clustering on Iris Dataset (Petal Length vs. Petal Width)')
plt.show()
