import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine, load_breast_cancer, load_iris
from sklearn.preprocessing import StandardScaler

dataset = load_iris()    #dataset = load_wine() or dataset = load_breast_cancer()
X_std = StandardScaler().fit_transform(dataset.data)
y = dataset.target

cov_matrix = np.cov(X_std.T)
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues, eigenvectors = eigenvalues[idx], eigenvectors[:, idx]

explained_variance_ratio = eigenvalues / eigenvalues.sum()
print("Explained variance ratio:", np.round(explained_variance_ratio, 3))

cum_var = np.cumsum(explained_variance_ratio)
print("Components for 95% variance:", np.argmax(cum_var >= 0.95) + 1)

X_reduced = X_std.dot(eigenvectors[:, :2])

colors = ['navy', 'turquoise', 'darkorange']
for color, i, name in zip(colors, [0, 1, 2], dataset.target_names):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1], color=color, label=name)
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA")
plt.legend()
plt.show()
