import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

X_std = StandardScaler().fit_transform(X)

cov_matrix = np.cov(X_std.T)

eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

sorted_idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_idx]
eigenvectors = eigenvectors[:, sorted_idx]


eigenvectors_subset = eigenvectors[:, [0, 1]]

X_reduced = np.dot(X_std, eigenvectors_subset)

plt.figure(figsize=(8, 6))
colors = ['navy', 'turquoise', 'darkorange']
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_reduced[y == i, 0], X_reduced[y == i, 1],
                color=color, lw=2, label=target_name)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Manual PCA of IRIS Dataset (NumPy)")
plt.legend()
plt.show()
