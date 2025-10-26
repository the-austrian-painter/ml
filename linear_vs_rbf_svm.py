import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


cancer = datasets.load_breast_cancer()
X, y = cancer.data, cancer.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


models = {
    'Linear SVM': SVC(kernel='linear', C=1.0, random_state=42),
    'RBF SVM': SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
}


metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1-score': [], 'Cross-Val': []}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics['Accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['Precision'].append(precision_score(y_test, y_pred))
    metrics['Recall'].append(recall_score(y_test, y_pred))
    metrics['F1-score'].append(f1_score(y_test, y_pred))
    cv = cross_val_score(model, X, y, cv=5).mean()
    metrics['Cross-Val'].append(cv)


labels = list(metrics.keys())
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x - width/2, [metrics[m][0] for m in labels], width, label='Linear SVM')
ax.bar(x + width/2, [metrics[m][1] for m in labels], width, label='RBF SVM')

ax.set_ylabel('Score')
ax.set_title('Linear vs RBF SVM Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.ylim(0.85, 1.0)
plt.tight_layout()
plt.show()
