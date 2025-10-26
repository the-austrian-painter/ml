from sklearn.datasets import load_iris
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
import matplotlib.pyplot as plt


iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names) 
y = pd.Series(iris.target) 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


train_acc = rf.score(X_train, y_train)
test_acc = rf.score(X_test, y_test)

print("\nTraining Accuracy:", train_acc)
print("Testing Accuracy:", test_acc)

plt.figure(figsize=(20,10))
tree.plot_tree(rf.estimators_[0], feature_names=iris.feature_names, class_names=iris.target_names, filled=True, rounded=True)
plt.title("Decision Tree from Random Forest")
plt.show()

