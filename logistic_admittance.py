import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, classification_report

data = pd.read_csv('/home/apsit/Downloads/2.01.+Admittance.csv')

print(data.head())
print(data.describe())
print(data.info())


data.Admitted = data.Admitted.map({'Yes':1, 'No':0})
print(data['Admitted'].value_counts())


x = data[['SAT']]
y = data['Admitted']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f"\nR2 Score: {r2}")
print(f"\nAccuracy: {acc}")
print("\nClassification Report:", classification_report(y_test, y_pred))
