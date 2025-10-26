import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, classification_report

data = pd.read_csv('/home/apsit/Downloads/BankDataset.csv')

print(data.head())
print(data.describe())
print(data.info())

data = data.dropna()

data.Loan_Status = data.Loan_Status.map({'Y': 1, 'N': 0})


corr = data.corr(numeric_only=True)


sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()


cor_target = corr['Loan_Status'].abs().sort_values(ascending=False)

print("\nCorrelation with target variable:")
print(cor_target)

best_feature = cor_target.index[1]
print(f"\nSelected feature for prediction: {best_feature}")


x = data[[best_feature]] 
y = data['Loan_Status'] 

if y.dtype == 'object':
    y = y.map({'Y':1, 'N':0})


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print(f"\nR2 Score: {r2}")
print(f"\nAccuracy: {acc}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))
