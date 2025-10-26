import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('/home/apsit/Downloads/50_Startups.csv')

data = pd.get_dummies(data, drop_first=True)

print(data.head())
print(data.describe())
print(data.info())


x = data.drop('Profit', axis=1)
y = data['Profit']


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)

print(f"R2 Score: {r2}")

results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print("\nComparison of Actual and Predicted Profit:")
print(results.head())


sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title("Actual vs Predicted Profit")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)
plt.show()


sns.regplot(x='R&D Spend', y='Profit', data=data)
plt.title('R&D Spend vs Profit')
plt.show()
