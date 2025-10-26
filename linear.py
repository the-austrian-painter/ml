
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

IceCream = pd.read_csv('/home/apsit/Downloads/IceCream.csv')

print(IceCream.head())

print(IceCream.describe())

print(IceCream.info())

x = IceCream[['Temperature']]
y = IceCream['Revenue']

sns.jointplot(x='Temperature', y='Revenue', data=IceCream)
plt.title('Temperature vs Revenue')
plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

model = LinearRegression()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
r2 = r2_score(y_test, y_predict)

print(f"R2 Score: {r2}")

plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, model.predict(x_train))
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()
