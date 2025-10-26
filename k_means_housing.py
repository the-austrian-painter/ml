import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

home_data = pd.read_csv("/home/apsit/Downloads/housing.csv", usecols = ['longitude', 'latitude', 'median_house_value'])


sns.scatterplot(data = home_data, x = 'longitude', y = 'latitude', hue = 'median_house_value')
plt.show()

xtrain_set, xtest_set,ytrain_set,ytest_set  = train_test_split(home_data[['latitude','longitude']],home_data['median_house_value'], test_size = 0.3, random_state = 0)

xtrain_norm = normalize(xtrain_set)


kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(xtrain_norm)


sns.boxplot(x=kmeans.labels_, y = ytrain_set)
plt.show()

silhouette_score(xtrain_norm, kmeans.labels_)

K = range(2,8)
fits=[]
score=[]

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0, n_init='auto')
    kmeans.fit(xtrain_norm)
    fits.append(kmeans)
    score.append(silhouette_score(xtrain_norm, kmeans.labels_))

sns.scatterplot(data=xtrain_set, x='longitude', y='latitude', hue=fits[0].labels_)
plt.show()

sns.scatterplot(data=xtrain_set, x='longitude', y='latitude', hue=fits[3].labels_)
plt.show()

sns.boxplot(x=fits[3].labels_, y = ytrain_set)
plt.show()
