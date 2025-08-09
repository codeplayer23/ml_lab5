#importing the necessary packages 
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score , calinski_harabasz_score , davies_bouldin_score

#loading the dataset 
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

#splitting the dataset into training and test set 
X = df.iloc[:,0:2]
Y = df.iloc[:,196:197]
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2)

#applying KMeans
kmeans = KMeans(n_clusters=2 ,random_state=0 , n_init="auto").fit(X_train)
print(kmeans.labels_) 
print(kmeans.cluster_centers_)

#silhouette score 
ss = silhouette_score(X_train,kmeans.labels_)
print("Silhouette Score :",ss)

#CH score 
ch = calinski_harabasz_score(X_train,kmeans.labels_)
print("CH score :",ch)

#DB index
db = davies_bouldin_score(X_train,kmeans.labels_)
print("DB index :",db)