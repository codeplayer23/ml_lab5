#importing the necessary packages 
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score , calinski_harabasz_score , davies_bouldin_score
import matplotlib.pyplot as plt

#loading the dataset 
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

#splitting the dataset into training and test set 
X = df.iloc[:,0:2]
Y = df.iloc[:,196:197]
X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2)

#applying KMeans 
k_values = range(2, 20) 
distortions = []

for k in k_values:
    kmeans = KMeans(n_clusters=k , random_state=42)
    kmeans.fit(X_train)
    distortions.append(kmeans.inertia_)

#plotting elbow plot 
plt.figure(figsize=(8, 5))
plt.plot(k_values, distortions, marker='o', color='blue')
plt.title("Elbow Method for Optimal k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Distortion / Inertia")
plt.grid(True)
plt.show()