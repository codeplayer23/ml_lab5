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
k_values = range(2, 11) 
ss_scores = []
ch_scores = []
db_scores = []

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)
    
    ss = silhouette_score(X,labels)
    ch = calinski_harabasz_score(X, labels)
    db = davies_bouldin_score(X, labels)
    
    ss_scores.append(ss)
    ch_scores.append(ch)
    db_scores.append(db)


#plotting SS score  
plt.figure(figsize=(10,5))

plt.subplot(1, 3, 1)
plt.plot(k_values, ss_scores, marker='o',color='blue')
plt.title("Silhouette Score vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("SS Score")
plt.grid(True)

#plotting CH score
plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o' , color='green')
plt.title("Calinski-Harabasz Score vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("CH Score")
plt.grid(True)

#plotting DB index
plt.subplot(1, 3, 3)
plt.plot(k_values, db_scores, marker='o', color='red')
plt.title("Davies-Bouldin Index vs k")
plt.xlabel("Number of clusters (k)")
plt.ylabel("DB Index (Lower is Better)")
plt.grid(True)

plt.tight_layout()
plt.show()



