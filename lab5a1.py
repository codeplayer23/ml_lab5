#importing the necessary packages 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

#loading the dataset 
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

#splitting the dataset into training and test set 
X = df.iloc[:,0:2]
Y = df.iloc[:,196:197]

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2)

#linear regression
reg = LinearRegression().fit(X_train,Y_train)
Y_train_pred = reg.predict(X_train)
