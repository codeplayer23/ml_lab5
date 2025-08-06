#importing the necessary packages 
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error , root_mean_squared_error , mean_absolute_percentage_error , r2_score

#loading the dataset 
df = pd.read_csv("/Users/niteshnirranjan/Downloads/DCT_mal.csv")

#splitting the dataset into training and test set 
X = df.iloc[:,0:2]
Y = df.iloc[:,196:197]

X_train , X_test , Y_train , Y_test = train_test_split(X,Y,test_size=0.2)

#linear regression
reg = LinearRegression().fit(X_train,Y_train)
Y_train_pred = reg.predict(X_train)
Y_test_pred = reg.predict(X_test)

#Mean Squared Error 
mse = mean_squared_error(Y_test,Y_test_pred)
print("Mean Squared Error :",mse)

#Root Mean Squared Error 
rmse = root_mean_squared_error(Y_test,Y_test_pred)
print("Root Mean Squared Error :",rmse)

#Mean Absolute Percentage Error 
mape = mean_absolute_percentage_error(Y_test,Y_test_pred)
print("Mean Absolute Percentage Error :",mape)

#R2 Score 
r2score = root_mean_squared_error(Y_test,Y_test_pred)
print("R2 score:",r2score)
