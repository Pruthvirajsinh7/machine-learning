import numpy as np
import pandas as pd

dataset=pd.read_csv("C:/Users/Pruthvirajsinh/Downloads/prices-split-adjusted.csv")
y=dataset.iloc[:30,3].values
dataset.drop('close', inplace=True, axis=1)
X=dataset.iloc[:30,2:].values
print(X)
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state =0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))