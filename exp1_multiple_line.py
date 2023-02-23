import numpy as n
from sklearn.linear_model import LinearRegression

x= [[0,1] , [5,1] , [15,2] , [25,5] , [35,11] , [45,15] , [55,34] , [60,35]]
y = [4,5,20,14,32,22,38,43]

x,y = n.array(x) , n.array(y)

print(x)
print(y)

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)
print('coefficient of determination' , r_sq)
print('intercept:' , model.intercept_)
print('slope:' , model.coef_)

y_pred = model.predict(x)
print('predicted response:', y_pred, sep='\n')

y_pred2= model.intercept_ + n.sum(model.coef_ * x , axis=1)
print('predicterd response:' , y_pred2 , sep='\n')

x_new = n.arange(10).reshape(-1,2)
print(x_new)
y_new = model.predict(x_new)
print(y_new)
