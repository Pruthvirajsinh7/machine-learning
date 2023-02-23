import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
t=np.linspace(0,20*np.pi,100)
x=4*np.sin(t/(2*np.pi))
plt.plot(t,x)
plt.title("wave")
plt.show()
mean=0
sd=0.5
noisy=np.random.normal(mean,sd,100)
osc=x+noisy
plt.plot(t,osc)
plt.title('Oscillating')
plt.show()
t=t.reshape(-1,1)
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
degree=int(input('enter degree 1 to 9:'))
polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
polyreg.fit(t,osc)
plt.figure()
plt.scatter(t,osc)
plt.plot(t,polyreg.predict(t),color="black")
plt.title("Polynomial regression with degree"+ str(degree))
plt.show()
r_sq=polyreg.score(t,osc)
print('coefficient of determination', r_sq)