import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

f = pd.read_csv("Position_Salaries.csv")

X = f.iloc[:,1:2].values
y = f.iloc[:, 2].values


#Linear Regression
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(X,y)
y_pred = linreg.predict(X)

#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

#Plot
plt.scatter(X,y,color = "red")
plt.plot(X, lin_reg.predict(poly_reg.fit_transform(X)), color = "blue")
plt.show()

#Predict
print(lin_reg.predict(poly_reg.fit_transform(6.5)))
