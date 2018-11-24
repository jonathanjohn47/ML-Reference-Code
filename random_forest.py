import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("Position_Salaries.csv")
X = data.iloc[:, 1:2].values
y = data.iloc[:,-1].values

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in range(0, len(X[0]), 1):
	X[:,i] = le.fit_transform(X[:,i])

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 100, random_state = 0)
reg.fit(X,y)

y_pred = reg.predict(X)
print(y_pred)

plt.scatter(y, y_pred)
plt.show()

X_grid = np.arange(min(X), max(X), 0.001)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X_grid, reg.predict(X_grid), color = 'red')
plt.show()


