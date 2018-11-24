import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#---------Get/Read the data------------#
data = pd.read_csv("Position_Salaries.csv")
X = data.iloc[:,1:2].values
y = data.iloc[:,-1].values

#---------Encode the string parts---------#
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for i in range(0, len(X[0]), 1):
	X[:,i] = le.fit_transform(X[:,i])

#---------Split the data into training/testing parts---------#
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#---------Apply Decision Tree Regressor and train the machine------------#
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train,y_train)

#---------Predict using what the machine has been trained-------------#
y_pred = regressor.predict(X)

#---------Apply high resolution graph------------------#
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.plot(X,y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.show()
