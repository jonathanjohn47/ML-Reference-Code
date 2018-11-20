import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

f = pd.read_csv("Position_Salaries.csv")
X = f.iloc[:, 1:2].values
y = f.iloc[:, 2:3].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#Regressor creation
from sklearn.svm import SVR
re = SVR()
re.fit(X, y)

y_pred = sc_y.inverse_transform(re.predict(sc_X.transform(np.array([[6.5]]))))

print(y_pred)

#plot SVR
plt.scatter(X, y, color = 'red')
plt.plot(X, re.predict(X), color='blue')
plt.show()
