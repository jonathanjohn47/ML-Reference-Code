from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt

data = load_boston()
X = data.data
y = data.target
print(data.feature_names)
print(X.shape)
print(X[:,1:10])
print(y)

#------------RFR-------------#

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor()
rf.fit(X_train,y_train)
predictions=rf.predict(X)
print (mean_squared_error(y,predictions))
print (mean_absolute_error(y,predictions))
plt.scatter(y,predictions)
plt.show()
#-----------Lasso------------#

from sklearn.linear_model import Lasso
rf=Lasso()
rf.fit(X,y)
p=rf.predict(X)
print (mean_squared_error(y,p))
print(mean_squared_error(y,p))
plt.scatter(y,p)
plt.show()

#------------SVR-------------#
from sklearn.svm import SVR
rf=SVR()
rf.fit(X,y)
predictions=rf.predict(X)
print (mean_squared_error(y,predictions))
plt.scatter(y,predictions)
plt.show()

#------------KNN-------------#
from sklearn.neighbors import KNeighborsRegressor
rf=KNeighborsRegressor()
rf.fit(X,y)
predictions=rf.predict(X)
print (mean_squared_error(y,predictions))
plt.scatter(y,predictions)
plt.show()

#------------Ridge-------------#
from sklearn.linear_model import Ridge
rf=Ridge(0.5)
rf.fit(X,y)
predictions=rf.predict(X)
print (mean_squared_error(y,predictions))
plt.scatter(y,predictions)
plt.show()

#--------------------------------------------------------------------#
