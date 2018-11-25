import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Social_Network_Ads.csv")
X = data.iloc[:,2:3].values
y = data.iloc[:,4].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
sc_X= StandardScaler()
X = sc_X.fit_transform(X)
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

from sklearn.svc import SVC
svm = SVC(kernel = "linear", random_state = 0)
svm.fit(X_train, y_train)
svm.predict(X)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y, svm.predict(X))

print(cm)
