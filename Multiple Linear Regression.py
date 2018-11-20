import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
f = pd.read_csv("File_Name.csv")

X = f.iloc[:,:-1].values
y = f.iloc[:,-1].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:,j] = le.fit_transform(X[:,j])                # 'j' is the index of the column we need to encode 
ohe = OneHotEncoder(categorical_features = [j])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print(y_pred)
