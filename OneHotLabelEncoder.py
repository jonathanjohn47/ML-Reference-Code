from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
for i in range(1,19,1):
	X[:,i] = le.fit_transform(X[:,i])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
