from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
for i in range(0,k,1):				#'k' is the last row in the X array created using iloc function
	X[:,i] = le.fit_transform(X[:,i])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
