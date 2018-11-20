from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le = LabelEncoder()
X[:,i] = le.fit_transform(X[:,i])                 # 'i' is the index of the array X that we want to encode.
ohe = OneHotEncoder(categorical_features = [i])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]                                       # avoid Dummy Variable trap
