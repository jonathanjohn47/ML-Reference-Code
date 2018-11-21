from sklearn import preprocessing
f = pd.read_csv("FileName.csv")
X = f.iloc[:, :-1].values
y = f.iloc[:, -1].values

le = preprocessing.LabelEncoder()
X[:,i] = le.fit_transform(X[:,i])                                         #Replace the original column with the values in 'ki'
