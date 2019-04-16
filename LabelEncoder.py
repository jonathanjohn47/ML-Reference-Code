from sklearn.preprocessing import LabelEncoder, OneHotEncoder
for column in dataset.columns:
    if dataset[column].dtype == type(object):
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
