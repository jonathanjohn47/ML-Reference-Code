from sklearn import preprocessing
f = pd.read_csv("File_Name.csv")                                        #First read the csv file
F = f.as_matrix()                                                       #Converting the csv file into python matrix

le = preprocessing.LabelEncoder()                                       #Creating the object for Label Encoder
le.fit(F[:,i])                                                          #i is the column of strings which we want to encode.
ki  = le.transform(F[:,i])                                              #transforms the i-th column into a coded column and stores in the variable 'ki'
F[:,0] = ki                                                             #Replace the original column with the values in 'ki'
