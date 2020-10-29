import numpy as np
import pandas as pd
import seaborn as sns

dataset = pd.read_excel('C:/Users/USER/Hash Analytics Dataset.xlsx')

X = dataset.iloc[:, 1:10].values
Y = dataset.iloc[:,-1].values

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(transformers=[('one_hot_encoder', OneHotEncoder(categories='auto'),
                                      [7,8])], remainder='passthrough')

X = np.array(ct.fit_transform(X))

#Avoid Dummy Variable Trap
X = X[:, 1:]

#split dataset into train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.30, random_state = 0)


#Fitting Random Forest into training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 20, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, Y_train)

#Predicting test set result
Y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


from sklearn import metrics
print('Accuracy:', metrics.accuracy_score(Y_test,Y_pred))
print('Precision:', metrics.precision_score(Y_test,Y_pred))
print('Recall:', metrics.recall_score(Y_test,Y_pred))
print('F1:', metrics.f1_score(Y_test,Y_pred))
