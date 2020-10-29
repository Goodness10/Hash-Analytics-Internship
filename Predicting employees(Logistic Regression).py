import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

dataset = pd.read_excel('C:/Users/USER/Hash Analytics Dataset.xlsx')

corr_matrix = dataset.corr()
plt.figure(figsize=(10,10))
sns.heatmap(corr_matrix,annot=True)
plt.savefig('Total Employee Correlation.png')
plt.show()

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

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#Predicting Logistic Regression into training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver='lbfgs')
classifier.fit(X_train, Y_train)

#Predicting test set result
Y_pred = classifier.predict(X_test)

var_prob = classifier.predict_proba(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)


from sklearn import metrics
print('Accuracy:', metrics.accuracy_score(Y_test,Y_pred))
print('Precision:', metrics.precision_score(Y_test,Y_pred))
print('Recall:', metrics.recall_score(Y_test,Y_pred))
print('F1:', metrics.f1_score(Y_test,Y_pred))
