import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##import seaborn as sns
##%matplotlib inline
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('voice_sample.csv')
#df.head()
#df.info()
#df.describe() ## Gives description about the data

## df.isnull().sum()  ## To check for any null values in the daa

## Shape and the distribution of the data

print('Shape of Data :', df.shape)
print('Total No. of Labels :{}' .format(df.shape[0]))
print('No. of Male: {}' .format(df[df.label == 'male'].shape[0]))
print('No. of Female: {}' .format(df[df.label == 'female'].shape[0]))

X=df.iloc[:,:-1]  ## Take all columns except the last column which is Male/Female label
print(df.shape)
print(X.shape)

y=df.iloc[:,-1]

gender_encoder = LabelEncoder()
y = gender_encoder.fit_transform(y)
print(y)

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)
print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

svc_model=SVC()
svc_model.fit(X_train,y_train)
y_predict=svc_model.predict(X_test)

print('Accuracy Score:')
print(metrics.accuracy_score(y_test,y_predict))

print(confusion_matrix(y_test,y_predict))
print(classification_report(y_test,y_predict))


