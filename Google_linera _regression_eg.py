import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

quandl.ApiConfig.api_key = "4Ujz8FknYTUyX9tsL34-"

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0


df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out = int(math.ceil(0.1*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
#df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]  # 10% of the data
X = X[:-forecast_out]  # 90% data

#X = X[:-forecast_out+1]  -- bcoz we have already dropped labels above, so this line is not needed

df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)  #This will run as many jobs as possible in parallel
#clf = LinearRegression()           and will significantly improve the speed and accuracy
#clf = svm.SVR(kernel='poly')
#clf.fit(X_train, y_train)
#with open('linearregression.pickle','wb') as f:
#    pickle.dump(clf,f)

#  With the help of pickle, we don't need to train our classifier again and again as
#  it is already trained and thus we can dump the training data in a pickle file

pickle_in = open('linearregression.pickle','rb')
clf = pickle.load(pickle_in) 

accuracy = clf.score(X_test,y_test)
forecast_set = clf.predict(X_lately)
print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]
          #next_date here is the index of the dataframe
          # This line ([np.nan for _ in range(len(df.columns)-1)]) is the list of values which are np.nan
          # And [i] is the forecast_set which are the future values of date

print(df.head())
        
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
    

#print(accuracy)
#print(len(X), len(y))
#print(df.tail())
#print(df.head())

