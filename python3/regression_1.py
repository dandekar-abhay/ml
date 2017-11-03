import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plot
from matplotlib import style

style.use('ggplot')

# df = Quandl.get_table('WIKI/PRICES', date='1999-11-18', ticker='A')
df = quandl.get('NSE/SBIN')

#print df.head()

df = df[['Open','High','Low','Close']]

df['HL_PCT'] = (df['High'] - df['Close']) / df['Close'] * 100

df['DAILY_PCT_CHANGE'] = (df['Close'] - df['Open']) / df['Open'] * 100

df = df [['Close', 'High', 'Low', 'DAILY_PCT_CHANGE', 'HL_PCT']]

#print df.head()
# print df

forecast_col = 'Close'

#print "Before ====="
#print df.head()

df.fillna('-9999', inplace=True)

#print "After ====="
#print df.head()

forecast_out = int(math.ceil(0.01*len(df)))
#print forecast_out

df['label'] = df[forecast_col].shift(-forecast_out)

#df.dropna(inplace=True)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:forecast_out]


df.dropna(inplace=True)

y = np.array(df['label'])
#X = preprocessing.scale(X)
#X = X[:-forecast_out+1]
y = np.array(df['label'])

print (len(X))
print (len(y))

print type(X)," ", type(y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=10)
#clf = svm.SVR(kernel='poly')

clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print (accuracy)

#print type(df)
#print df
