import pandas as pd
import quandl
import math
import time
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plot
from matplotlib import style

import pickle

style.use('ggplot')

df = quandl.get('WIKI/CRUS', authtoken='1_3L8XRiXjpFMf1SUo5V')
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-.99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf = LinearRegression()
# clf.fit(X_train, y_train)
# with open('/tmp/linear_reg.pickle', 'wb') as f:
#     pickle.dump(clf, f)

pickle_in = open('/tmp/linear_reg.pickle',"rb")
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
# print(accuracy)

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

df['forecast'] = np.nan

last_date = df.iloc[-1].name
print(type(last_date))
last_unix = time.mktime(last_date.timetuple())  #last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = next_unix
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]

df['Adj. Close'].plot()
df['forecast'].plot()
plot.legend(loc=4)
plot.xlabel('Date')
plot.ylabel('Price')
plot.show()
