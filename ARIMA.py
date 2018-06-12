# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 19:05:10 2018

@author: GY
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from pandas.tools.plotting import autocorrelation_plot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

def data_pre(data):
    """
    :param data: 
    :param seq_len: 
    :return: 
    """
    # sequence_length = sequence_length + 1
    row = round(0.9 * data.shape[0])
    data = data.values
    train = data[:int(row), :]
    x_train = train[:, :]
    y_train = train[:,16:17]
    x_test = data[int(row):, :]
    y_test = data[int(row):, 16:17]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))
    return [x_train, y_train, x_test, y_test]

print('> Loading data... ')
df = pd.read_csv('./data/data(1).csv', header=None)
X_train, y_train, X_test, y_test = data_pre(df)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#一阶差分
# y = pd.DataFrame(y_train)
# fig = plt.figure(figsize=(12,8))
# ax1= fig.add_subplot(111)
# diff1 = y.diff(1)
# diff1.plot(ax=ax1)

# fig = plt.figure(figsize=(12,8))
# ax2= fig.add_subplot(111)
# diff2 = y.diff(2)
# diff2.plot(ax=ax2)

#q,p
#y= y.diff(1)
# fig = plt.figure(figsize=(12,8))
# ax1=fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(y,lags=40,ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(y,lags=40,ax=ax2)


history = [x for x in y_train]
predictions = list()
for t in range(len(y_test)):
    model = ARIMA(history, order=(1, 0, 0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = y_test[t]
    history.append(obs)
    print('t=%d, predicted=%f, expected=%f' % (t, yhat, obs))
error = mean_squared_error(y_test, predictions)
print('Test MSE: %.3f' % error)

predictions=np.array(predictions)
y = y_test * 20450.83322 + 4975.270898
pre = predictions * 20450.83322 + 4975.270898

plt.plot(y, label='True Data')
plt.plot(pre, label='Predict')
plt.legend()
plt.show()
