# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 10:49:01 2018

@author: GY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor 
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

clf = RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None,
                            min_samples_split=2, min_samples_leaf=1,
                            max_features='auto', max_leaf_nodes=None,
                            bootstrap=True, oob_score=False, n_jobs=1,
                            random_state=None, verbose=0)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

error = mean_squared_error(y_test, predictions)
print('Random Forest Test MSE: %.3f' % error)

y = y_test * 20450.83322 + 4975.270898
p = predictions * 20450.83322 + 4975.270898

plt.plot(y, label='True Data')
plt.plot(p, label='Random Forest Predict')
plt.title('Random Forest Regression')
plt.legend()
plt.show()

