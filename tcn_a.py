# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 09:22:30 2018

@author: GY
"""

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.utils import plot_model
from tcn import tcn
#from utils import data_generator

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
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return [x_train, y_train, x_test, y_test]

print('> Loading data... ')
df = pd.read_csv('./data/data(1).csv', header=None)
X_train, y_train, X_test, y_test = data_pre(df)
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


def run_task():
    model, param_str = tcn.dilated_tcn(output_slice_index='last',
                                       num_feat=X_train.shape[2],
                                       num_classes=0,
                                       nb_filters=24,
                                       kernel_size=3,
                                       dilatations=[1, 2, 4, 8],
                                       nb_stacks=8,
                                       max_len=X_train.shape[1],
                                       activation='norm_relu',
                                       use_skip_connections=False,
                                       return_param_str=True,
                                       regression=True)

    print('x_train.shape = {}'.format(X_train.shape))
    print('y_train.shape = {}'.format(y_train.shape))
    plot_model(model, to_file='tcn.png')
    model.summary()

    history = model.fit(X_train, y_train, epochs=500, batch_size=128, validation_split=0.2)
    
    plt.plot()
    plt.plot(history.history['loss'], label='Trainig Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    
    predicted = model.predict(X_test)
    score = model.evaluate(X_test, y_test, batch_size=128)
    print('Test score is: {}!\n'.format(score))
    
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(np.absolute(y_test-predicted), label='MAE')
    ax.legend()
    plt.show()

    y = y_test * 20450.83322 + 4975.270898
    pre = predicted * 20450.83322 + 4975.270898

    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y, label='True Data')
    ax.plot(pre, label='Predict')
    ax.legend()
    plt.show()
    return model, param_str

if __name__ == '__main__':
    model, param_str = run_task()
