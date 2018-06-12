# -*- coding: utf-8 -*-
"""
Created on Thu May 31 18:57:58 2018

@author: GY
"""


import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml

import CNN_LSTM


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    print('Start ploting')
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    # Pad the list of predictions to shift it in the graph to it's correct start
    for i, data in enumerate(predicted_data):
        if i < 10:
            continue
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.show()


# Main Run Thread
if __name__ == '__main__':

    epochs = 5

    print('> Loading data... ')
    df = pd.read_csv('./data/data(1).csv', header=None)
    column = list(range(df.shape[1]))
    column.remove(16)
    column.append(16)
    df = df.ix[:, column]
    values = df.values.astype('float32')

    X_train, y_train, X_test, y_test = CNN_LSTM.data_pre(df)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # fit model
    global_start_time = time.time()
    print('> Data Loaded. Compiling...')

    model = CNN_LSTM.build_model()
    model = CNN_LSTM.fit_model(X_train, y_train, model, batch_size=16, nb_epoch=1000, validation_split=0.2)
    print('Training duration (s) : ', time.time() - global_start_time)

    # Predict
    predicted, score = CNN_LSTM.predict_point_by_point(X_test, y_test)
    print('Test score is: ', score)

    y = y_test*20446.87563+4975.270898
    pre = predicted*20446.87563+4975.270898
    
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(y, label='True Data')
    ax.plot(pre, label='Predict')
    ax.legend()
    plt.show()

