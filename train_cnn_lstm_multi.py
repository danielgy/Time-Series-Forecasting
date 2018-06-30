# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:47:14 2018

@author: GY
"""
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import yaml
import tensorboard
import CNN_LSTM_6


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
    values = df.values.astype('float32')

    X_train, y_train, X_test, y_test = CNN_LSTM_6.data_pre(df)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # fit model
    global_start_time = time.time()
    print('> Data Loaded. Compiling...')

    model = CNN_LSTM_6.build_model()
    model = CNN_LSTM_6.fit_model(X_train, y_train, model, batch_size=128, nb_epoch=1000, validation_split=0.2)
    print('Training duration (s) : ', time.time() - global_start_time)

    # Predict
    predicted, score = CNN_LSTM_6.predict_point_by_point(X_test, y_test)
    print('Test score is: ', score)
    
    fig = plt.figure(facecolor='white')
    for i in range(6):
        ax = fig.add_subplot(2, 3, (i+1))
        ax.plot(np.absolute(y_test[:, i]-predicted[:, i]),label='MAE')
        ax.legend()
    plt.show()
    
    fig = plt.figure(facecolor='white')
    for i in range(6):
        ax = fig.add_subplot(2, 3, (i+1))
        ax.plot(y[:, i], label='True Data')
        ax.plot(pre[:, i], label='Predict')
        ax.legend()
    plt.show()
