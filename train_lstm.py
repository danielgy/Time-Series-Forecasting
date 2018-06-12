#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/4/28 14:26
# @Author  : Zoe
# @Site    : 
# @File    : train_lstm.py
# @Software: PyCharm Community Edition

import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml
import lstm


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
    seq_len = 5

    print('> Loading data... ')
    df = pd.read_csv('./data/data(1).csv', header=None)
    values = df.values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled = scaler.fit_transform(values)
    scaled = pd.DataFrame(scaled)

    X_train, y_train, X_test, y_test = lstm.data_pre(scaled, seq_len)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # fit model
    global_start_time = time.time()
    print('> Data Loaded. Compiling...')
    # layers [X_input feature dim,  LSTM[1].unit,    LSTM[2].unit,   output_dim]
    model = lstm.build_model(layers=[X_train.shape[-1], 20, 20, X_train.shape[-1]], sequence_length=seq_len)
    model = lstm.fit_model(X_train, y_train, model, batch_size=64, nb_epoch=1000, validation_split=0.2)
    print('Training duration (s) : ', time.time() - global_start_time)

    # Predict
    predicted = lstm.predict_point_by_point(X_test)
    predicted = scaler.inverse_transform(predicted)
    
    y_test = scaler.inverse_transform(y_test)
    
    fig = plt.figure(facecolor='white')
    for i in range(6):
        ax = fig.add_subplot(2, 3, (i+1))
        ax.plot(y_test[:, i], label='True Data')
        ax.plot(predicted[:, i], label='Predict')
        ax.legend()
    plt.show()

    fig = plt.figure(facecolor='white')
    for i in range(4):
        ax = fig.add_subplot(2, 2, (i+1))
        ax.plot(y_test[:, -(i+1)], label='True Data')
        ax.plot(predicted[:, -(i+1)], label='Predict')
        ax.legend()
    plt.show()

