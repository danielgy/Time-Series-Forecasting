# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 23:47:44 2018

@author: GY
"""

import os
import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout, Reshape, Flatten, Permute
from keras.layers.convolutional import Conv2D,Conv1D
from keras.layers.recurrent import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.models import Sequential, model_from_yaml
from keras.callbacks import History
from keras.utils import plot_model
import matplotlib.pyplot as plt
import yaml


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Hide messy TensorFlow warnings
warnings.filterwarnings("ignore")  # Hide messy Numpy warnings


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
    y_train = np.concatenate((train[:, 3:7],train[:,16:18]),axis=1)
    x_test = data[int(row):, :]
    y_test = np.concatenate((data[int(row):, 3:7],data[int(row):, 16:18]),axis=1)
    x_train = np.reshape(x_train, (x_train.shape[0], 1, 1, 1, x_train.shape[1]))
    x_test = np.reshape(x_test, (x_test.shape[0], 1, 1, 1, x_test.shape[1]))
    return [x_train, y_train, x_test, y_test]


def build_model():
    model = Sequential()
    model.add(TimeDistributed(Conv2D(nb_filter=512, nb_row=1, nb_col=3, init='glorot_uniform',
                     activation='relu', weights=None, border_mode='valid',
                     subsample=(1, 1), dim_ordering='th', W_regularizer=None,
                     b_regularizer=None, activity_regularizer=None, W_constraint=None,
                     b_constraint=None, bias=True), input_shape=(None, 1, 1, 19)))

    model.add(TimeDistributed(AveragePooling2D(pool_size=(1, 2), strides=None, border_mode='valid', dim_ordering='th')))
    model.add(TimeDistributed(Conv2D(nb_filter=256, nb_row=1, nb_col=3, init='glorot_uniform',
                     activation='relu', weights=None, border_mode='valid',
                     subsample=(1, 1), dim_ordering='th', W_regularizer=None,
                     b_regularizer=None, activity_regularizer=None, W_constraint=None,
                     b_constraint=None, bias=True)))
    model.add(TimeDistributed(AveragePooling2D(pool_size=(1, 2), strides=None, border_mode='valid', dim_ordering='th')))
    model.add(TimeDistributed(Conv2D(nb_filter=128, nb_row=1, nb_col=3, init='glorot_uniform',
                     activation='relu', weights=None, border_mode='valid',
                     subsample=(1, 1), dim_ordering='th', W_regularizer=None,
                     b_regularizer=None, activity_regularizer=None, W_constraint=None,
                     b_constraint=None, bias=True)))
    model.add(TimeDistributed(Flatten()))
    print(model.output_shape)
    model.add(TimeDistributed(Dense(1024)))
    model.add(TimeDistributed(Activation('relu')))
    print(model.output_shape)
    model.add(LSTM(1024, return_sequences=True, activation='relu'))
    model.add(LSTM(1024, return_sequences=True, activation='relu'))
    print(model.output_shape)
    model.add(TimeDistributed(Dropout(0.25)))
    model.add(TimeDistributed(Activation('relu')))
    model.add(TimeDistributed(Dense(1024)))
    model.add(TimeDistributed(Activation('linear')))
    model.add(GlobalAveragePooling1D(name="global_avg"))
    model.add(Dense(6))
    model.add(Activation('linear'))
    print(model.output_shape)
    model.summary()
    plot_model(model, to_file='model.png')
    start = time.time()
    model.compile(loss="mse", optimizer="adam")
    print("> Compilation Time : ", time.time() - start)
    return model


def fit_model(X_train, y_train, model, batch_size=128, nb_epoch=10, validation_split=0.2):
    history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, validation_split=validation_split)
    plt.plot()
    plt.plot(history.history['loss'], label='Trainig Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training')
    plt.xlabel('epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()
    yaml_string = model.to_yaml()
    with open('cnn_lstm/cnn_lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('cnn_lstm/cnn_lstm.h5')
    return model

def predict_point_by_point(data, label):
    # Predict each timestep given the last sequence of true data, in effect only predicting 1 step ahead each time
    print('loading model....')
    with open('cnn_lstm/cnn_lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)
    print('loading weights...')
    model.load_weights('cnn_lstm/cnn_lstm.h5')
    model.compile(loss='mean_squared_error', optimizer='adam')
    predicted = model.predict(data)
    score = model.evaluate(data, label, batch_size=128)
    return predicted, score


def predict_sequence_full(model, data, window_size):
    # Shift the window by 1 new prediction each time, re-run predictions on new window
    curr_frame = data[0]
    predicted = []
    for i in range(len(data)):
        predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
        curr_frame = curr_frame[1:]
        curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
    return predicted


def predict_sequences_multiple(model, data, window_size, prediction_len):
    # Predict sequence of 50 steps before shifting prediction run forward by 50 steps
    prediction_seqs = []
    for i in range(int(len(data) / prediction_len)):
        curr_frame = data[i * prediction_len]
        predicted = []
        for j in range(prediction_len):

            predicted.append(model.predict(curr_frame[newaxis, :, :])[0, 0])
            curr_frame = curr_frame[1:]
            curr_frame = np.insert(curr_frame, [window_size - 1], predicted[-1], axis=0)
        prediction_seqs.append(predicted)
    return prediction_seqs