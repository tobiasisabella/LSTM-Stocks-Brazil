#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  5 10:46:58 2020

@author: isabella
"""

import numpy as np
import os
import sys
import time
import pandas as pd
import pickle
import logging
import matplotlib.pyplot as plt
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import statistics

df = pd.read_csv("PETR4.csv")
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)
df = df["Close"]
df.head()
len(df)


plt.figure(figsize = (15,10))
plt.plot(df, label="Company A")
plt.legend(loc="best")
plt.show()

array = df.values.reshape(df.shape[0],1)
array[:5]
scl = MinMaxScaler()
normalizado = scl.fit_transform(array)
normalizado[:5]

sc = StandardScaler()
training_set_scaled = sc.fit_transform(normalizado) 
sc_predict = StandardScaler() 
sc_predict.fit_transform(normalizado[:,0:5])

X_train = [] 
y_train = [] 
n_future = 1 
n_past = 5 
for i in range(n_past, len(normalizado)):
    X_train.append(normalizado[i - n_past:i, 0:5])
    y_train.append(normalizado[i+n_future-1:i + n_future, 0])
X_train, y_train = np.array(X_train), np.array(y_train) 
X_train = np.reshape (X_train, (X_train.shape [0], X_train.shape [1], 1)) 

regressor = Sequential()
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)


dataset_test = pd.read_csv('WDO1.csv')
y_true = np.array(dataset_test['Close'])
y_true = y_true[0:10]
predictions = regressor.predict(X_train[-20:])
predictions_to_compare = predictions[[3,4,5,6,9,10,11,12,13,17]]
y_pred = sc_predict.inverse_transform(predictions_to_compare)


plt.figure(figsize = (15,10))
hfm, = plt.plot(y_pred, 'r', label='predicted_stock_price')
plt.title('Predictions and Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price Future')
plt.savefig('graph.png', bbox_inches='tight')
plt.show()
plt.close() 
