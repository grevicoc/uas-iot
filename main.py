import pandas as pd

import numpy as np

import sklearn

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix,accuracy_score,f1_score,roc_curve

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from keras.utils import pad_sequences

from keras.models import Sequential

from keras.callbacks import ReduceLROnPlateau, EarlyStopping

from keras.layers import Activation, Dense, Dropout, Embedding, LSTM

import re

from IPython.display import display

import os

import string

import time

import random

import matplotlib.pyplot as plt

import pickle

random.seed(10)

df = pd.read_csv('./data_cleaned_v1.csv')

df.drop('Is_Raining', axis=1, inplace=True)

df['Date'] = pd.to_datetime(df['Date'])
df['Time_Scaled'] = df['Date'].dt.hour + df['Date'].dt.minute / 60.0
df['Time_Scaled'] = df['Time_Scaled'] / 23.5 
df.drop('Date', axis=1, inplace=True)

mapping = {'NO': -1, 'YES': 1}

df['Will_Rain'] = df['Will_Rain'].map(mapping)

# print(df)

scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df[['Time_Scaled', 'Temp', 'Hum', 'Light']])
df[['Time_Scaled', 'Temp', 'Hum', 'Light']] = pd.DataFrame(scaled_data, columns=['Time_Scaled', 'Temp', 'Hum', 'Light'])

scaler_params = scaler.get_params()
print(scaler_params)
# Save the scaling parameters to a file
with open('scaler_params.pkl', 'wb') as f:
    pickle.dump(scaler_params, f)

# print(df)

training_data, testing_data = train_test_split(df, test_size=0.2, random_state=25)

print(f"No. of training examples: {training_data.shape[0]}")
print(f"No. of testing examples: {testing_data.shape[0]}")

x_train = training_data[['Time_Scaled', 'Temp', 'Hum', 'Light']]
y_train = training_data['Will_Rain']

x_test = testing_data[['Time_Scaled', 'Temp', 'Hum', 'Light']]
y_test = testing_data['Will_Rain']

# print(x_train.head())
# print(x_test.head())

# print(y_train.head())
# print(y_test.head())

# Define the LSTM model 1
model = Sequential()
model.add(LSTM(32, input_shape=(4,1)))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Reshape the input data
x_train = x_train.values.reshape((-1, 4, 1))
x_test = x_test.values.reshape((-1, 4, 1))

# Train the model
history = model.fit(x_train, y_train, epochs=50, validation_data=(x_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test loss: {loss}, Test accuracy: {accuracy}")


# # Define the LSTM model 2
# model = Sequential()

# model.add(LSTM( 128 , dropout = 0.25, recurrent_dropout = 0.25))

# model.add(Dense(1, activation = 'sigmoid' ))

# model.compile( optimizer = "adam" , loss = 'binary_crossentropy' , metrics = ['accuracy'] )

# early_stopper = EarlyStopping( monitor = 'val_acc' , min_delta = 0.0005, patience = 3 )

# reduce_lr = ReduceLROnPlateau( monitor = 'val_loss' , patience = 2 , cooldown = 0)

# callbacks = [ reduce_lr , early_stopper]

# train_history = model.fit( x_train , y_train , epochs = 5,validation_split = 0.1 , verbose = 1 , callbacks = callbacks)

# score = model.evaluate( x_test , y_test )

# print( "Accuracy: {:0.4}".format( score[1] ))

# print( "Loss:", score[0] )

#Export model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)