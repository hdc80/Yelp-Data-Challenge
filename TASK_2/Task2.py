# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 17:33:27 2017

@author: pbhadani
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from numpy import newaxis
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

# Importing the training set
training_set = pd.read_csv('sorted_part2_update.csv')

training_set = training_set.iloc[:,1:2].values
training_set = training_set.astype('float32')

print(training_set.head())
# let us visualise the trend of the prices

#sns.distplot(training_set['Review_No'])

plt.plot(training_set)
plt.show()

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()
training_set = sc.fit_transform(training_set)

# Getting the inputs and the ouputs
#X_train = training_set[0:748]
#y_train = training_set[1:749]

# Reshaping
#X_train = np.reshape(X_train, (106, 7, 1))

train_size = int(len(training_set) * 0.8)
test_size = len(training_set) - train_size
train, test = training_set[0:train_size,:], training_set[train_size:len(training_set),:]

#test1 = training_set[740: , :]

look_back = 4
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

#test1X, test1Y = create_dataset(test1, look_back)

#print(test1X.shape[0],test1X.shape[1])

trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))
#test1X = np.reshape(test1X, (test1X.shape[0], test1X.shape[1], 1))

# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the input layer and the LSTM layer
regressor.add(LSTM(units = 10, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(trainX, trainY, batch_size = 32, epochs = 2000)

print(trainY.shape)

# Part 3 - Making the predictions and visualising the results
# make predictions
trainPredict = regressor.predict(trainX)
testPredict = regressor.predict(testX)

# plot the prediction value
plt.plot(train)
plt.plot(trainPredict*3)
plt.plot(testPredict)
plt.show()
# invert predictions
trainPredict = sc.inverse_transform(trainPredict)
trainY = sc.inverse_transform([trainY])
testPredict = sc.inverse_transform(testPredict)
testY = sc.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.5f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.5f RMSE' % (testScore))