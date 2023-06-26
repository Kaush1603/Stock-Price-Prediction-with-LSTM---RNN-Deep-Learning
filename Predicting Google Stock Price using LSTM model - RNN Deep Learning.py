# -*- coding: utf-8 -*-
"""
Created on Sun Jun 25 11:55:44 2023

@author: tarun
"""

##Building RNN

## Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Importing the training set. Note: we will only be using training set and no test set for RNN.
## Test set will only be used for prediction and comparsion with actual resuult purpose.

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

##now change the dataframe to numpy array because model take inputs only in numpy arrays

training_set = dataset_train.iloc[:,1:2].values ##values will turn it to arrays


##apply feature scaling - apply normalization instead of standardization. It is recommened to do so in RNN if there is sigmoid fuction in output layer

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1)) ##instance created of scaler standardization 
training_set_scaled = sc.fit_transform(training_set) ##now every data point is scaled to 0 to 1

##now we will create special data structure (time steps: important) which RNN will remember while predicting stock prices
##creating a data structure with 60 timesteps and 1 output, that mean network will look at 60 financial days before
##first T time steps and so on. So there are 20 financial days in a month so that means RNN will look at data for 
##3 months and then try to predict T+1

X_train = []
y_train = []
##now using a for loop, we will populate 60 stock prices in X_train and next stock price in y_train

for i in range(60,1258):
    X_train.append(training_set_scaled[i-60:i, 0]) ##when i=60 from for loop, it will append X_train all the values from 0 index to 59th index since i is excluded in this line of code
    y_train.append(training_set_scaled[i, 0])
#now X_train and y_train are list and network take only arrays as input so we will convert them in arrays
X_train, y_train = np.array(X_train), np.array(y_train)

##so when you check the data X_train, first row will have first 60 stock prices and 2nd row will have 60 data point from 1st to 61st. So it is a sliding window of 1 stock price in each row.
##and y_train will have 61st in 1st row which is last element of X_train in 2nd row
##now, we wll add some other indicators such as open price, volume etc or maybe other stock prices that may have some correlation with google stock price. This will make the network more sophisticated

## Rehape the data, adding dimentionality
## we will add open price as indicator, we can add more if we want. This will be done in X_train as this is the input
## To understand, how reshape work where we can add another dimention to X_trian  - 3d tensor(we are making it 3d)  with shape
## (batch_size, timesteps, input_dim), (optional) 2D tensors with shape (batch_size, output_dim) . batch size is stock prices from 2012 and 2016 and timestep is 60 as choosen and third point is new dimentions
## Here batch size is number of rows and timesteps are number of columns, input_dim will be 1 since we are only choosing one but if we want we can chose 2 or 3 as well
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

## Lets build RNN network now
# import libraries
## it is going to be a stacked LSTM with dropout regularization to prevent overfitting
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


## layers will be sequential layers as opposed to computational graph
## initializing the RNN, this time we are prediction continous value
regressor = Sequential()

## adding the first LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50, return_sequences= True, input_shape = (X_train.shape[1], 1)))     ##3 inputs units = which is number of LSTM cells in this layer, return_seq = True if we are going to add another layer. Once done, we will set it to false but it is its default value. Third arugument is input shape ie 3d. But we will only indicate 2 which is timesteps and indicator because 1st one which is observation, it will be automatically taken
regressor.add(Dropout(0.2))  ##20% rate at which you want to drop neurons for regulaization. It will help preventing overfitting


## adding the second LSTM layer and some Dropout regularization. Input layer wont be added since it was added as input in first layer. Now network is at level 2
regressor.add(LSTM(units=50, return_sequences = True)) 
regressor.add(Dropout(0.2))

## adding the thrid LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50, return_sequences= True))
regressor.add(Dropout(0.2))

## adding the fourth LSTM layer and some Dropout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

#now adding output layer. Because we were adding LSTM layer till now, we kept return sequence to True, but now we are adding classic output layer so we will now use dense layer like we did it in ANN and CNN to make a fully connected output layer connected to last lstm layer
regressor.add(Dense(units=1)) ## here Units will be 1 since dim for output is also one, meaning stock price prediction


##Two steps are remaining - The first one is compiling the RNN  with a optimizer and right loss and then fit the network on training set

#compiling the RNN. Optimizer that we are going to use for is Adam and RMSprop, that is recommeded for RNN and same has been recommened on documentation on Keras
regressor.compile(optimizer= 'adam', loss = 'mean_squared_error')

## Fiting the RNN on training set which is X_train and y_train
regressor.fit(X_train, y_train, epochs= 100, batch_size = 32)

## making the prediction and visializing the result

##getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values

##getting the predicted stock price of 2017
## we are going to predict 1 day in january based on previous 60 financial days. So some days will come from dec 2016 and some days will come from jan 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0 )
inputs = dataset_total[len(dataset_total) - len(dataset_test) -60 :].values # len here will get us the index of 3rd jan 2017 index number because we need 60 previous financial days before 3rd jan 2017 as lower bound. so -60 will take us the index which is 60 previous financial day, and upper bound will be last day of jan so that next day can be predicted
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)            ##we will have to scale the inputs, remember not to scale the test value as we need them as they are. After that we will make a 3d structure of the input as made during training
                          ## also we will use transform method rather fit transform because fit was already applied before and we want the same scale to be applied on input as it was applied in training


## now we will reshape it to 3d structure
                          
X_test = []
for i in range(60,80):  ##this will get us 60 previous value for each of the value from jan which is starting from 3rd jan
    X_test.append(inputs[i-60:i, 0]) 
#now X_train and y_train are list and network take only arrays as input so we will convert them in arrays
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

## Visualization the result

plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()


### this model predicts one day ahead as trained during the training model so now it has predicted jan stock price using each previous day of jan