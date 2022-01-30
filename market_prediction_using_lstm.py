# imports
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

# Get the stock quote
df = web.DataReader('TATAMOTORS.NS', data_source='yahoo', start='2012-01-01', end='2021-04-01')
# show
df
# Get no of rows and Columns
df.shape
# Visualize the data
plt.figure(figsize=(16,8))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Close Price INR (e+05)', fontsize = 18)
plt.show()
# create a new dataframe with closing column
data = df.filter(['Close'])
# convert the dataframe to numpy array
dataset = data.values
# get the no of rows to train the model on
training_data_length = math.ceil(len(dataset) * 0.8)

training_data_length
# scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

scaled_data

# create the training data set
#
train_data = scaled_data[0:training_data_length]
# split data to x_train and Y_train dataset
x_train = []
y_train = []

for i in range(100, len(train_data)):
  x_train.append(train_data[i-100:i,0])
  y_train.append(train_data[i,0])
  if i<=101:
    print(x_train)
    print(y_train)
    print()

# convert the x_train and y_train to numpy array
x_train, y_train = np.array(x_train),np.array(y_train)
print(x_train.shape)
print(y_train.shape)

# reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

# build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(25))
model.add(Dense(1))

# compile the model
model.compile(optimizer='adam',loss='mean_squared_error')
print(x_train.shape)
print(y_train.shape)

# train the model
model.fit(x_train, y_train, batch_size = 1, epochs = 1)

# create the testing dataset
# create a new array containing scaled values from 1543 to 2003
test_data = scaled_data[training_data_length-100:,:]
# create datasets x_test and y_test
x_test = []
y_test = dataset[training_data_length:, :]

for i in range(100, len(test_data)):
  x_test.append(test_data[i-100:i, 0])

x_test = np.array(x_test)

x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# get predicted price value
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# RMSE
rmse = np.sqrt(np.mean(predictions - y_test)**2)
rmse

train = data[:training_data_length]
valid = data[training_data_length:]
valid['Predictions'] = predictions
plt.figure(figsize = (16,8))
plt.title('Model Predictions')
plt.xlabel('Date', fontsize = 18)
plt.ylabel('Price', fontsize = 18)
plt.plot(train['Close'])
plt.plot(valid[['Close','Predictions']])
plt.legend(['Train','Validations','Predictions'], loc = 'lower right')
plt.show()


tata_quote = web.DataReader('TATAMOTORS.NS', data_source='yahoo', start='2012-01-01', end='2021-05-14')
new_df = tata_quote.filter(['Close'])
last_60_days = new_df[-60:].values
last_60_days_scaled = scaler.transform(last_60_days)
X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
pred_price

tata_quote2 = web.DataReader('TATAMOTORS.NS', data_source='yahoo', start='2021-05-14', end='2021-05-14')
print(tata_quote2['Close'])