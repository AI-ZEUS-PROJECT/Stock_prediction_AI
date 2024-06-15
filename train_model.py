import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import tensorflow as tf
import pickle

# List of tickers
tickers = ["GOOGL", "MSFT", "AAPL", "IBM"]

# Download the data for the specified period for all tickers
data = yf.download(tickers, start="2010-01-01", end="2023-01-01")

# Stack the data for all tickers into a single DataFrame
data = data.stack(level=1).reset_index(level=1)
data = data.rename_axis('Date').reset_index()

# One-hot encode the 'Ticker' column
encoder = OneHotEncoder(sparse_output=False)
ticker_encoded = encoder.fit_transform(data[['Ticker']])

# Create a DataFrame for the one-hot encoded tickers
ticker_encoded_df = pd.DataFrame(ticker_encoded, columns=encoder.get_feature_names_out(['Ticker']))

# Concatenate the one-hot encoded tickers with the original data
data = pd.concat([data, ticker_encoded_df], axis=1)

# Drop the original 'Ticker' column as it's now encoded
data.drop(columns=['Ticker'], inplace=True)

data.dropna(inplace=True)
data.reset_index(inplace=True)

# Extract features and target
scaler = MinMaxScaler()
data['Close'] = scaler.fit_transform(data[['Close']])

# Create the dataset for LSTM
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 100
data_values = data[['Close']].values
X, y = create_dataset(data_values, time_step)

X = X.reshape(X.shape[0], X.shape[1], 1)

train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
y_train, y_test = y[0:train_size], y[train_size:len(y)]

# Build the model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model with Adam optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, batch_size=1, epochs=1)

# Save the model
model.save('stock_lstm_model2.h5')

# Save the scaler
with open('scaler2.pkl', 'wb') as f:
    pickle.dump(scaler, f)
