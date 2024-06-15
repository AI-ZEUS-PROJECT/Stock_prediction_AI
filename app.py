from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from datetime import datetime, timedelta
import pickle

# Initialize the Flask app
app = Flask(__name__)

# Load the trained LSTM model
model = load_model('stock_lstm_model2.h5')

# Load the scaler
with open('scaler2.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Helper function to preprocess input data
def preprocess_data(ticker, start_date, end_date, time_step=100):
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]
    data['Close'] = scaler.transform(data[['Close']])
    data_values = data.values
    
    X = []
    for i in range(time_step, len(data_values)):
        X.append(data_values[i-time_step:i, 0])
    
    X = np.array(X)
    X = X.reshape(X.shape[0], X.shape[1], 1)
    return X, data

# Homepage with input form
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to handle prediction request
@app.route('/predict', methods=['POST'])
def predict():
    # Fetch input data from HTML form
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Preprocess data and make predictions
    X, data = preprocess_data(ticker, start_date, end_date)
    predictions = model.predict(X)
    predictions = scaler.inverse_transform(predictions)
    
    # Prepare response with predictions and dates
    response = {
        'ticker': ticker,
        'predictions': predictions.tolist(),
        'dates': data.index[100:].strftime('%Y-%m-%d').tolist()  # Adjusted for time_step
    }
    return jsonify(response)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
