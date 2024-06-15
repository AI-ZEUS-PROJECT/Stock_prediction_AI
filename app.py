import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

app = Flask(__name__)

# Load the trained LSTM model
with open('stock_lstm_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the scaler (assuming it was trained with the model)
scaler = MinMaxScaler()

def preprocess_data(ticker, start_date, end_date, time_step=100):
    # Download data
    data = yf.download(ticker, start=start_date, end=end_date)
    data = data[['Close']]

    # Check if enough data is available
    if len(data) < time_step:
        raise ValueError("Not enough data to create a valid dataset")

    # Scale the data
    scaled_data = scaler.fit_transform(data)

    # Create dataset
    def create_dataset(data, time_step):
        X = []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
        return np.array(X)

    X = create_dataset(scaled_data, time_step)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    return X, data

@app.route('/predict', methods=['POST'])
def predict():
    # Parse request JSON
    request_data = request.get_json()
    ticker = request_data['ticker']
    start_date = request_data['start_date']
    end_date = request_data['end_date']
    time_step = request_data.get('time_step', 100)

    try:
        X, original_data = preprocess_data(ticker, start_date, end_date)

        # Make prediction
        y_pred = model.predict(X)
        y_pred = scaler.inverse_transform(y_pred.reshape(-1, 1))

        # Combine predictions with original dates
        prediction_dates = original_data.index[time_step:]
        prediction_df = pd.DataFrame(data={'Date': prediction_dates, 'Predicted_Close': y_pred.flatten()})

        # Convert to JSON
        response = prediction_df.to_json(orient='records', date_format='iso')
        return jsonify(response)

    except ValueError as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
