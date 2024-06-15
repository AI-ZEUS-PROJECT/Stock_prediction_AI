# Stock Market Prediction with LSTM
[Home](p2.PNG)
[Result](p1.PNG)

## Overview

This project focuses on predicting stock market prices using a Long Short-Term Memory (LSTM) neural network. The application includes a Flask API (`app.py`) for serving predictions and a Jupyter notebook (`stock_market_prediction_LSTM.ipynb`) for data preprocessing, model training, and evaluation.

## File Descriptions

### 1. app.py

This script defines a Flask web application that serves predictions based on an LSTM model.

- **Libraries Used**: `tensorflow`, `keras`, `joblib`, `numpy`, `pandas`, `flask`, `sklearn`, `yfinance`
- **Endpoints**:
  - `/predict`: Accepts POST requests with JSON payload to predict stock prices.

### 2. stock_market_prediction_LSTM.ipynb

A Jupyter notebook that outlines the steps for downloading stock data, preprocessing, model training, and evaluation.

- **Libraries Used**: `yfinance`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`, `tensorflow`, `pickle`
- **Data Source**: Yahoo Finance
- **Tickers**: ["GOOGL", "MSFT", "AAPL", "IBM"]
- **Period**: 2010-01-01 to 2023-01-01

## How to Run

### Prerequisites

- Python 3.7+
- Required packages (install via pip): 
  ```bash
  pip install tensorflow keras joblib numpy pandas flask scikit-learn yfinance matplotlib seaborn
  ```

### 1. Run the Flask App

- Ensure `stock_lstm_model.h5` and `scaler.pkl` are in the same directory as `app.py`.
- Execute the Flask application:
  ```bash
  python app.py
  ```

### 2. Predicting Stock Prices

- Send a POST request to the `/predict` endpoint with JSON payload:
  ```json
  {
    "ticker": "AAPL",
    "start_date": "2022-01-01",
    "end_date": "2023-01-01",
    "time_step": 100
  }
  ```

### Example Request (using `curl`)

```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{
  "ticker": "AAPL",
  "start_date": "2022-01-01",
  "end_date": "2023-01-01",
  "time_step": 100
}'
```

## Model Training and Evaluation

### 1. Data Download and Preprocessing

- Download stock data for specified tickers and period.
- Perform one-hot encoding on ticker symbols.
- Scale the `Close` prices using `MinMaxScaler`.

### 2. Model Creation

- Define an LSTM model with:
  - Two LSTM layers
  - Two Dense layers
  - Dropout layers for regularization
- Compile with `adam` optimizer and `mean_squared_error` loss function.
- Train the model on the preprocessed data.

### 3. Save and Load Model

- Save the trained model using `pickle`.
- Use `joblib` to save the scaler object.

### 4. Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- R-squared (R²) and Adjusted R-squared

## Visualization

- Plot the actual vs. predicted stock prices.
- Generate time series plots using Matplotlib.

