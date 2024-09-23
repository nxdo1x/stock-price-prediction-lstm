import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

# Load model and scaler
model = load_model('keras_model.h5')
scaler = MinMaxScaler(feature_range=(0, 1))

# Streamlit Title
st.title("Stock Price Prediction")

# User Input for Stock Ticker
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL)", "AAPL")

# Date Range Input
start_date = st.date_input("Start Date", value=pd.to_datetime("2008-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2017-12-31"))

# Fetch Data
if st.button("Get Data"):
    df = yf.download(ticker, start=start_date, end=end_date)
    st.write(df.tail())
    
    # Prepare Data
    df = df[['Close']]
    df = df.dropna()
    
    # Scale Data
    data_array = scaler.fit_transform(df.values)

    # Create Train/Test Split
    train_size = int(len(data_array) * 0.8)
    train_data = data_array[:train_size]
    test_data = data_array[train_size:]

    # Prepare X and y for training
    x_train, y_train = [], []
    for i in range(100, len(train_data)):
        x_train.append(train_data[i-100:i])
        y_train.append(train_data[i, 0])
    
    x_train, y_train = np.array(x_train), np.array(y_train).reshape(-1, 1)

    # Prepare X for testing
    x_test = []
    for i in range(100, len(test_data)):
        x_test.append(test_data[i-100:i])
    
    x_test = np.array(x_test)

    # Make Predictions
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Actual Prices
    actual_prices = df['Close'].values[train_size + 100:]

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(actual_prices, color='blue', label='Actual Prices')
    plt.plot(predictions, color='red', label='Predicted Prices')
    plt.title(f'Stock Price Prediction for {ticker}')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.legend()
    
    # Display Plot
    st.pyplot(plt)

