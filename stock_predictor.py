import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

class Stock_Predictor:
    def __init__(self, ticker):
        self.scaler_prices = MinMaxScaler(feature_range=(0, 1))
        self.scaler_volumes = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.ticker = ticker
        self.model = None

    def set_window_size(self, window_size):
        self.window_size = window_size

    # Download historical stock data
    def download_stock_data(self, start_date, end_date):
        print(f"Downloading stock data for {self.ticker}.")

        data = yf.download(self.ticker, start=start_date, end=end_date)
        #print('acceptible metrics ', data.head())
        prices = data['Close'].values.ravel()
        volumes = data['Volume'].values.ravel()
        print('part of prices:', prices[1:5])
        print('part of volumes:', volumes[1:5])
        return prices, volumes

    def generate_dummy_data(self):
        prices = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200])
        volumes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        return prices, volumes

    def generate_fibonacci(self, n):
        fib_sequence = [0, 1]
        while len(fib_sequence) < n:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        return fib_sequence[:n]

    def normalize_data(self, prices, volumes):
        # convert data to right shape
        X_prices, X_volumes, y = [], [], []
        for i in range(len(prices) - self.window_size):
            X_prices.append(prices[i:i + self.window_size])
            X_volumes.append(volumes[i:i + self.window_size])
            y.append(prices[i + self.window_size])  # Predict next price

        X_prices = np.array(X_prices).reshape((len(X_prices), self.window_size, 1))
        X_volumes = np.array(X_volumes).reshape((len(X_volumes), self.window_size, 1))
        y = np.array(y).reshape(-1, 1)

        return X_prices, X_volumes, y

    # scale data to range 0-1
    def scale(self, X_prices, X_volumes, y):
        # scale data
        X_prices_scaled = self.scaler_prices.fit_transform(X_prices.reshape(-1, 1)).reshape(X_prices.shape)
        X_volumes_scaled = self.scaler_volumes.fit_transform(X_volumes.reshape(-1, 1)).reshape(X_volumes.shape)
        y_scaled = self.scaler_y.fit_transform(y)

        return X_prices_scaled, X_volumes_scaled, y_scaled

    def build_model(self):
        input_prices = Input(shape=(self.window_size, 1))
        x1 = LSTM(50, activation='relu')(input_prices)

        input_volumes = Input(shape=(self.window_size, 1))
        x2 = LSTM(50, activation='relu')(input_volumes)

        merged = Concatenate()([x1, x2])
        dense_1 = Dense(25, activation='relu')(merged)
        dropout = Dropout(0.2)(dense_1)
        output = Dense(1)(dropout)

        self.model = Model(inputs=[input_prices, input_volumes], outputs=output)
        self.model.compile(optimizer='adam', loss='mse')

    def train_model(self, X_prices_scaled, X_volumes_scaled, y_scaled, epochs=500):
        self.model.fit([X_prices_scaled, X_volumes_scaled], y_scaled, epochs=epochs, verbose=1)

    def scale_test_data(self, test_prices, test_volumes):
        points_num = self.window_size
        print('test_prices points:', points_num)
        # scale test data
        test_prices_scaled = self.scaler_prices.transform(test_prices.reshape(-1, 1)).reshape(1, points_num, 1)
        test_volumes_scaled = self.scaler_volumes.transform(test_volumes.reshape(-1, 1)).reshape(1, points_num, 1)
        return test_prices_scaled, test_volumes_scaled

    def predict_next_value(self, test_prices_scaled, test_volumes_scaled):
        # predict
        predicted_scaled_value = self.model.predict([test_prices_scaled, test_volumes_scaled])
        predicted_value = self.scaler_y.inverse_transform(predicted_scaled_value)
        return predicted_value[0][0]