import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

class Data_Loader:
    def __init__(self, ticker, window_size):
        self.scaler_prices = MinMaxScaler(feature_range=(0, 1))
        self.scaler_volumes = MinMaxScaler(feature_range=(0, 1))
        self.scaler_y = MinMaxScaler(feature_range=(0, 1))
        self.ticker = ticker
        self.last_date = None
        self.window_size = window_size

    # Download historical stock data
    def download_stock_data(self, start_date, end_date):
        print(f"Downloading stock data for {self.ticker}.")

        data = yf.download(self.ticker, start=start_date, end=end_date)
        #print('acceptible metrics ', data.head())
        open_prices = data['Open'].values.ravel()
        close_prices = data['Close'].values.ravel()
        volumes = data['Volume'].values.ravel()

        self.first_date = data.index[0].strftime("%Y-%m-%d")
        self.last_date = data.index[-1].strftime("%Y-%m-%d")
        #print('part of prices:', prices[1:5])
        #print('part of volumes:', volumes[1:5])
        # save prices and volumes to csv
        #data.to_csv(f'{self.ticker}_data.csv')

        return open_prices, close_prices, volumes
    
    def read_stock_data_from_csv(self):
        data = pd.read_csv(f'{self.ticker}_data.csv')
        open_prices = data['Open'].values.ravel()
        close_prices = data['Close'].values.ravel()
        volumes = data['Volume'].values.ravel()

        # remove first element (header)
        volumes = volumes[2:].astype(int)
        open_prices = open_prices[2:].astype(float)
        close_prices = close_prices[2:].astype(float)
        #print('part of prices:', prices[0:10])
        #print('part of volumes:', volumes[0:10])
        self.first_date = data.index[0].strftime("%Y-%m-%d")
        self.last_date = data.index[-1].strftime("%Y-%m-%d")

        return open_prices, close_prices, volumes

    def generate_dummy_data(self):
        open_prices = np.array([100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200])
        close_prices = np.array([105, 205, 405, 805, 1605, 3205, 6405, 12805, 25605, 51205])
        volumes = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        return open_prices, close_prices, volumes

    def generate_fibonacci(self, n):
        fib_sequence = [0, 1]
        while len(fib_sequence) < n:
            fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
        return fib_sequence[:n]
    
    def generate_dummy_data2(self):
        test_prices = np.array([6400, 12800, 25600]).reshape(1, 3, 1)
        test_volumes = np.array([1, 1, 1]).reshape(1, 3, 1)
        return test_prices, test_volumes

    def normalize_data(self, open_prices, close_prices, volumes):
        # convert data to right shape
        X_prices_open, X_volumes, y, X_prices_close = [], [], [], []
        for i in range(len(open_prices) - self.window_size):
            X_prices_open.append(open_prices[i:i + self.window_size])
            X_prices_close.append(close_prices[i:i + self.window_size])
            X_volumes.append(volumes[i:i + self.window_size])

            y.append(open_prices[i + self.window_size])  # Predictions of next prices

        X_prices_open = np.array(X_prices_open).reshape((len(X_prices_open), self.window_size, 1))
        X_prices_close = np.array(X_prices_close).reshape((len(X_prices_close), self.window_size, 1))
        X_volumes = np.array(X_volumes).reshape((len(X_volumes), self.window_size, 1))
        #print('X_prices_open:', X_prices_open.shape)

        y = np.array(y).reshape(-1, 1)

        return X_prices_open, X_prices_close, X_volumes, y

    # scale data to range 0-1
    def scale(self, X_prices_open, X_prices_close, X_volumes, y):
        # scale data
        X_prices_open_scaled = self.scaler_prices.fit_transform(X_prices_open.reshape(-1, 1)).reshape(X_prices_open.shape)
        X_prices_close_scaled = self.scaler_prices.fit_transform(X_prices_close.reshape(-1, 1)).reshape(X_prices_close.shape)

        X_volumes_scaled = self.scaler_volumes.fit_transform(X_volumes.reshape(-1, 1)).reshape(X_volumes.shape)
        y_scaled = self.scaler_y.fit_transform(y)

        return X_prices_open_scaled, X_prices_close_scaled, X_volumes_scaled, y_scaled
    
    def merge_data(self, X_prices_open, X_prices_close, X_volumes):
        return np.concatenate((X_prices_open, X_prices_close, X_volumes), axis=2)

    def scale_test_data(self, test_prices_open, test_prices_close, test_volumes):
        points_num = self.window_size
        #print('test_prices points:', points_num)
        # scale test data
        test_prices_open_scaled = self.scaler_prices.transform(test_prices_open.reshape(-1, 1)).reshape(1, points_num, 1)
        test_prices_close_scaled = self.scaler_prices.transform(test_prices_close.reshape(-1, 1)).reshape(1, points_num, 1)
        test_volumes_scaled = self.scaler_volumes.transform(test_volumes.reshape(-1, 1)).reshape(1, points_num, 1)
        return test_prices_open_scaled, test_prices_close_scaled, test_volumes_scaled

    # Returns next trading date
    def get_next_trading_day(self):
        # Get the last available date
        print('last trading date ',self.last_date)
        last_date = pd.to_datetime(self.last_date)
        next_trading_day = last_date + pd.tseries.offsets.BDay()
        return next_trading_day.strftime("%Y-%m-%d")

    def get_trading_range(self):
        return self.first_date, self.last_date
    
    def inverse_data(self, predicted_scaled_value):
        predicted_value = self.scaler_y.inverse_transform(predicted_scaled_value)
        return predicted_value[0][0]
    