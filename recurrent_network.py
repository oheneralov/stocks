import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf


"""
A class used to predict stock prices using a recurrent neural network (RNN) with LSTM layers.
Attributes
----------
scaler : MinMaxScaler
    An instance of MinMaxScaler to normalize the data.
model : Sequential
    The neural network model used for prediction.
Methods
-------
generate_dummy_data():
    Generates dummy stock price data for testing.
normalize_data(df):
    Normalizes the stock price data using MinMaxScaler.
prepare_data(scaled_data):
    Prepares the data for training the neural network.
build_and_train_model(X_train, y_train):
    Builds and trains the LSTM neural network model.
predict_future_prices(scaled_data, days=7):
    Predicts future stock prices for a given number of days.
plot_prices(df, predicted_prices):
    Plots the actual and predicted stock prices.
"""
class StockPredictor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None

    # Download historical stock data
    def download_stock_data(self, ticker, start_date, end_date):
        data = yf.download(ticker, start=start_date, end=end_date)
        df = pd.DataFrame(data['Close']).reset_index(drop=True)
        df.columns = ['prices']
        return df

    def generate_dummy_data(self):
        data = {
            'prices': [100, 102, 105, 107, 110, 108, 109]
        }
        df = pd.DataFrame(data)
        return df

    def normalize_data(self, df):
        scaled_data = self.scaler.fit_transform(df)
        return scaled_data

    def prepare_data(self, scaled_data, time_step=60):
        X_train = []
        y_train = []
        for i in range(time_step, len(scaled_data)):
            X_train.append(scaled_data[i-time_step:i, 0])
            y_train.append(scaled_data[i, 0])

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        return X_train, y_train

    def build_and_train_model(self, X_train, y_train, epochs=1, batch_size=32):
        print("Training the model... epochs: {}, batch_size: {}".format(epochs, batch_size))
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(60, 1)))
        self.model.add(LSTM(units=50))
        self.model.add(Dense(units=1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

    def predict_future_prices(self, scaled_data, days=7):
        predicted_prices = []
        input_data = scaled_data[-60:]
        for _ in range(days):
            input_data = np.reshape(input_data, (1, 60, 1))
            predicted_price = self.model.predict(input_data)
            predicted_prices.append(predicted_price[0, 0])
            input_data = np.append(input_data[0, 1:], predicted_price[0, 0]).reshape(1, 60, 1)

        predicted_prices = self.scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))
        return predicted_prices

    def plot_prices(self, df, predicted_prices):
        plt.figure(figsize=(10, 6))
        plt.plot(df['prices'], color='blue', label='Actual Prices')
        plt.plot(range(len(df), len(df) + 7), predicted_prices, color='red', label='Predicted Prices')
        plt.title('Stock Prices Prediction')
        plt.xlabel('Days')
        plt.ylabel('Prices')
        plt.legend()
        plt.show()

