import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import yfinance as yf
from recurrent_network import StockPredictor

# Allow the user to input the number of epochs for training the model
epochs = int(input("Enter the number of epochs for training the model: "))
predictor = StockPredictor()
# df = predictor.generate_dummy_data()
# Download historical stock data for NVIDIA from January 1, 2020 to January 1, 2025
# Parameters:
# ticker: The stock ticker symbol (e.g., 'NVDA' for NVIDIA)
# start_date: The start date for the historical data in 'YYYY-MM-DD' format
# end_date: The end date for the historical data in 'YYYY-MM-DD' format
df = predictor.download_stock_data('NVDA', '2020-01-01', '2025-01-23')
scaled_data = predictor.normalize_data(df)
# Specify batch_size explicitly
batch_size = 32
# Prepare the data for training the neural network
# time_step defines the number of previous time steps to use for predicting the next time step
X_train, y_train = predictor.prepare_data(scaled_data, time_step=365)

# Build and train the LSTM neural network model
predictor.build_and_train_model(X_train, y_train, epochs, batch_size)
predicted_prices = predictor.predict_future_prices(scaled_data)
predictor.plot_prices(df, predicted_prices)
print("Predicted prices for the next 7 days:", predicted_prices.flatten())
