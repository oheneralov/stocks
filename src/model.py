import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Concatenate
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

class Stock_Predictor:
    def __init__(self):
        self.model = None

    def set_window_size(self, window_size):
        self.window_size = window_size

    def build_model(self):
        self.model = Sequential([
            LSTM(50, activation='relu', return_sequences=True, input_shape=(self.window_size, 3)),
            Dropout(0.2),
            LSTM(50, activation='relu', return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])

        self.model.compile(optimizer='adam', loss='mse',  metrics=['mae'])

    def train_model(self, params_scaled, y_scaled, epochs=500):
        print('Training model...')
        self.model.fit(params_scaled, y_scaled, epochs=epochs, verbose=1)

    def predict_next_value(self, params_scaled):
        # predict
        return self.model.predict(params_scaled)
        #print('predicted_scaled_value:', predicted_scaled_value.shape)
       

    