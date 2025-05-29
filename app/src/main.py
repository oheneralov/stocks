from stock_predictor import Stock_Predictor
import numpy as np
import os

# hyperparameters
epochs = 100
window_size = 7 # set to trading period (3, 5, 7, 10, 20, 30, 60, 90, 120, 180, 365)
ticker = 'NVDA'

predictor = Stock_Predictor(ticker)
#prices, volumes = predictor.generate_dummy_data()
# end="2025-01-29" in yfinance is exclusive, meaning data won't include January 29.
open_prices, close_prices, volumes = predictor.download_stock_data('2025-01-01', '2025-05-22')
#prices, volumes = predictor.read_stock_data_from_csv()
print('number of days:', len(open_prices))
print(f'epochs: {epochs}, ticker: {ticker}, window_size: {window_size}')

if (len(open_prices) < window_size or len(close_prices) < window_size or len(volumes) < window_size):
    print('Not enough data! Decrease window_size or increase data range in download_stock_data()')
    exit(1)
predictor.set_window_size(window_size)

# normalize so that X_prices are array of matrices of shape (1, window_size)
X_prices_open, X_prices_close, X_volumes, y = predictor.normalize_data(open_prices, close_prices, volumes)
X_prices_open_scaled, X_prices_close_scaled, X_volumes_scaled, y_scaled = predictor.scale(X_prices_open, X_prices_close, X_volumes, y)
predictor.build_model()

params_scaled = predictor.merge_data(X_prices_open_scaled, X_prices_close_scaled, X_volumes_scaled)
predictor.train_model(params_scaled, y_scaled, epochs=epochs)

# predict next price
test_prices_open = np.array(open_prices[-window_size:]).reshape(1, window_size, 1)
test_prices_close = np.array(close_prices[-window_size:]).reshape(1, window_size, 1)
test_volumes = np.array(volumes[-window_size:]).reshape(1, window_size, 1)
test_prices_open_scaled, test_prices_close_scaled, test_volumes_scaled = predictor.scale_test_data(test_prices_open, test_prices_close, test_volumes)

params_scaled = predictor.merge_data(test_prices_open_scaled, test_prices_close_scaled, test_volumes_scaled)

predicted_price = predictor.predict_next_value(params_scaled)

first_date, last_date = predictor.get_trading_range()
print(f'neural network is trained on the range from {first_date} to {last_date}')
print("\n\n\n----------------------------------------------------")
# last 5 days of data
#print(f'open_prices: {open_prices[-5:]}')

print(f'Predicted open price: {predicted_price:.2f}$ for {ticker} on {predictor.get_next_trading_day()} Open')

