from stock_predictor import Stock_Predictor
import numpy as np
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# hyperparameters
epochs = 1000
window_size = 7 # set to trading period (3, 5, 7, 10, 20, 30, 60, 90, 120, 180, 365)
ticker = 'NVDA'

predictor = Stock_Predictor(ticker)
#prices, volumes = predictor.generate_dummy_data()
# end="2025-01-29" in yfinance is exclusive, meaning data won't include January 29.
prices, volumes = predictor.download_stock_data('2023-01-01', '2025-01-31')
#prices, volumes = predictor.read_stock_data_from_csv()
print('number of days:', len(prices))
print(f'epochs: {epochs}, ticker: {ticker}, window_size: {window_size}')

if (len(prices) < window_size):
    print('Not enough data! Decrease window_size or increase data range in download_stock_data()')
    exit(1)
predictor.set_window_size(window_size)

# normalize so that X_prices are array of matrices of shape (1, window_size)
X_prices, X_volumes, y = predictor.normalize_data(prices, volumes)
X_prices_scaled, X_volumes_scaled, y_scaled = predictor.scale(X_prices, X_volumes, y)
predictor.build_model()

params_scaled = predictor.merge_data(X_prices_scaled, X_volumes_scaled)
predictor.train_model(params_scaled, y_scaled, epochs=epochs)

# predict next price
test_prices = np.array(prices[-window_size:]).reshape(1, window_size, 1)
test_volumes = np.array(volumes[-window_size:]).reshape(1, window_size, 1)
test_prices_scaled, test_volumes_scaled = predictor.scale_test_data(test_prices, test_volumes)

params_scaled = predictor.merge_data(test_prices_scaled, test_volumes_scaled)

predicted_price = predictor.predict_next_value(params_scaled)

first_date, last_date = predictor.get_trading_range()
print(f'neural network is trained on the range from {first_date} to {last_date}')
print("\n\n\n----------------------------------------------------")
print(f'Predicted price: {predicted_price:.2f}$ for {ticker} on {predictor.get_next_trading_day()} Close')

# Predicted price: 126.29 for NVDA on 29 January 2025 Close

