from stock_predictor import Stock_Predictor
import numpy as np

# hyperparameters
epochs = 1000
window_size = 10 # set to trading period (3, 5, 7, 10, 20, 30, 60, 90, 120, 180, 365)
tcker = 'NVDA'

predictor = Stock_Predictor(tcker)
#prices, volumes = predictor.generate_dummy_data()
prices, volumes = predictor.download_stock_data('2025-01-01', '2025-01-29')
if (len(prices) < window_size):
    print('Not enough data! Decrease window_size or increase data range in download_stock_data()')
    exit(1)
predictor.set_window_size(window_size)

# normalize so that X_prices are array of matrices of shape (1, window_size)
X_prices, X_volumes, y = predictor.normalize_data(prices, volumes)
#print('X_prices:', X_prices)
X_prices_scaled, X_volumes_scaled, y_scaled = predictor.scale(X_prices, X_volumes, y)
predictor.build_model()
predictor.train_model(X_prices_scaled, X_volumes_scaled, y_scaled, epochs=epochs)

#test_prices = np.array([6400, 12800, 25600]).reshape(1, 3, 1)
#test_volumes = np.array([1, 1, 1]).reshape(1, 3, 1)
print('test prices:', prices[-window_size:])
test_prices = np.array(prices[-window_size:]).reshape(1, window_size, 1)
test_volumes = np.array(volumes[-window_size:]).reshape(1, window_size, 1)
test_prices_scaled, test_volumes_scaled = predictor.scale_test_data(test_prices, test_volumes)
predicted_price = predictor.predict_next_value(test_prices_scaled, test_volumes_scaled)
print("Predicted next price at Close:", predicted_price)
