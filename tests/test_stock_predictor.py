import pytest
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from model import Stock_Predictor
from data_loader import Data_Loader

@pytest.fixture
def predictor():
    predictor = Stock_Predictor()
    predictor.set_window_size(3)
    return predictor

@pytest.fixture
def data_loader_obj():
    data_loader = Data_Loader('NVDA', 3)
    return data_loader

def test_generate_dummy_data(data_loader_obj):
    open_prices, close_prices, volumes = data_loader_obj.generate_dummy_data()
    assert len(open_prices) > 0
    assert len(volumes) > 0
    assert len(close_prices) == len(volumes)

def test_normalize_data(data_loader_obj):
    open_prices, close_prices, volumes = data_loader_obj.generate_dummy_data()
    X_prices_open, X_prices_close, X_volumes, y = data_loader_obj.normalize_data(open_prices, close_prices, volumes)
    assert X_prices_open.shape[0] == X_volumes.shape[0] == y.shape[0]
    assert X_prices_open.shape[1] == X_volumes.shape[1] == data_loader_obj.window_size

def test_scale(predictor):
    prices, volumes = predictor.generate_dummy_data()
    X_prices, X_volumes, y = predictor.normalize_data(prices, volumes)
    X_prices_scaled, X_volumes_scaled, y_scaled = predictor.scale(X_prices, X_volumes, y)
    assert X_prices_scaled.shape == X_prices.shape
    assert X_volumes_scaled.shape == X_volumes.shape
    assert y_scaled.shape == y.shape

def test_build_model(predictor):
    predictor.build_model()
    assert predictor.model is not None

def test_train_model(predictor):
    prices, volumes = predictor.generate_dummy_data()
    X_prices, X_volumes, y = predictor.normalize_data(prices, volumes)
    X_prices_scaled, X_volumes_scaled, y_scaled = predictor.scale(X_prices, X_volumes, y)
    predictor.build_model()
    predictor.train_model(X_prices_scaled, X_volumes_scaled, y_scaled, epochs=1)
    assert predictor.model is not None

def test_predict_next_value(predictor):
    predictor.build_model()
    test_prices = np.array([6400, 12800, 25600]).reshape(1, 3, 1)
    test_volumes = np.array([1, 1, 1]).reshape(1, 3, 1)
    test_prices_scaled, test_volumes_scaled = predictor.scale_test_data(test_prices, test_volumes)
    predicted_price = predictor.predict_next_value(test_prices_scaled, test_volumes_scaled)
    assert predicted_price is not None