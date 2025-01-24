import pytest
import numpy as np
import pandas as pd
from recurrent_network import StockPredictor

@pytest.fixture
def predictor():
    return StockPredictor()

def test_generate_dummy_data(predictor):
    df = predictor.generate_dummy_data()
    assert isinstance(df, pd.DataFrame)
    assert 'prices' in df.columns
    assert len(df) == 7

def test_normalize_data(predictor):
    df = predictor.generate_dummy_data()
    scaled_data = predictor.normalize_data(df)
    assert scaled_data.shape == (7, 1)
    assert np.min(scaled_data) >= 0
    assert np.max(scaled_data) <= 1

def test_prepare_data(predictor):
    df = predictor.generate_dummy_data()
    scaled_data = predictor.normalize_data(df)
    X_train, y_train = predictor.prepare_data(scaled_data, time_step=3)
    assert X_train.shape == (4, 3, 1)
    assert y_train.shape == (4,)

def test_build_and_train_model(predictor):
    df = predictor.generate_dummy_data()
    scaled_data = predictor.normalize_data(df)
    X_train, y_train = predictor.prepare_data(scaled_data, time_step=3)
    predictor.build_and_train_model(X_train, y_train, epochs=1, batch_size=1)
    assert predictor.model is not None

def test_predict_future_prices(predictor):
    df = predictor.generate_dummy_data()
    scaled_data = predictor.normalize_data(df)
    X_train, y_train = predictor.prepare_data(scaled_data, time_step=3)
    predictor.build_and_train_model(X_train, y_train, epochs=1, batch_size=1)
    predicted_prices = predictor.predict_future_prices(scaled_data, days=2)
    assert len(predicted_prices) == 2
    assert isinstance(predicted_prices, np.ndarray)


