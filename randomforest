# pip install yfinance scikit-learn pandas

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import datetime as dt

# Download historical stock data
def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Prepare data for modeling
def prepare_data(data):
    data['Date'] = data.index
    data['Day'] = data['Date'].dt.day
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    data['Previous Close'] = data['Close'].shift(1)
    data = data.dropna()
    X = data[['Day', 'Month', 'Year', 'Previous Close']]
    y = data['Close']
    return X, y

# Train model and make prediction
def predict_stock_price(ticker, Day, Month, Year, days_ago=365):
    today = dt.date.today()
    start_date = today - dt.timedelta(days=days_ago)
    end_date = today
    
    # Download data
    data = download_stock_data(ticker, start_date, end_date)
    
    # Prepare data
    X, y = prepare_data(data)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    #print(X_train.values)
    model.fit(X_train.values, y_train.values.ravel())
    
    # Test model accuracy
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Mean Absolute Error: {mae}")
    
    # Predict today's stock price
    today_features = pd.DataFrame({
        'Day': Day,
        'Month': Month,
        'Year': Year,
        'Previous Close': [data['Close'].iloc[-1]]
    })
    #predicted_price = model.predict(today_features)[0]
    predicted_price = model.predict(today_features.values)[0]
    return predicted_price

# Main execution
if __name__ == "__main__":
    ticker = "NVDA"  # Nvidia's stock ticker symbol
    today = dt.date.today()
    Day = '23'
    Month = '1'
    Year = '2025'
    days_ago = 2*365  # Number of days of historical data to use for prediction

    predicted_price = predict_stock_price(ticker, Day, Month, Year, days_ago)
    print(f"Predicted stock price for {ticker} on {Day}-{Month}-{Year}: ${predicted_price:.2f}")
