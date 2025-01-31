import yfinance as yf

# Specify the stock ticker
ticker_symbol = "AAPL"

# Fetch historical data
data = yf.download(ticker_symbol, period="max")

# Get the last available date as a string
last_date = data.index[-1].strftime("%Y-%m-%d")
print("Last available date:", last_date)
