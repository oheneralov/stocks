import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Load sample stock price data (replace this with your actual dataset)
data = pd.DataFrame({
    'Price': np.sin(np.linspace(0, 100, 1000)) * 100 + 1000 + np.random.normal(0, 5, 1000)
})

# Prepare the data
sequence_length = 10

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

x_data = []
y_data = []

# Create sequences for LSTM
for i in range(len(scaled_data) - sequence_length):
    x_data.append(scaled_data[i:i + sequence_length - 1, 0])
    y_data.append(scaled_data[i + sequence_length - 1, 0])

x_data = np.array(x_data)
y_data = np.array(y_data)

# Reshape input data for LSTM [samples, timesteps, features]
x_data = np.reshape(x_data, (x_data.shape[0], x_data.shape[1], 1))

# Split into training and testing data
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length - 1, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))

# Predict example
predicted_values = model.predict(x_test)

# Inverse transform to get real stock prices
predicted_values = scaler.inverse_transform(predicted_values.reshape(-1, 1))
real_values = scaler.inverse_transform(y_test.reshape(-1, 1))

print("First 5 predicted vs real stock prices:")
for i in range(5):
    print(f"Predicted: {predicted_values[i][0]:.2f}, Real: {real_values[i][0]:.2f}")
