from statsmodels.tsa.stattools import coint
import numpy as np
import pandas as pd

futures_df = pd.read_csv('BTCUSD_241227_1h_futures.csv')
spot_df = pd.read_csv('backup/BTCUSDT_1h_spot.csv')
# Convert the 'timestamp' column to datetime
futures_df['timestamp'] = pd.to_datetime(futures_df['timestamp'])
spot_df['timestamp'] = pd.to_datetime(spot_df['timestamp'])

# Merge the two dataframes on the 'timestamp' column to ensure matching timestamps
merged_df = pd.merge(futures_df, spot_df, on='timestamp', suffixes=('_futures', '_spot'))

# Extract the closing prices for cointegration analysis
futures_close = merged_df['close_futures']
spot_close = merged_df['close_spot']

# Perform the cointegration test
coint_result = coint(futures_close, spot_close)
coint_stat, p_value, critical_values = coint_result

# Display the cointegration results
print({
    'Cointegration Statistic': coint_stat,
    'P-Value': p_value,
    'Critical Values': critical_values
})


import matplotlib.pyplot as plt

# Calculate the spread and spread percentage
spread = futures_close - spot_close
spread_percentage = (spread / spot_close) * 100

# Plot the closing prices of futures and spot
plt.figure(figsize=(12, 6))
plt.plot(merged_df['timestamp'], futures_close, label='Futures Close Price', color='blue')
plt.plot(merged_df['timestamp'], spot_close, label='Spot Close Price', color='orange')
plt.title('BTC Futures vs Spot Closing Prices')
plt.xlabel('Time')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid()
plt.show()

# Plot the spread percentage
plt.figure(figsize=(12, 6))
plt.plot(merged_df['timestamp'], spread_percentage, label='Spread Percentage', color='green')
plt.title('Spread (Price difference)')
plt.xlabel('Time')
plt.ylabel('Spread Percentage (%)')
plt.legend()
plt.grid()
plt.show()

# Displaying a summary of the spread percentage
spread_percentage_summary = {
    'Mean Spread Percentage': spread_percentage.mean(),
    'Standard Deviation of Spread Percentage': spread_percentage.std(),
    'Max Spread Percentage': spread_percentage.max(),
    'Min Spread Percentage': spread_percentage.min()
}

print(spread_percentage_summary)
