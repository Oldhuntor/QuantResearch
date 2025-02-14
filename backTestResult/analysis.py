import pandas as pd
import matplotlib.pyplot as plt

# Load the newly uploaded performance metrics data
file_path_latest_upload = '/Users/hxh/PycharmProjects/RL/pythonProject1/backTestResult/experiment/result.csv'
data_latest_upload = pd.read_csv(file_path_latest_upload)

# Extract all required metrics for analysis
cumulative_returns_latest_upload = data_latest_upload['Cumulative Return']
max_drawdowns_latest_upload = data_latest_upload['Max Drawdown']
market_returns_latest_upload = data_latest_upload['Market Return']
market_max_drawdowns_latest_upload = data_latest_upload['Market Max Drawdown']

# data_latest_upload['win'] = data_latest_upload[(data_latest_upload['Cumulative Return'] > data_latest_upload['Market Return'])]

# Plotting Pareto front for strategy vs. market with all data points (latest uploaded data)
plt.figure(figsize=(10, 6))

# Plot all strategy metrics (latest uploaded data)
plt.scatter(max_drawdowns_latest_upload, cumulative_returns_latest_upload, color='blue', label='Strategy', s=10, alpha=0.6)

# Plot all market metrics (latest uploaded data)
plt.scatter(market_max_drawdowns_latest_upload, market_returns_latest_upload, color='red', label='Market', s=10, alpha=0.6)

# Adding labels and title
plt.xlabel('Max Drawdown')
plt.ylabel('Cumulative Return')
plt.title('Pareto Front: Strategy vs Market Performance (Latest Uploaded Data)')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()
