import pandas as pd

spot = pd.read_csv('backup/BTCUSDT_1h_spot.csv')
futures = pd.read_csv('BTCUSD_241227_1h_futures.csv')


# Convert the 'timestamp' column to datetime format for both datasets
futures['timestamp'] = pd.to_datetime(futures['timestamp'])
spot['timestamp'] = pd.to_datetime(spot['timestamp'])
# Set timestamp as the index for both DataFrames
futures.set_index('timestamp', inplace=True)
spot.set_index('timestamp', inplace=True)

merged_data = futures.merge(spot, how='inner', left_index=True, right_index=True, suffixes=('_futures', '_spot'))
merged_data.reset_index(inplace=True)

merged_data.to_csv('BTCUSD_241227_spot_1h_merged.csv')
# futures_timestamps = futures['timestamp']
# spot_timestamps = spot['timestamp']
#
# missing_timestamps = spot_timestamps - futures_timestamps
#
# filtered_spot_data = spot[~spot['timestamp'].isin(missing_timestamps)]

# missing_period_start = pd.to_datetime('2024-10-22')
# missing_period_end = pd.to_datetime('2024-10-27')
#
# # Filter futures: Keep rows outside the missing period
# futures_filtered = futures[
#     (futures['timestamp'] < missing_period_start) | (futures['timestamp'] > missing_period_end)
# ]
#
# # Filter spot: Keep rows outside the missing period
# spot_filtered = spot[
#     (spot['timestamp'] < missing_period_start) | (spot['timestamp'] > missing_period_end)
# ]