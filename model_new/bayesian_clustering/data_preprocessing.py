import pandas as pd
import numpy as np
import os
import re

def organize_stock_data(data_dir, interval='daily'):
    """Organizes stock data into a wide format (transposed)."""

    all_data = []
    for filename in os.listdir(data_dir):
        match = re.match(r"([A-Z]+)_(daily|hourly)_(\d+)_(\d+)\.csv", filename)
        if match:
            symbol, file_interval, _, _ = match.groups()
            if file_interval == interval:
                try:
                    df = pd.read_csv(os.path.join(data_dir, filename))
                    df['Symbol'] = symbol
                    all_data.append(df)
                except FileNotFoundError:
                    print(f"File not found: {filename}")
                except pd.errors.ParserError:
                    print(f"Error parsing {filename}. Check the file format.")
                except Exception as e:
                    print(f"An unexpected error occurred with {filename}: {e}")

    if not all_data:
        print("No data found for the specified interval.")
        return None

    all_df = pd.concat(all_data)
    all_df = all_df.set_index(['Symbol', 'Datetime'])

    try:
        close_prices = all_df['Close'].unstack(level=0)
    except KeyError as e:
        print(f"Error: 'Close' column not found: {e}")
        return None

    close_prices = close_prices.dropna(axis=1, how='all')
    close_prices = close_prices.ffill()
    close_prices = close_prices.bfill()

    transposed_data = close_prices.T  # Transpose the DataFrame

    return transposed_data

# Example usage:
if __name__ == "__main__":
    data_directory = "/Users/hxh/PycharmProjects/Quantitative_Research/data/sp500_data/stock_data"
    interval = 'hourly'  # Or 'daily'
    organized_data = organize_stock_data(data_directory, interval)

    if organized_data is not None:
        print(organized_data.head())
        organized_data.to_csv("transposed_stock_data.csv")
        print(f"Shape of transposed data: {organized_data.shape}")