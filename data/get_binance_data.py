import requests
import pandas as pd
from datetime import datetime, timedelta
from zipfile import ZipFile
import io
import numpy as np

def construct_binance_url(data_category='spot', market_type=None, interval_type='daily', data_type='klines'):

    base_url = "https://data.binance.vision/data"

    if data_category == 'futures':
        if not market_type:
            raise ValueError("For 'futures' data_category, 'market_type' must be specified ('cm' or 'um').")
        prefix = f"{data_category}/{market_type}/{interval_type}/{data_type}/"
    elif data_category == 'spot':
        prefix = f"{data_category}/{interval_type}/{data_type}/"
    else:
        raise ValueError("Invalid data_category. Use 'spot' or 'futures'.")

    full_url = f"{base_url}/{prefix}"
    return full_url


def download_data(data_category='spot', interval_type='daily', market_type=None, data_type='klines', symbol='BTCUSDT',
                  start_date: str = '2024-10-01', end_date: str = '2024-10-10', interval='1d'):

    base_url = construct_binance_url(data_category, market_type, interval_type, data_type)

    # Convert start and end dates to datetime objects
    start = datetime.strptime(start_date, '%Y-%m-%d')
    end = datetime.strptime(end_date, '%Y-%m-%d')

    combined_data = pd.DataFrame()

    # Loop through each date in the range
    current = start
    while current <= end:
        if interval_type == 'daily':
            date_str = current.strftime('%Y-%m-%d')
            filename = f"{symbol}-{interval}-{date_str}.zip"
        elif interval_type == 'monthly':
            date_str = current.strftime('%Y-%m')
            filename = f"{symbol}-{interval}-{date_str}.zip"
        else:
            raise ValueError("Invalid interval_type. Use 'daily' or 'monthly'.")

        url = f"{base_url}{symbol}/{interval}/{filename}"

        try:
            # Download the zip file
            response = requests.get(url)
            response.raise_for_status()  # Raise an error for failed requests

            # Unzip and read CSV file from zip
            with ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]  # There should be only one file in each zip
                with z.open(csv_filename) as csv_file:
                    df = pd.read_csv(csv_file, header=None)
                    if df[0][0] == 'open_time':
                        df = df[1:]
                        df[0] = df[0].apply(np.int64)
                    df[0] = pd.to_datetime(df[0], unit='ms')
                    combined_data = pd.concat([combined_data, df], ignore_index=True)

            print(f"Downloaded and processed: {filename}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {filename}: {e}")

            # Move to the next day or month
        if interval_type == 'daily':
            current += timedelta(days=1)
        elif interval_type == 'monthly':
            current += timedelta(days=31)
            current = current.replace(day=1)

    # Set column names based on Binance Kline format
    combined_data.columns = [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ]

    return combined_data


data_category = 'futures'  # spot futures
interval_type = 'monthly'  # daily monthly
date_type = 'klines'  # aggTrades bookTicker fundingRate trades

start_date = '2024-01-01'
end_date = '2024-03-01'
interval = '1d'  # 12h 15m 1d 1h 1m 1mo 1w 2h 30m 3d 3m 4h 5m 6h 8h
market_type ='cm' # cm um, for spot None

symbol = ['DOTUSDT', 'LINKUSDT', "BCHUSDT", 'SHIBUSDT', 'SUIUSDT', 'UNIUSDT','DAIUSDT', 'CROUSDT', 'TAOUSDT']
symbol = ['BTCUSD_240329']
for symbol in symbol:
    data = download_data(data_category, interval_type, market_type, date_type, symbol, start_date,
                         end_date, interval)

    data.to_csv(f"{symbol}_{interval}_{data_category}.csv", index=False)

