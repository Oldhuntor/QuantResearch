import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime, timedelta
import logging


class SP500DataFetcher:
    def __init__(self, start_date, end_date, output_dir='stock_data'):
        self.output_dir = output_dir
        self.processed_count = 0
        self.total_symbols = 0

        # Set up date ranges
        self.start_date = pd.Timestamp(start_date)
        self.end_date = min(pd.Timestamp(end_date), pd.Timestamp.now())

        # Calculate hourly data range - strictly last 730 days from now
        self.today = pd.Timestamp.now()
        self.hourly_start = (self.today - pd.Timedelta(days=729)).floor('D')  # 729 to ensure we're within limit

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{output_dir}/fetch_log.txt")
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_sp500_symbols(self):
        """Get list of S&P 500 symbols using Wikipedia"""
        try:
            table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            symbols = table[0]['Symbol'].tolist()
            self.total_symbols = len(symbols)
            self.logger.info(f"\nFound {self.total_symbols} symbols in S&P 500")
            return symbols
        except Exception as e:
            self.logger.error(f"Error fetching S&P 500 symbols: {e}")
            return []

    def get_stock_data(self, symbol):
        """Fetch stock data for a single symbol"""
        try:
            ticker = yf.Ticker(symbol)

            # Get daily data for the entire historical period
            self.logger.info(
                f"\nFetching daily data for {symbol}: {self.start_date.strftime('%Y-%m-%d')} to {self.hourly_start.strftime('%Y-%m-%d')}")

            daily_df = ticker.history(
                start=self.start_date.strftime('%Y-%m-%d'),
                end=self.hourly_start.strftime('%Y-%m-%d'),
                interval='1d'
            )

            if not daily_df.empty:
                daily_file = f"{self.output_dir}/{symbol}_daily_{self.start_date.strftime('%Y%m%d')}_{self.hourly_start.strftime('%Y%m%d')}.csv"
                daily_df.to_csv(daily_file)
                self.logger.info(f"Saved daily data for {symbol}: {len(daily_df)} days")

            # Get hourly data only for last 730 days
            self.logger.info(
                f"Fetching hourly data for {symbol}: {self.hourly_start.strftime('%Y-%m-%d')} to {self.today.strftime('%Y-%m-%d')}")

            hourly_df = ticker.history(
                start=self.hourly_start.strftime('%Y-%m-%d'),
                end=self.today.strftime('%Y-%m-%d'),
                interval='1h'
            )

            if not hourly_df.empty:
                hourly_file = f"{self.output_dir}/{symbol}_hourly_{self.hourly_start.strftime('%Y%m%d')}_{self.today.strftime('%Y%m%d')}.csv"
                hourly_df.to_csv(hourly_file)
                self.logger.info(f"Saved hourly data for {symbol}: {len(hourly_df)} hours")

            # Save metadata
            metadata = {
                'symbol': symbol,
                'daily_records': len(daily_df) if not daily_df.empty else 0,
                'hourly_records': len(hourly_df) if not hourly_df.empty else 0,
                'daily_start': self.start_date.strftime('%Y-%m-%d'),
                'daily_end': self.hourly_start.strftime('%Y-%m-%d'),
                'hourly_start': self.hourly_start.strftime('%Y-%m-%d'),
                'hourly_end': self.today.strftime('%Y-%m-%d')
            }

            pd.DataFrame([metadata]).to_csv(f"{self.output_dir}/{symbol}_metadata.csv", index=False)

            self.processed_count += 1
            self.logger.info(f"Completed processing {symbol} ({self.processed_count}/{self.total_symbols})")

        except Exception as e:
            self.logger.error(f"Error processing {symbol}: {str(e)}")
            self.processed_count += 1

    def fetch_all_data(self, max_workers=5):
        """Fetch data for all S&P 500 stocks using multiple threads"""
        symbols = self.get_sp500_symbols()

        self.logger.info(f"\nStarting data fetch with {max_workers} workers")
        self.logger.info(
            f"Daily data range: {self.start_date.strftime('%Y-%m-%d')} to {self.hourly_start.strftime('%Y-%m-%d')}")
        self.logger.info(
            f"Hourly data range: {self.hourly_start.strftime('%Y-%m-%d')} to {self.today.strftime('%Y-%m-%d')}")
        self.logger.info("=" * 50)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(self.get_stock_data, symbols)

        self.logger.info("\nData fetch completed!")
        self.logger.info(f"Successfully processed {self.processed_count} symbols")
        self.logger.info("=" * 50)


def main():
    START_DATE = "2019-01-01"
    END_DATE = "2024-12-01"

    try:
        fetcher = SP500DataFetcher(START_DATE, END_DATE)
        fetcher.fetch_all_data()
    except ValueError as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()