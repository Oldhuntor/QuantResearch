from backtest_multi_symbols import Broker, BarData, Backtestor, Position, DailyBar
from collections import deque
from statsmodels.tsa.stattools import coint
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller
from cointegration import Log
from cointegration import Cointegration
from cointegration import BacktestorPairTrading


class BayesianPairTrading(BacktestorPairTrading):
    def __init__(self, broker: Broker, strategy, log, start, end):
        super().__init__(broker, strategy, log, start, end)
        self.log: Log = log

    def run_backtest(self):
        # Create a DailyBar instance for each symbol
        daily_bar_aggregators = {symbol: DailyBar() for symbol in self.symbols}

        for i in range(self.data_length):
            bars = {}  # Hourly bars for this iteration
            daily_bars = {}  # Store mature daily bars for this iteration

            for symbol in self.symbols:
                # Get hourly data and create a BarData instance
                data = self.data[f'{symbol}'].iloc[i]
                bar = BarData(
                    time=data['timestamp'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    volume=data['volume'],
                )
                bars[symbol] = bar

                # Feed hourly bar into the DailyBar aggregator
                daily_bar = daily_bar_aggregators[symbol].add_hourly_bar(bar)

                # If a daily bar is mature, store it for processing
                if daily_bar:
                    daily_bars[symbol] = daily_bar

            # Process hourly bars
            self.strategy.processBar(bars)

            # Process daily bars (if any mature bars are available)
            if daily_bars:
                self.strategy.process_daily_bar(daily_bars)

            # Log hourly bars
            self.log.logging(bars)

            # Calculate account equity and handle stop loss
            equity = self.broker.calculate_total_account_equity(bars)
            accountPNL = equity - self.strategy.initial_balance
            if accountPNL / self.strategy.initial_balance < -self.strategy.stop_loss:
                print(f"Stop loss triggered at {self.strategy.stop_loss}!")
                break

        # Finalize backtesting
        self.on_close(bars)
        self.log.save_logs()


class BayesianCointegration(Cointegration):
    def __init__(self, broker: Broker, setting : dict):
        super().__init__(broker, setting)
        self.global_prior_len = setting['global_prior_len']
        self.daily_bar_array = deque(maxlen=self.array_len)

    def process_daily_bar(self, bars:dict):
        self.daily_bar_array.append(bars)
        print(f"daily bars{bars}")


    def processBar(self, bars: dict):
        pass


if __name__ == '__main__':
    from backTestResult.cointegration_analysis import analyze_arbitrage_cycles

    cash = 1000000
    transactionFeeRate = 0.00025
    start = '2022-01-01'
    end = '2024-10-31'
    long_leg = 'DOGE'
    short_leg = 'XRP'
    selected_symbols = [long_leg, short_leg]
    broker = Broker(cash, transactionFeeRate, selected_symbols)
    setting = {
        'short_leg': short_leg,
        'long_leg': long_leg,
        'open_margin': 0.05,
        'close_margin': 0.04,
        'stop_loss': 0.4,
        'pValue': 0.05,
        'grid_amount': cash * 0.2,
        'array_len': 200,
        'global_prior_len' : 10,
        'maximum_holding_period': 24* 30,
    }

    strategy = BayesianCointegration(broker, setting)
    log = Log(broker)
    backtestor = BayesianPairTrading(broker, strategy, log, start, end)
    backtestor.load_data()
    backtestor.run_backtest()
    result = backtestor.performance()
    msg_df = pd.DataFrame(broker.msg)
    analyze_arbitrage_cycles(msg_df)
    msg_df.to_csv('backTestResult/msg_df.csv')