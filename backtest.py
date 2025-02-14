import pandas as pd
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Position:
    time: str
    avg_price: float
    symbol: str
    direction: str
    volume: float = 0.0


DATEMAP = {"1d": 1,
           "7d": 7,
           "1m": 30,
           "3m": 90, }


@dataclass
class BarData:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float

from dataclasses import dataclass

@dataclass
class BarData:
    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class DailyBar:
    def __init__(self):
        self.current_day = None
        self.hourly_bars = []  # Store hourly bars for the current day

    def add_hourly_bar(self, bar: BarData):
        """
        Add an hourly bar to the aggregator. If the day changes, return the aggregated daily bar.
        """
        bar_date = bar.time.split(' ')[0]  # Extract date from the timestamp

        if self.current_day is None:
            self.current_day = bar_date

        if bar_date != self.current_day:
            # The day has changed, aggregate the current day's bars
            daily_bar = self._aggregate_daily_bar()
            # Reset for the new day
            self.current_day = bar_date
            self.hourly_bars = [bar]  # Start accumulating bars for the new day
            return daily_bar  # Return the completed daily bar

        # Accumulate the bar for the current day
        self.hourly_bars.append(bar)
        return None  # Not enough data to produce a daily bar yet

    def _aggregate_daily_bar(self):
        """
        Aggregate the hourly bars into a daily bar.
        """
        if not self.hourly_bars:
            return None  # No data to aggregate

        open_price = self.hourly_bars[0].open
        high_price = max(bar.high for bar in self.hourly_bars)
        low_price = min(bar.low for bar in self.hourly_bars)
        close_price = self.hourly_bars[-1].close
        total_volume = sum(bar.volume for bar in self.hourly_bars)

        return BarData(
            time=self.current_day,
            open=open_price,
            high=high_price,
            low=low_price,
            close=close_price,
            volume=total_volume,
        )


class Broker:
    def __init__(self, cash: float, feeRate: float):
        self.position = 0
        self.cash = cash
        self.feeRate = feeRate
        self.totalFee = 0

    def buy(self, price, vol):
        cash = self.cash
        if cash - price * vol * (1 + self.feeRate) < 0:
            print("not enough cash")
            return
        self.cash -= price * vol * (1 + self.feeRate)
        self.position += vol
        self.totalFee += price * vol * self.feeRate
        print(f"Buy{self.position} at {price} at {vol}")

    def sell(self, price, vol):
        self.cash += price * vol * (1 - self.feeRate)
        self.position -= vol
        self.totalFee += price * vol * self.feeRate
        print(f"sell{self.position} at {price} at {vol}")


class Log:
    def __init__(self, broker: Broker):
        self.resultPath = 'backTestResult/backtestLog.csv'
        self.logs = []
        self.actionLogs = []
        self.broker = broker

    def logging(self, bar: BarData):
        data = {
            "cash": self.broker.cash,
            "position": self.broker.position,
            "position_worth": self.broker.position * bar.close,
            "equity": self.broker.cash + self.broker.position * bar.close,
            "open": bar.close,
            "high": bar.close,
            "low": bar.close,
            "close": bar.close,
            "volume": bar.volume,
            "time": bar.time,
        }

        self.logs.append(data)

    def logAction(self, bar: BarData, action: str):
        data = {
            "action": action,
            "time": bar.time,
            "price": bar.close,
        }
        self.actionLogs.append(data)

    def save_logs(self):
        df = pd.DataFrame(self.logs)
        df.to_csv(self.resultPath + '.csv')


class RebalanceStrategy:
    """
    infinity grid strategy class
    """

    def __init__(self, broker: Broker, period: str, ):
        self.broker = broker
        self.start = False
        self.period = period
        self.dateCount: int = 0
        self.dateMap = DATEMAP
        self.dateNum = self.dateMap[self.period]

    def initBuy(self, bar: BarData):
        close = bar.close
        initAmount = (self.broker.cash / 2) / close
        self.broker.buy(bar.close, initAmount)

    def processBar(self, bar: BarData):
        if not self.start:
            self.initBuy(bar)
            self.start = True

        currentPositionWorth = self.broker.position * bar.close
        diff = currentPositionWorth - self.broker.cash
        Amount = abs(diff / 2 / bar.close)
        self.dateCount += 1

        if self.dateCount % self.dateMap[self.period] == 0:

            if diff > 0:
                self.broker.sell(bar.close, Amount)
            else:
                self.broker.buy(bar.close, Amount)


class RegularStrategy(RebalanceStrategy):
    """
    regular investing period 7d 1m
    """

    def __init__(self, broker: Broker, period: str, buyAmount: float):
        super().__init__(broker, period)
        self.broker = broker
        self.period = period
        self.buyAmount = buyAmount
        self.start = False
        self.dateCount: int = 0
        self.dateMap = DATEMAP
        self.dateNum = self.dateMap[self.period]

    def processBar(self, bar: BarData):
        self.dateCount += 1
        if self.dateCount % self.dateNum == 0:
            self.broker.buy(bar.close, self.buyAmount / bar.close)


class Backtestor:
    def __init__(self, broker: Broker, symbol: str, strategy: RebalanceStrategy, log: Log):
        self.broker = broker
        self.strategy = strategy
        self.log = log
        self.symbol = symbol

    def load_data(self):
        self.df = pd.read_csv(f'data/{self.symbol}.csv')

    def run_backtest(self):
        for i in range(len(self.df)):
            data = self.df.iloc[i]
            bar = BarData(data['timestamp'], data['open'], data['high'], data['low'], data['close'], data['volume'])
            self.strategy.processBar(bar)
            self.log.logging(bar)

        self.log.save_logs()

    def plot(self):
        backtest_log = pd.DataFrame(self.log.logs)

        # Ensure 'time' is in datetime format
        if 'time' in backtest_log.columns:
            backtest_log['time'] = pd.to_datetime(backtest_log['time'])

        # Create figure and subplots
        fig, axs = plt.subplots(5, 1, figsize=(14, 18), sharex=True)

        # Plot the close price as a line chart
        axs[0].plot(backtest_log['time'], backtest_log['close'], color='black', label='Close Price')
        axs[0].set_ylabel('Close Price')
        axs[0].legend()
        axs[0].grid(True)

        # Plot cash over time
        axs[1].plot(backtest_log['time'], backtest_log['cash'], color='blue', label='Cash')
        axs[1].set_ylabel('Cash')
        axs[1].legend()
        axs[1].grid(True)

        # Plot position over time
        axs[2].plot(backtest_log['time'], backtest_log['position'], color='green', label='Position')
        axs[2].set_ylabel('Position')
        axs[2].legend()
        axs[2].grid(True)

        # Plot position worth over time
        axs[3].plot(backtest_log['time'], backtest_log['position_worth'], color='purple', label='Position Worth')
        axs[3].set_ylabel('Position Worth')
        axs[3].legend()
        axs[3].grid(True)

        # Plot equity over time
        axs[4].plot(backtest_log['time'], backtest_log['equity'], color='red', label='Equity')
        axs[4].set_ylabel('Equity')
        axs[4].legend()
        axs[4].grid(True)

        # Format x-axis
        axs[4].set_xlabel('Time')
        fig.autofmt_xdate()  # Rotate date labels
        fig.suptitle('Backtest Results with Line Charts')

        # Show plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def performance(self):
        backtest_log = pd.DataFrame(self.log.logs)

        # Calculate returns
        backtest_log['returns'] = backtest_log['equity'].pct_change()

        # Cumulative Return
        cumulative_return = (backtest_log['equity'].iloc[-1] / backtest_log['equity'].iloc[0]) - 1

        # Annualized Volatility (assuming daily data)
        volatility = backtest_log['returns'].std() * np.sqrt(252)

        # Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
        sharpe_ratio = backtest_log['returns'].mean() / backtest_log['returns'].std() * np.sqrt(252)

        # Maximum Drawdown
        cumulative_max = backtest_log['equity'].cummax()
        drawdown = (backtest_log['equity'] - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()

        marketReturn = (self.df.iloc[-1]['close'] - self.df.iloc[0]['close']) / self.df.iloc[0]['close']
        # Maximum Drawdown of the market itself (based on closing prices)
        cumulative_max_market = backtest_log['close'].cummax()
        market_drawdown = (backtest_log['close'] - cumulative_max_market) / cumulative_max_market
        max_drawdown_market = market_drawdown.min()
        # Package performance metrics in a dictionary
        performance_metrics = {
            'Cumulative Return': cumulative_return,
            'Annualized Volatility': volatility,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Market Return': marketReturn,
            'Market Max Drawdown': max_drawdown_market,
            'total Trading Fee': self.broker.totalFee,
            'referral': self.broker.totalFee * 0.3
        }

        print(performance_metrics)
        return performance_metrics


if __name__ == '__main__':
    cash = 2200
    symbol = "BTCUSDT"
    transactionFeeRate = 0.005
    period = "1m"

    broker = Broker(cash, transactionFeeRate)
    # strategy = RegularStrategy(broker, period, 400)
    strategy = RebalanceStrategy(broker, period)
    log = Log(broker)

    backtestor = Backtestor(broker, symbol, strategy, log)
    backtestor.load_data()
    backtestor.run_backtest()
    backtestor.performance()
    backtestor.plot()
