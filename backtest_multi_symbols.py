import time

import pandas as pd
from backtest import BarData, Position
import matplotlib.pyplot as plt
import numpy as np
from backtest import DATEMAP
from collections import deque
import random
from multiprocessing import Pool
import torch
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
        bar_date = bar.time.date()  # Extract date using the date() method

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
    def __init__(self, cash: float, feeRate: float, symbols: list):
        self.positions = {}
        self.wallet = {}
        for symbol in symbols:
            self.wallet[symbol] = 0
        self.cash = cash
        self.feeRate = feeRate
        self.totalFee = 0
        self.symbols = symbols
        self.msg = []

    def buy(self, price, symbol, vol):
        # cash = self.cash
        # if cash - price * vol * (1 + self.feeRate) < 0:
        #     # print("not enough cash, but amount that minus fees")
        #     vol -= price * vol * self.feeRate / price
        self.cash -= price * vol * (1 + self.feeRate)
        self.wallet[symbol] += vol
        self.totalFee += price * vol * self.feeRate
        msg = f"Buy {symbol} at price: {price} for {vol} amount"
        print(msg)
        msg_dict = {
            "symbol": symbol,
            "price": price,
            "vol": vol,
            "action": "buy"

        }
        self.msg.append(msg_dict)

    def sell(self, price, symbol, vol):
        self.cash += price * vol * (1 - self.feeRate)
        self.wallet[symbol] -= vol
        self.totalFee += price * vol * self.feeRate
        msg = f"Sell {symbol} at price: {price} for {vol} amount"
        print(msg)
        msg_dict = {
            "symbol": symbol,
            "price": price,
            "vol": vol,
            "action": "sell"
        }
        self.msg.append(msg_dict)

    def _update_position(self, symbol, price, vol, direction, time):
        if symbol in self.positions:
            position = self.positions[symbol]
            if position.direction == direction:
                # 加仓，计算新的加权平均价格
                new_volume = position.volume + vol
                new_avg_price = (position.avg_price * position.volume + price * vol) / new_volume
                position.avg_price = new_avg_price
                position.volume = new_volume
                position.time = time
            else:
                # 如果方向相反，考虑部分平仓或完全平仓的情况
                if abs(position.volume) > vol:
                    # 部分平仓
                    position.volume -= vol
                elif abs(position.volume) == vol:
                    # 完全平仓，删除仓位
                    del self.positions[symbol]
                else:
                    # 平掉原仓位并开立新方向的仓位
                    new_vol = vol - abs(position.volume)
                    position.direction = direction
                    position.avg_price = price
                    position.volume = new_vol
                    position.time = time
        else:
            # 开立新仓位
            self.positions[symbol] = Position(time=time, avg_price=price, symbol=symbol, direction=direction,
                                              volume=vol)

    def long(self, price, symbol, vol, leverage, time):
        """
        开立多头仓位，使用杠杆购买资产。
        :param price: 当前资产价格
        :param symbol: 资产符号
        :param vol: 购买的资产数量
        :param leverage: 杠杆倍数
        :param time: 当前时间
        """
        margin_required = (price * vol) / leverage
        # if self.cash < margin_required:
        #     print("Not enough cash to open long position with leverage.")
        #     return

        self.cash -= margin_required
        fee = price * vol * self.feeRate
        self.totalFee += fee
        self._update_position(symbol, price, vol * leverage, 'long', time)
        msg = f"Long {symbol} at price: {price} for {vol} amount with leverage {leverage}"
        print(msg)
        msg_dict = {
            "symbol": symbol,
            "price": price,
            "vol": vol,
            "action": "long"
        }
        self.msg.append(msg_dict)

    def short(self, price, symbol, vol, leverage, time):
        """
        开立空头仓位，使用杠杆卖出资产。
        :param price: 当前资产价格
        :param symbol: 资产符号
        :param vol: 卖出的资产数量
        :param leverage: 杠杆倍数
        :param time: 当前时间
        """
        margin_required = (price * vol) / leverage
        # if self.cash < margin_required:
        #     print("Not enough cash to open short position with leverage.")
        #     return

        self.cash -= margin_required
        fee = price * vol * self.feeRate
        self.totalFee += fee
        self._update_position(symbol, price, vol * leverage, 'short', time)
        msg = f"Short {symbol} at price: {price} for {vol} amount with leverage {leverage}"
        print(msg)
        msg_dict = {
            "symbol": symbol,
            "price": price,
            "vol": vol,
            "action": "short"
        }
        self.msg.append(msg_dict)

    def cover_long(self, price, symbol, vol, time):
        """
        平掉多头仓位。
        :param price: 当前资产价格
        :param symbol: 资产符号
        :param vol: 平仓的资产数量
        :param time: 当前时间
        """
        if symbol not in self.positions or self.positions[symbol].direction != 'long':
            print("No long position to cover.")
            return
        fee = price * vol * self.feeRate
        self.totalFee += fee
        position_worth = self.positions[symbol].avg_price * vol
        pnl = (price - self.positions[symbol].avg_price) * vol
        self.cash += pnl + position_worth - fee
        self._update_position(symbol, price, vol, 'cover_long', time)
        msg = f"Cover long {symbol} at price: {price} for {vol} amount"
        print(msg)
        msg_dict = {
            "symbol": symbol,
            "price": price,
            "vol": vol,
            "action": "cover long"
        }
        self.msg.append(msg_dict)

    def cover_short(self, price, symbol, vol, time):
        """
        平掉空头仓位。
        :param price: 当前资产价格
        :param symbol: 资产符号
        :param vol: 平仓的资产数量
        :param time: 当前时间
        """
        if symbol not in self.positions or self.positions[symbol].direction != 'short':
            print("No short position to cover.")
            return
        fee = price * vol * self.feeRate
        self.totalFee += fee
        position_worth = self.positions[symbol].avg_price * vol
        pnl = (self.positions[symbol].avg_price - price) * vol
        self.cash += pnl + position_worth - fee
        self._update_position(symbol, price, vol, 'cover_short', time)
        msg = f"Cover short {symbol} at price: {price} for {vol} amount"
        print(msg)
        msg_dict = {
            "symbol": symbol,
            "price": price,
            "vol": vol,
            "action": "cover short"
        }
        self.msg.append(msg_dict)

    def calculate_total_account_equity(self, current_prices: dict):
        """
        计算当前账户的总净值，包括现金和所有持仓的浮动盈亏。
        :param current_prices: dict，包含每个资产的当前价格，例如：{"BTC": 40000, "ETH": 3000}
        """
        total_pnl = 0
        position_worth = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price: BarData = current_prices[symbol]

                if position.direction == 'long':
                    pnl = (current_price.close - position.avg_price) * position.volume
                elif position.direction == 'short':
                    pnl = (position.avg_price - current_price.close) * position.volume
                position_worth += position.volume * position.avg_price
                total_pnl += pnl
        total_equity = self.cash + total_pnl + position_worth
        # print(f"Total Account Equity: {total_equity:.2f} USD")
        return total_equity

    def get_current_pos_profit_rate(self, current_prices:dict):
        total_pnl = 0
        position_worth = 0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                current_price: BarData = current_prices[symbol]

                if position.direction == 'long':
                    pnl = (current_price.close - position.avg_price) * position.volume
                elif position.direction == 'short':
                    pnl = (position.avg_price - current_price.close) * position.volume
                position_worth += position.volume * position.avg_price
                total_pnl += pnl

        profit_rate = total_pnl / position_worth
        return profit_rate




class Log:
    def __init__(self, broker: Broker):
        self.resultPath = 'backTestResult/backtestLog.csv'
        self.logs = []
        self.actionLogs = []
        self.broker = broker

    def logging(self, bars: dict, volatility: dict, wallet: dict):
        data = {}
        for symbol in bars:
            data[f"close_{symbol}"] = bars[symbol].close
            data[f"position_{symbol}"] = wallet.get(symbol, 0)
            data[f"volatility_{symbol}"] = volatility.get(symbol, 0)
            data[f"position_{symbol}_worth"] = wallet.get(symbol, 0) * bars[symbol].close

        data["position_worth"] = sum(wallet[s] * bars[s].close for s in wallet)
        data["equity"] = self.broker.cash + sum(wallet[s] * bars[s].close for s in wallet)
        data['time'] = bars[f'{symbol}'].time
        data['cash'] = self.broker.cash
        self.logs.append(data)

    def save_logs(self):
        df = pd.DataFrame(self.logs)
        df.to_csv(self.resultPath + '.csv', index=False)


class PortfolioManagement:
    """
     strategy class
    """

    def __init__(self, broker, period: str, window: int):
        self.broker = broker
        self.start = False
        self.period = period
        self.window_size = window
        self.dateCount: int = 0
        self.dateMap = DATEMAP
        self.bar_array = deque(maxlen=self.window_size)
        self.dateNum = self.dateMap[self.period]
        self.volatility = {}

    def initBuy(self, bars: dict):
        initAmount = int(self.broker.cash / len(self.broker.symbols)) * (1 - 0.01)
        for symbol in self.broker.symbols:
            bar = bars[symbol]
            qty = initAmount / bar.close
            self.broker.buy(bar.close, symbol, qty)

    def processBar(self, bars: dict):
        if not self.start:
            self.initBuy(bars)
            self.start = True

        self.bar_array.append(bars)

        self.dateCount += 1

        # Calculate volatility when enough data is accumulated
        if len(self.bar_array) == self.window_size:
            self.calculate_volatility()

        days = int(self.dateCount / 24)
        # Rebalance based on volatility every dateNum periods
        if days % self.dateNum == 0:
            ## equal weight
            # self.rebalance_equal_positions(bars)

            #  weight base on variance
            if self.dateCount < self.window_size:
                return
            self.rebalance_positions(bars)

    def rebalance_equal_positions(self, bars: dict):
        # Rebalance all positions to have equal weight
        total_equity = self.broker.cash + sum(
            self.broker.wallet[symbol] * bars[symbol].close for symbol in self.broker.symbols)
        equal_weight = total_equity / len(self.broker.symbols)

        for symbol in self.broker.symbols:
            current_price = bars[symbol].close
            target_position = equal_weight / current_price
            current_position = self.broker.wallet.get(symbol, 0)

            if target_position > current_position:
                self.broker.buy(current_price, symbol, target_position - current_position)
            elif target_position < current_position:
                self.broker.sell(current_price, symbol, current_position - target_position)

    def rebalance_positions(self, bars: dict):
        # Calculate the total volatility for weighting
        total_volatility = sum(vol for vol in self.volatility.values() if vol is not None and vol > 0)

        # Calculate target weight for each symbol based on volatility
        target_weights = {}
        for symbol, vol in self.volatility.items():
            if vol is not None and vol > 0:
                target_weights[symbol] = vol / total_volatility
            else:
                target_weights[symbol] = 0

        # Calculate current position worth for each symbol
        current_worth = {symbol: self.broker.wallet[symbol] * bars[symbol].close for symbol in self.broker.symbols}
        total_worth = sum(current_worth.values())

        # Rebalance each symbol to match the target weight
        for symbol in self.broker.symbols:
            target_worth = total_worth * target_weights[symbol]
            current_worth_symbol = current_worth[symbol]
            diff = target_worth - current_worth_symbol

            if diff > 0:
                # Buy to increase position
                amount = diff / bars[symbol].close
                self.broker.buy(bars[symbol].close, symbol, amount)
            elif diff < 0:
                # Sell to decrease position
                amount = abs(diff) / bars[symbol].close
                self.broker.sell(bars[symbol].close, symbol, amount)

    def calculate_volatility(self):
        # Calculate volatility for each symbol based on the log returns in bar_array
        for symbol in self.broker.symbols:
            close_prices = [bar[symbol].close for bar in self.bar_array if symbol in bar]
            if len(close_prices) > 1:
                # Calculate log returns
                log_returns = np.diff(np.log(close_prices))
                # Calculate the standard deviation of the log returns as volatility
                volatility = np.std(log_returns)
                self.volatility[symbol] = volatility
            else:
                # Not enough data to calculate volatility
                self.volatility[symbol] = None


class Backtestor:
    def __init__(self, broker: Broker, strategy, logger: Log, start, end):
        self.daily_data = None
        self.broker = broker
        self.strategy = strategy
        self.logger = logger
        self.symbols = self.broker.symbols
        self.data = {}
        self.data_length = None
        self.start = start
        self.end = end

    def load_data(self):
        self.daily_data = {}  # Initialize a dictionary to store daily aggregated data
        for symbol in self.symbols:
            # Load hourly data
            df = pd.read_csv(f'data/{symbol}USDT_1h_spot.csv')
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # Filter based on start and end dates
            df = df[(df['timestamp'] >= self.start) & (df['timestamp'] <= self.end)]
            self.data[f'{symbol}'] = df
            self.data_length = len(self.data[f'{symbol}'])

            # Aggregate hourly data into daily data
            df['date'] = df['timestamp'].dt.date
            daily_df = df.groupby('date').agg({
                'open': 'first',  # First hourly open of the day
                'high': 'max',  # Maximum high of the day
                'low': 'min',  # Minimum low of the day
                'close': 'last',  # Last hourly close of the day
                'volume': 'sum'  # Total volume of the day
            }).reset_index()

            # Store daily aggregated data
            self.daily_data[f'{symbol}'] = daily_df

    def run_backtest(self):
        try:
            for i in range(self.data_length):
                bars = {}
                for symbol in self.symbols:
                    data = self.data[f'{symbol}'].iloc[i]
                    bar = BarData(data['timestamp'], data['open'], data['high'], data['low'], data['close'],
                                  data['volume'])
                    bars[symbol] = bar
                self.strategy.processBar(bars)
                self.logger.logging(bars, self.strategy.volatility, self.broker.wallet)
            self.logger.save_logs()
        except Exception as e:
            print(f'{symbol}, Error: {e}')

    def plot(self):
        backtest_log = pd.DataFrame(self.log.logs)

        # Ensure 'time' is in datetime format
        if 'time' in backtest_log.columns:
            backtest_log['time'] = pd.to_datetime(backtest_log['time'])

        # Create figure and subplots
        fig, axs = plt.subplots(6, 1, figsize=(14, 18), sharex=True)

        # Plot the close price as a line chart
        for symbol in self.symbols:
            axs[0].plot(backtest_log['time'], backtest_log[f'close_{symbol}'], label=f'{symbol} Close Price')
        axs[0].set_ylabel('Close Price')
        axs[0].legend()
        axs[0].grid(True)

        # Plot cash over time
        axs[1].plot(backtest_log['time'], backtest_log['equity'], color='blue', label='Equity')
        axs[1].set_ylabel('Equity')
        axs[1].legend()
        axs[1].grid(True)

        # Plot position over time for each symbol
        for symbol in self.symbols:
            axs[2].plot(backtest_log['time'], backtest_log[f'position_{symbol}'], label=f'{symbol} Position')
        axs[2].set_ylabel('Position')
        axs[2].legend()
        axs[2].grid(True)

        # Plot position worth over time
        axs[3].plot(backtest_log['time'], backtest_log['position_worth'], color='purple', label='Position Worth')
        axs[3].set_ylabel('Position Worth')
        axs[3].legend()
        axs[3].grid(True)

        for symbol in self.symbols:
            axs[4].plot(backtest_log['time'], backtest_log[f'position_{symbol}_worth'],
                        label=f'{symbol} Position worth')
        axs[4].set_ylabel('symbol Position worth')
        axs[4].legend()
        axs[4].grid(True)

        # Plot volatility over time for each symbol
        for symbol in self.symbols:
            axs[5].plot(backtest_log['time'], backtest_log[f'volatility_{symbol}'], label=f'{symbol} Volatility')
        axs[5].set_ylabel('Volatility')
        axs[5].legend()
        axs[5].grid(True)

        # Format x-axis
        axs[5].set_xlabel('Time')
        fig.autofmt_xdate()  # Rotate date labels
        fig.suptitle('Backtest Results with Line Charts')

        # Show plot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def performance(self):
        backtest_log = pd.DataFrame(self.logger.logs)

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

        # Calculate Market Return and Market Max Drawdown based on equal positions in all symbols
        initial_market_value = sum(self.data[symbol].iloc[0]['close'] for symbol in self.symbols) / len(self.symbols)
        final_market_value = sum(self.data[symbol].iloc[-1]['close'] for symbol in self.symbols) / len(self.symbols)
        marketReturn = (final_market_value - initial_market_value) / initial_market_value

        market_values = backtest_log[[f'close_{symbol}' for symbol in self.symbols]].mean(axis=1)
        cumulative_max_market = market_values.cummax()
        market_drawdown = (market_values - cumulative_max_market) / cumulative_max_market
        max_drawdown_market = market_drawdown.min()

        # Package performance metrics in a dictionary
        performance_metrics = {
            'Cumulative Return': cumulative_return,
            'Annualized Volatility': round(volatility, 3),
            'Sharpe Ratio': round(sharpe_ratio, 3),
            'Max Drawdown': round(max_drawdown, 3),
            'Market Return': round(marketReturn, 3),
            'Market Max Drawdown': round(max_drawdown_market, 3),
            'Total Trading Fee': self.broker.totalFee,
            'Referral': self.broker.totalFee * 0.3,
            'Portfolio': self.broker.symbols
        }

        print(performance_metrics)
        return performance_metrics


@staticmethod
def select_random_symbols(symbols_list, num):
    return random.sample(symbols_list, num)


def main():
    symbols = ['ETH', 'DOGE', 'DOT', 'AVAX', 'ADA', 'IOTA', 'LINK'
        , 'TRX', 'XRP', 'UNI', 'SOL', 'LTC', 'EOS',
               'BCH', 'DASH', 'LRC']

    cash = 1000000
    selected_symbols = select_random_symbols(symbols, 6)

    transactionFeeRate = 0
    period = "1m"
    window_size = 500
    start = '2023-10-01'
    end = '2024-10-31'

    broker = Broker(cash, transactionFeeRate, selected_symbols)
    # strategy = RegularStrategy(broker, period, 400)
    strategy = PortfolioManagement(broker, period, window_size)
    log = Log(broker)

    backtestor = Backtestor(broker, strategy, log, start, end)
    backtestor.load_data()
    backtestor.run_backtest()
    result = backtestor.performance()
    return result


if __name__ == '__main__':

    # 进程池做实验
    result = []

    with Pool(10) as pool:
        # 提交 100 个异步任务到进程池
        async_results = [pool.apply_async(main) for _ in range(300)]

        # 使用 .get() 方法收集每个任务的返回值
        for r in async_results:
            result.append(r.get())

        pool.close()
        pool.join()

    # 将结果转换为 DataFrame 并保存为 CSV
    result_df = pd.DataFrame(result)
    result_df.to_csv('backTestResult/experiment/result.csv', index=False)

    # 单个实验

    # cash = 2000
    # selected_symbols = ['ETH', 'SOL']
    # transactionFeeRate = 0.001
    # period = "1d"
    # window_size = 500
    # start = '2024-3-01'
    # end = '2024-10-31'
    #
    # broker = Broker(cash, transactionFeeRate, selected_symbols)
    # # strategy = RegularStrategy(broker, period, 400)
    # strategy = PortfolioManagement(broker, period, window_size)
    # log = Log(broker)
    #
    # backtestor = Backtestor(broker, strategy, log, start, end)
    # backtestor.load_data()
    # backtestor.run_backtest()
    # result = backtestor.performance()
