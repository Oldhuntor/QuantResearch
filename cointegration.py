from backtest_multi_symbols import Broker, BarData, Backtestor, Position, DailyBar
from collections import deque
from statsmodels.tsa.stattools import coint
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import adfuller

from model_new.bayesian_regression import bayesian_rolling_window
from model_new.jointCov2 import fit_HGP
from model_new.BayesGAM import bayesian_gam_with_splines
from model_new.BSTS import bsts_fit



class Log:
    def __init__(self, broker: Broker):
        self.resultPath = 'backTestResult/backtestLog'
        self.logs = []
        self.broker = broker

    def logging(self, bars: dict):
        data = {}
        for symbol in bars:
            data[f"close_{symbol}"] = bars[symbol].close
            pos = self.broker.positions.get(symbol, 0)
            if pos != 0:
                data[f"position_{symbol}"] = self.broker.positions[symbol].volume
                data[f"position_{symbol}_worth"] = self.broker.positions[symbol].volume * bars[symbol].close

            data['time'] = bars[symbol].time

        equity = self.broker.calculate_total_account_equity(bars)
        data["equity"] = equity
        data['cash'] = self.broker.cash
        self.logs.append(data)

    def save_logs(self):
        df = pd.DataFrame(self.logs)
        # df.to_csv(self.resultPath + '.csv', index=False)


class BacktestorPairTrading(Backtestor):
    def __init__(self, broker: Broker, strategy, log, start, end):
        super().__init__(broker, strategy, log, start, end)
        self.log: Log = log

    def run_backtest(self):
        for i in range(self.data_length):
            bars = {}
            for symbol in self.symbols:
                data = self.data[f'{symbol}'].iloc[i]
                bar = BarData(data['timestamp'], data['open'], data['high'], data['low'], ((data['close'])),
                              data['volume'])
                bars[symbol] = bar
            self.strategy.processBar(bars)
            self.log.logging(bars)
            equity = self.broker.calculate_total_account_equity(bars)
            accountPNL = equity - self.strategy.initial_balance
            if accountPNL/self.strategy.initial_balance < -self.strategy.stop_loss:
                print(f"stop loss at {self.strategy.stop_loss} !!!")
                break

        self.on_close(bars)
        self.log.save_logs()

    def on_close(self, bars):
        if self.strategy.is_open:
            print('out of time force close')
            self.strategy.close_position_new(bars, ignore_diff=True)


class Cointegration:
    def __init__(self, broker: Broker, setting: dict):
        self.short_leg = setting['short_leg']
        self.long_leg = setting['long_leg']
        self.open_margin = setting['open_margin']
        self.close_margin = setting['close_margin']
        self.grid_amount = setting['grid_amount']
        self.array_len = setting['array_len']
        self.stop_loss = setting['stop_loss']
        self.pValue = setting['pValue']
        self.initial_balance = broker.cash
        self.maximum_holding_period = setting['maximum_holding_period']
        self.broker = broker
        self.beta = None
        self.intercept = None
        self.mean = None
        self.dev = None
        self.start = False
        self.open_diff = None
        self.bar_array = deque(maxlen=self.array_len)
        self.spread_array = deque(maxlen=self.array_len)
        self.is_open = False
        self.holding_count = 0

    def get_model(self):
        pass

    def get_legs(self):
        long_legs = []
        short_legs = []
        for i in range(self.array_len):

            bar = self.bar_array[i]
            long_legs.append(np.log(bar[self.long_leg]))
            short_legs.append(np.log(bar[self.short_leg]))

        return long_legs, short_legs

    def get_beta(self):
        long_legs, short_legs = self.get_legs()

        # 使用 scikit-learn 回归模型计算 Beta（不带截距）
        long_legs = np.array(long_legs).reshape(-1, 1)
        short_legs = np.array(short_legs)
        model = LinearRegression(fit_intercept=True).fit(long_legs, short_legs)
        beta = model.coef_[0]
        intercept = model.intercept_
        return beta, intercept

    def get_beta_BayesRegression(self):
        long_legs, short_legs = self.get_legs()
        result = bayesian_rolling_window(long_legs, short_legs, window_size=20)

        y_values = result['y']
        beta_values = result['beta']
        mu_values = result['mu']
        epsilon_values = result['epsilon']

        y_pred_mean = y_values['mean']
        y_pred_upper = y_values['upper']
        y_pred_lower = y_values['lower']

        beta_posterior = beta_values['mean']
        beta_upper = beta_values['upper']
        beta_lower = beta_values['lower']

        mu_posterior = mu_values['mean']
        mu_upper = mu_values['upper']
        mu_lower = mu_values['lower']

        epsilon_mean = epsilon_values['mean']
        epsilon_upper = epsilon_values['upper']
        epsilon_lower = epsilon_values['lower']

        return beta_posterior[-1], mu_posterior[-1]

    def get_beta_GPs(self):
        long_legs, short_legs = self.get_legs()
        long_legs = np.array(long_legs)
        short_legs = np.array(short_legs)
        result = fit_HGP(long_legs, short_legs)

        y_values = result['y']
        beta_values = result['beta']
        mu_values = result['mu']
        epsilon_values = result['epsilon']

        y_pred_mean = y_values['mean']
        y_pred_upper = y_values['upper']
        y_pred_lower = y_values['lower']

        beta_posterior = beta_values['mean']
        beta_upper = beta_values['upper']
        beta_lower = beta_values['lower']

        mu_posterior = mu_values['mean']
        mu_upper = mu_values['upper']
        mu_lower = mu_values['lower']

        epsilon_mean = epsilon_values['mean']
        epsilon_upper = epsilon_values['upper']
        epsilon_lower = epsilon_values['lower']

        return np.float64(beta_posterior[-1]), np.float64(mu_posterior[-1])

    def get_beta_BSTS(self):
        pass

    def get_beta_GAM(self):
        long_legs = []
        short_legs = []
        for i in range(self.array_len):
            bar = self.bar_array[i]
            long_legs.append(bar[self.long_leg])
            short_legs.append(bar[self.short_leg])
        long_legs = np.array(long_legs)
        short_legs = np.array(short_legs)

        result = bayesian_gam_with_splines(long_legs,short_legs,df=10)
        y_values = result['y']
        beta_values = result['beta']
        mu_values = result['mu']
        epsilon_values = result['epsilon']

        y_pred_mean = y_values['mean']
        y_pred_upper = y_values['upper']
        y_pred_lower = y_values['lower']

        beta_posterior = beta_values['mean']
        beta_upper = beta_values['upper']
        beta_lower = beta_values['lower']

        mu_posterior = mu_values['mean']
        mu_upper = mu_values['upper']
        mu_lower = mu_values['lower']

        epsilon_mean = epsilon_values['mean']
        epsilon_upper = epsilon_values['upper']
        epsilon_lower = epsilon_values['lower']

        return beta_posterior[-1], mu_posterior[-1]


    def processBar(self, bars: dict):
        from math import log
        # append log data
        bar = {
            self.short_leg: (bars[self.short_leg].close),
            self.long_leg: (bars[self.long_leg].close),
        }
        self.bar_array.append(bar)
        if len(self.bar_array) < self.array_len:
            return

        p_value = self.coin_test()

        spread, _ = self.spread(bars)
        self.spread_array.append(spread)
        if len(self.spread_array) < self.array_len/5:
            return

        if self.is_open:
            self.holding_count += 1
            spread, _ = self.spread(bars)
            diff = (spread - self.mean) / self.dev
            if self.holding_count >= self.maximum_holding_period:
                print('holding too long force quit')
                self.close_position_new(bars, ignore_diff=True)
                self.holding_count = 0
                return

        if self.mean:
            diff = (spread - self.mean) / self.dev
            # print(f"current diff {diff}")

        adf_center = adfuller(self.spread_array, regression='c')[1]
        adf_trend = adfuller(self.spread_array, regression='ct')[1]

        if adf_center < self.pValue:
            self.is_stationary = True
        else:
            self.is_stationary = False

        # if adf_trend < 0.05:
        #     print(f'adf ct test result {adf_trend}, has a trend, not enter trading')
        #     return

        # if p_value <= self.pValue:
        if self.is_stationary:
            self.hedge(bars)
        elif self.is_open:
            self.close_position_new(bars)

    def hedge(self, bars: dict):
        if not self.is_open:
            # self.open_position(bars)
            self.open_position_new(bars)
        else:
            self.close_position_new(bars)

    def open_position_new(self, bars: dict):
        spread, _ = self.spread(bars)
        self.mean, self.dev = self.get_diff(spread)
        diff = (spread - self.mean) / self.dev
        long_leg_price = (bars[self.long_leg].close)
        short_leg_price = (bars[self.short_leg].close)
        time = bars[self.short_leg].time
        short_leg_qty = self.grid_amount/(long_leg_price*abs(self.beta) + short_leg_price)
        long_leg_qty = short_leg_qty*abs(self.beta)
        if abs(diff) > 2:
            print(
                f'open parameter beta: {self.beta}, intercept: {self.intercept}, spread mean:{self.mean}, spread dev {self.dev}, diff:{diff} , time:{time}')
            self.open_diff = diff
            if self.beta < 0:

                if diff > 0:

                    self.broker.short(
                        price=long_leg_price,
                        symbol=self.long_leg,
                        vol=long_leg_qty,
                        leverage=1,
                        time=time)

                    self.broker.short(
                        price=short_leg_price,
                        symbol=self.short_leg,
                        vol=short_leg_qty,
                        leverage=1,
                        time=time)

                    self.is_open = True

                else:
                    self.broker.long(
                        price=long_leg_price,
                        symbol=self.long_leg,
                        vol=long_leg_qty,
                        leverage=1,
                        time=time)

                    self.broker.long(
                        price=short_leg_price,
                        symbol=self.short_leg,
                        vol=short_leg_qty,
                        leverage=1,
                        time=time)

                    self.is_open = True

            else:
                if diff > 0:
                    self.broker.long(
                        price=long_leg_price,
                        symbol=self.long_leg,
                        vol=long_leg_qty,
                        leverage=1,
                        time=time)

                    self.broker.short(
                        price=short_leg_price,
                        symbol=self.short_leg,
                        vol=short_leg_qty,
                        leverage=1,
                        time=time)

                    self.is_open = True

                else:
                    self.broker.long(
                        price=short_leg_price,
                        symbol=self.short_leg,
                        vol=short_leg_qty,
                        leverage=1,
                        time=time)

                    self.broker.short(
                        price=long_leg_price,
                        symbol=self.long_leg,
                        vol=long_leg_qty,
                        leverage=1,
                        time=time)

                    self.is_open = True

    # def open_position(self, bars: dict):
    #     # self.beta, self.intercept = self.get_beta()
    #     if self.beta < 0:
    #         return
    #
    #     _, diff = self.spread(bars)
    #     long_leg_price = bars[self.long_leg].close
    #     short_leg_price = bars[self.short_leg].close
    #
    #     # long_leg_qty = self.grid_amount / bars[self.long_leg].close
    #     short_leg_qty = self.grid_amount / short_leg_price
    #     long_leg_qty = (self.grid_amount / short_leg_price) * self.beta
    #     # print(f"open spread difference {diff}")
    #     if abs(diff) > self.open_margin:
    #         print(f'open parameter: {self.beta}')
    #         if diff > 0:  # short_leg > long_leg, long_leg*beta = short_leg
    #             self.broker.buy(long_leg_price, self.long_leg, long_leg_qty)
    #             self.broker.sell(short_leg_price, self.short_leg, short_leg_qty)
    #             self.is_open = True
    #
    #         else:
    #             self.broker.buy(short_leg_price, self.short_leg, short_leg_qty)
    #             self.broker.sell(long_leg_price, self.long_leg, long_leg_qty)
    #             self.is_open = True

    def close_position_new(self, bars: dict, ignore_diff=False):
        spread, _ = self.spread(bars)
        diff = (spread - self.mean) / self.dev
        # calculate position pnl
        long_leg_price = (bars[self.long_leg].close)
        short_leg_price = (bars[self.short_leg].close)
        time = bars[self.short_leg].time
        long_leg_pos: Position = self.broker.positions[self.long_leg]
        short_leg_pos: Position = self.broker.positions[self.short_leg]


        # Close the position when the unrealize loss is too much !!!
        # profit_rate = self.broker.get_current_pos_profit_rate(bars)
        # if profit_rate < -0.1:
        #     print('position profit rate is too low, force quit')
        #     ignore_diff = True
        if (self.open_diff > 0 and diff < 0) or (self.open_diff < 0 and diff > 0) or ignore_diff:

        # if abs(diff) < -2 or ignore_diff:
            self.holding_count = 0
            print(
                f'close parameter beta: {self.beta}, intercept: {self.intercept}, spread mean:{self.mean}, spread dev {self.dev}, diff:{diff} , time:{time}')

            if short_leg_pos.direction == 'short':
                self.broker.cover_short(
                    price=short_leg_price,
                    symbol=self.short_leg,
                    vol=short_leg_pos.volume,
                    time=time)
            else:
                self.broker.cover_long(
                    price=short_leg_price,
                    symbol=self.short_leg,
                    vol=short_leg_pos.volume,
                    time=time)

            if long_leg_pos.direction == 'short':
                self.broker.cover_short(
                    price=long_leg_price,
                    symbol=self.long_leg,
                    vol=long_leg_pos.volume,
                    time=time)
            else:
                self.broker.cover_long(
                    price=long_leg_price,
                    symbol=self.long_leg,
                    vol=long_leg_pos.volume,
                    time=time)

            self.is_open = False
            self.beta, self.intercept = None, None

    # def close_position(self, bars: dict, ignore_diff=False):
    #     _, diff = self.spread(bars)
    #     # print(f"close spread difference {diff}")
    #     if self.holding_count >= self.maximum_holding_period:
    #         print("holding too long force quit")
    #         self.beta = None
    #         self.intercept = None
    #     if abs(diff) <= self.close_margin or ignore_diff or self.holding_count >= self.maximum_holding_period:
    #         print(f'close parameter: {self.beta}')
    #         self.holding_count = 0
    #         long_leg_price = bars[self.long_leg].close
    #         short_leg_price = bars[self.short_leg].close
    #         long_leg_pos = self.broker.positions[self.long_leg]
    #         short_leg_pos = self.broker.positions[self.short_leg]
    #         if self.broker.positions[self.short_leg] > 0:
    #             self.broker.sell(short_leg_price, self.short_leg, short_leg_pos)
    #         else:
    #             self.broker.buy(short_leg_price, self.short_leg, abs(short_leg_pos))
    #
    #         if self.broker.positions[self.long_leg] > 0:
    #             self.broker.sell(long_leg_price, self.long_leg, long_leg_pos)
    #         else:
    #             self.broker.buy(long_leg_price, self.long_leg, abs(long_leg_pos))
    #         self.is_open = False
    #         self.beta, self.intercept = None, None

    def spread(self, bars: dict):
        long: BarData = bars[self.long_leg]
        short: BarData = bars[self.short_leg]

        long_price = (long.close)
        short_price = (short.close)

        spread = short_price - (self.beta * long_price + self.intercept)
        diff = spread / (self.beta * long_price + self.intercept)
        return spread, diff

    def get_diff(self, spread):
        mean = np.mean(self.spread_array)
        dev = np.std(self.spread_array)
        return mean, dev

    def coin_test(self):
        long_legs = []
        short_legs = []
        for i in range(self.array_len):
            bar = self.bar_array[i]
            long_legs.append(bar[self.long_leg])
            short_legs.append(bar[self.short_leg])

        if not self.beta:
            self.beta, self.intercept = self.get_beta_GPs()

        long_legs_modify = []

        for i in range(self.array_len):
            new_long_leg = long_legs[i] * self.beta + self.intercept
            long_legs_modify.append(new_long_leg)

        _, p_value, _ = coint(long_legs_modify, short_legs)
        return p_value


def generate_combinations(symbols):
    return list(combinations(symbols, 2))


p_value = 0.99
stop_loss = 0.5


def main(selected_symbols):
    cash = 1000000
    transactionFeeRate = 0
    start = '2023-01-01'
    end = '2024-10-31'
    broker = Broker(cash, transactionFeeRate, selected_symbols)
    setting = {
        'short_leg': selected_symbols[0],
        'long_leg': selected_symbols[1],
        'open_margin': 0.05,
        'close_margin': 0.02,
        'stop_loss': stop_loss,
        'pValue': 0.05,
        'grid_amount': cash * 0.2,
        'array_len': 300,
        'maximum_holding_period': 300,
    }

    strategy = Cointegration(broker, setting)
    log = Log(broker)
    backtestor = BacktestorPairTrading(broker, strategy, log, start, end)
    backtestor.load_data()
    backtestor.run_backtest()
    result = backtestor.performance()
    msg_df = pd.DataFrame(broker.msg)
    profit, loss, winrate = analyze_arbitrage_cycles(msg_df)
    # msg_df.to_csv('backTestResult/msg_df.csv')
    pnl_ratio = profit / loss
    result['pnl_ratio'] = pnl_ratio
    result['win_rate'] = winrate
    return result
    # msg_df = pd.DataFrame(broker.msg)
    # msg_df.to_csv('backTestResult/msg_df.csv')


def main2(selected_symbols):
    cash = 1000000
    transactionFeeRate = 0
    start = '2023-01-01'
    end = '2024-10-31'
    broker = Broker(cash, transactionFeeRate, selected_symbols)
    setting = {
        'short_leg': selected_symbols[0],
        'long_leg': selected_symbols[1],
        'open_margin': 0.05,
        'close_margin': 0.02,
        'stop_loss': stop_loss,
        'pValue': 0.1,
        'grid_amount': cash * 0.2,
        'array_len': 150,
        'maximum_holding_period': 300,
    }

    strategy = Cointegration(broker, setting)
    log = Log(broker)
    backtestor = BacktestorPairTrading(broker, strategy, log, start, end)
    backtestor.load_data()
    backtestor.run_backtest()
    result = backtestor.performance()
    msg_df = pd.DataFrame(broker.msg)

    profit, loss, winrate = analyze_arbitrage_cycles(msg_df)
    # msg_df.to_csv('backTestResult/msg_df.csv')
    pnl_ratio = profit / loss
    result['pnl_ratio'] = pnl_ratio
    result['win_rate'] = winrate
    return result


def main3(selected_symbols):
    cash = 1000000
    transactionFeeRate = 0
    start = '2023-01-01'
    end = '2024-10-31'
    broker = Broker(cash, transactionFeeRate, selected_symbols)
    setting = {
        'short_leg': selected_symbols[0],
        'long_leg': selected_symbols[1],
        'open_margin': 0.05,
        'close_margin': 0.02,
        'stop_loss': stop_loss,
        'pValue': 1.1,
        'grid_amount': cash * 0.2,
        'array_len': 150,
        'maximum_holding_period': 300,
    }

    strategy = Cointegration(broker, setting)
    log = Log(broker)
    backtestor = BacktestorPairTrading(broker, strategy, log, start, end)
    backtestor.load_data()
    backtestor.run_backtest()
    result = backtestor.performance()
    msg_df = pd.DataFrame(broker.msg)
    profit, loss, winrate = analyze_arbitrage_cycles(msg_df)
    # msg_df.to_csv('backTestResult/msg_df.csv')
    pnl_ratio = profit / loss
    result['pnl_ratio'] = pnl_ratio
    result['win_rate'] = winrate
    return result


if __name__ == '__main__':
    from backTestResult.cointegration_analysis import analyze_arbitrage_cycles

    cash = 1000000
    transactionFeeRate = 0.00025
    start = '2023-01-01'
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
        'maximum_holding_period': 24* 30,
    }

    strategy = Cointegration(broker, setting)
    log = Log(broker)
    backtestor = BacktestorPairTrading(broker, strategy, log, start, end)
    backtestor.load_data()
    backtestor.run_backtest()
    result = backtestor.performance()
    msg_df = pd.DataFrame(broker.msg)
    analyze_arbitrage_cycles(msg_df)
    msg_df.to_csv('backTestResult/msg_df.csv')

    ####    runing mulriple symbols experiment ####
    ####################################

    # from multiprocessing import Pool
    #
    #
    # symbols = ['ETH', 'DOGE', 'DOT', 'AVAX', 'ADA', 'IOTA', 'LINK',
    #            'TRX', 'XRP', 'UNI', 'SOL', 'LTC', 'EOS', 'BCH', 'DASH', 'LRC',
    #            'MKR','XLM','AAVE','BNB']
    #
    # symbol_combinations = generate_combinations(symbols)
    #
    # results = []
    #
    # with Pool(11) as pool:  # 使用 10 个进程
    #     # 提交任务并传入符号组合
    #     async_results = [pool.apply_async(main, (comb,)) for comb in symbol_combinations]
    #
    #     # 收集每个任务的返回值
    #     for r in async_results:
    #         results.append(r.get())
    #
    #     pool.close()
    #     pool.join()
    #
    # # 将结果转换为 DataFrame 并保存为 CSV
    # result_df = pd.DataFrame(results)
    # result_df.to_csv(f'backTestResult/experiment/ADF_result_0.05_with_stop_loss{stop_loss}and_pos_stop_loss.csv', index=False)
    #
    #
    # results = []
    #
    # with Pool(11) as pool:  # 使用 10 个进程
    #     # 提交任务并传入符号组合
    #     async_results = [pool.apply_async(main2, (comb,)) for comb in symbol_combinations]
    #
    #     # 收集每个任务的返回值
    #     for r in async_results:
    #         results.append(r.get())
    #
    #     pool.close()
    #     pool.join()
    #
    # # 将结果转换为 DataFrame 并保存为 CSV
    # result_df = pd.DataFrame(results)
    # result_df.to_csv(f'backTestResult/experiment/ADF_result_0.15_with_stop_loss{stop_loss}and_pos_stop_loss.csv', index=False)
    #
    # results = []
    #
    # with Pool(11) as pool:  # 使用 10 个进程
    #     # 提交任务并传入符号组合
    #     async_results = [pool.apply_async(main3, (comb,)) for comb in symbol_combinations]
    #
    #     # 收集每个任务的返回值
    #     for r in async_results:
    #         results.append(r.get())
    #
    #     pool.close()
    #     pool.join()
    #
    # # 将结果转换为 DataFrame 并保存为 CSV
    # result_df = pd.DataFrame(results)
    # result_df.to_csv(f'backTestResult/experiment/ADF_result_1_with_stop_loss{stop_loss}and_pos_stop_loss.csv', index=False)
