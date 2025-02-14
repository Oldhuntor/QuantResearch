from cointegration import *
import numpy as np


class BacktestPairTradingUSstock(BacktestorPairTrading):
    def __init__(self, broker: Broker, strategy: Cointegration, log: Log, start, end):
        # 继承父类的初始化方法
        super().__init__(broker, strategy, log, start, end)
        self.start = pd.to_datetime(self.start, utc=True)
        self.end = pd.to_datetime(self.end, utc=True)

    def load_data(self):
        for symbol in self.symbols:
            # 修改为读取新格式的 CSV 文件
            df = pd.read_csv(f'data/sp500_data/stock_data/{symbol}_hourly_20230117_20250115.csv')

            # 将 'Datetime' 列转换为 datetime 类型，并筛选数据
            df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
            df = df[(df['Datetime'] >= self.start) & (df['Datetime'] <= self.end)]

            # 将列名规范化为统一格式（如时间、开盘价、最高价、最低价、收盘价、成交量）
            df = df.rename(columns={
                'Datetime': 'timestamp',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })

            # 使用log 函数
            # df['close'] = np.log(df['close'])
            # 保存处理后的数据
            self.data[symbol] = df
            self.data_length = len(df)

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
                self.log.logging(bars)
                equity = self.broker.calculate_total_account_equity(bars)
                if (1 - equity / self.strategy.initial_balance) > self.strategy.stop_loss:
                    print(f"stop loss at {self.strategy.stop_loss} !!!")
                    break

            self.on_close(bars)
            self.log.save_logs()
        except Exception as e:
            print(f"{symbol}: {e}")


def main(selected_symbols):
    cash = 1000000
    transactionFeeRate = 0
    start = '2022-01-01'
    end = '2025-09-30'
    broker = Broker(cash, transactionFeeRate, selected_symbols)
    setting = {
        'short_leg': selected_symbols[0],
        'long_leg': selected_symbols[1],
        'open_margin': 0.05,
        'close_margin': 0.02,
        'stop_loss': 0.7,
        'pValue': p_value,
        'grid_amount': cash * 0.2,
        'array_len': 100,
        'maximum_holding_period': 24 * 30,
    }

    strategy = Cointegration(broker, setting)
    log = Log(broker)
    backtestor = BacktestPairTradingUSstock(broker, strategy, log, start, end)
    backtestor.load_data()
    backtestor.run_backtest()
    result = backtestor.performance()
    return result


if __name__ == '__main__':

    from multiprocessing import Pool

    symbols = ['BALL', 'BSX', 'CARR', 'CMS', 'ES', 'EVRG', 'HAS', 'INCY', 'KO', 'MAS', 'NDAQ', 'O', 'OXY', 'REG', 'SCHW', 'SOLV', 'TAP', 'UBER', 'VST', 'WMT', 'XEL']

    symbol_combinations = generate_combinations(symbols)

    results = []

    with Pool(9) as pool:  # 使用 10 个进程
        # 提交任务并传入符号组合
        async_results = [pool.apply_async(main, (comb,)) for comb in symbol_combinations]

        # 收集每个任务的返回值
        for r in async_results:
            results.append(r.get())

        pool.close()
        pool.join()

    # 将结果转换为 DataFrame 并保存为 CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(f'backTestResult/experiment/coint_result_{p_value}_with_stop_loss_us_stocks(GPs2).csv', index=False)
