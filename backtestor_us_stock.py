from backtest_multi_symbols import Backtestor, Broker, PortfolioManagement, Log, select_random_symbols
import pandas as pd
from multiprocessing import Pool

symbols = pd.read_csv('data/sp500_symbol.csv')['Symbol'].tolist()
symbols_to_remove = ['BRK.B','BF.B','TAP','ERIE','VLTO','AMTM','CPAY','SW','TPL','AZO','GEV','MTD','DOC','DAY','SOLV','BEN','SMCI','KLAC','QRVO','VMC','ORLY','POOL','COST','AJG','ROP']
for s in symbols_to_remove:
    symbols.remove(s)

class BacktestUSstock(Backtestor):
    def __init__(self, broker: Broker, strategy: PortfolioManagement, log: Log, start, end):
        # 继承父类的初始化方法
        super().__init__(broker, strategy, log, start, end)
        self.start = pd.to_datetime(self.start, utc=True)
        self.end = pd.to_datetime(self.end, utc=True)

    def load_data(self):
        for symbol in self.symbols:
            # 修改为读取新格式的 CSV 文件
            df = pd.read_csv(f'data/sp500/{symbol}.csv')

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

            # 保存处理后的数据
            self.data[symbol] = df
            self.data_length = len(df)

# 示例用法：
# broker = Broker(...)
# strategy = PortfolioManagement(...)
# log = Log(...)
# backtest = BacktestUSstock(broker, strategy, log, start='2023-01-01', end='2023-12-31')
# backtest.load_data()


def main():
    cash = 1000000
    selected_symbols = select_random_symbols(symbols, 4)

    transactionFeeRate = 0
    period = "1d"
    window_size = 10
    start = '2023-10-01'
    end = '2024-10-31'

    broker = Broker(cash, transactionFeeRate, selected_symbols)
    strategy = PortfolioManagement(broker, period, window_size)
    log = Log(broker)

    backtestor = BacktestUSstock(broker, strategy, log, start, end)
    backtestor.load_data()
    backtestor.run_backtest()
    result = backtestor.performance()
    return result



if __name__ == '__main__':

    main()

    # result = []
    #
    # with Pool(1) as pool:
    #     # 提交 100 个异步任务到进程池
    #     async_results = [pool.apply_async(main) for _ in range(300)]
    #
    #     # 使用 .get() 方法收集每个任务的返回值
    #     for r in async_results:
    #         result.append(r.get())
    #
    #     pool.close()
    #     pool.join()
    #
    # # 将结果转换为 DataFrame 并保存为 CSV
    # result_df = pd.DataFrame(result)
    # result_df.to_csv('backTestResult/USstock/result.csv', index=False)
