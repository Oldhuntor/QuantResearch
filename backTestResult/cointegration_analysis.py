import pandas as pd
import numpy as np


def analyze_arbitrage_cycles(df: pd.DataFrame) :
    """
    Analyze arbitrage cycles where each cycle consists of 4 trades.
    """
    results = []

    # Process trades in groups of 4
    for i in range(0, len(df), 4):
        cycle = df.iloc[i:i + 4]
        if len(cycle) < 4:  # Skip incomplete cycle
            break

        cycle_pnl = 0
        trades = []

        # Track positions for this cycle
        positions = {}

        # Analyze each trade in the cycle
        for _, trade in cycle.iterrows():
            symbol = trade['symbol']

            trade_pnl = 0
            if trade['action'] in ['long', 'short']:
                positions[symbol] = {
                    'entry_price': trade['price'],
                    'volume': trade['vol'],
                    'action': trade['action']
                }
            else:  # Cover trade
                position = positions[symbol]
                if position['action'] == 'long':
                    trade_pnl = (trade['price'] - position['entry_price']) * position['volume']
                else:  # short
                    trade_pnl = (position['entry_price'] - trade['price']) * position['volume']

                trades.append({
                    'symbol': symbol,
                    'type': position['action'],
                    'entry': position['entry_price'],
                    'exit': trade['price'],
                    'volume': position['volume'],
                    'pnl': trade_pnl
                })

                cycle_pnl += trade_pnl

        results.append({
            'cycle_number': i // 4,
            'start_index': i,
            'end_index': i + 3,
            'trades': trades,
            'total_pnl': cycle_pnl
        })

        # Print detailed analysis
    print("Arbitrage Cycles Analysis:")
    print("=" * 80)

    total_pnl = 0
    profit = 0
    loss = 0
    for cycle in results:
        print(f"\nCycle {cycle['cycle_number']} (Trades {cycle['start_index']}-{cycle['end_index']}):")
        print("-" * 40)

        for trade in cycle['trades']:
            print(f"{trade['symbol']} {trade['type'].upper()}")
            print(f"Entry: ${trade['entry']:.4f} | Exit: ${trade['exit']:.4f}")
            print(f"Volume: {trade['volume']:.2f}")
            print(f"PnL: ${trade['pnl']:,.2f}")
            print("-" * 20)
            trade_pnl = trade['pnl']
            if trade_pnl < 0:
                loss += -trade_pnl
            elif trade_pnl > 0:
                profit += trade_pnl


        print(f"Cycle Total PnL: ${cycle['total_pnl']:,.2f}")
        print("=" * 40)
        total_pnl += cycle['total_pnl']

    # Calculate summary statistics
    pnl_array = [cycle['total_pnl'] for cycle in results]
    print("\nOverall Summary:")
    print("=" * 80)
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Average PnL per cycle: ${np.mean(pnl_array):,.2f}")
    print(f"Best cycle: ${max(pnl_array):,.2f}")
    print(f"Worst cycle: ${min(pnl_array):,.2f}")
    print(f"Profitable cycles: {sum(pnl > 0 for pnl in pnl_array)} out of {len(results)}")
    print(f"Win rate: {(sum(pnl > 0 for pnl in pnl_array) / len(results) * 100):.1f}%")
    winrate = sum(pnl > 0 for pnl in pnl_array) / len(results) * 100
    return profit, loss, winrate

if __name__ == '__main__':

    df = pd.read_csv('msg_df.csv')

    # Process the data
    df = pd.DataFrame({
        'symbol': df['symbol'],
        'price': df['price'].astype(float),
        'vol': df['vol'].astype(float),
        'action': df['action']
    })

    # Calculate results
    profit, loss, winrate = analyze_arbitrage_cycles(df)
    print(profit, loss)
    # Print detailed analysis
    # print("Arbitrage Cycles Analysis:")
    # print("=" * 80)
    #
    # total_pnl = 0
    # for cycle in cycles:
    #     print(f"\nCycle {cycle['cycle_number']} (Trades {cycle['start_index']}-{cycle['end_index']}):")
    #     print("-" * 40)
    #
    #     for trade in cycle['trades']:
    #         print(f"{trade['symbol']} {trade['type'].upper()}")
    #         print(f"Entry: ${trade['entry']:.4f} | Exit: ${trade['exit']:.4f}")
    #         print(f"Volume: {trade['volume']:.2f}")
    #         print(f"PnL: ${trade['pnl']:,.2f}")
    #         print("-" * 20)
    #
    #     print(f"Cycle Total PnL: ${cycle['total_pnl']:,.2f}")
    #     print("=" * 40)
    #     total_pnl += cycle['total_pnl']
    #
    # # Calculate summary statistics
    # pnl_array = [cycle['total_pnl'] for cycle in cycles]
    # print("\nOverall Summary:")
    # print("=" * 80)
    # print(f"Total PnL: ${total_pnl:,.2f}")
    # print(f"Average PnL per cycle: ${np.mean(pnl_array):,.2f}")
    # print(f"Best cycle: ${max(pnl_array):,.2f}")
    # print(f"Worst cycle: ${min(pnl_array):,.2f}")
    # print(f"Profitable cycles: {sum(pnl > 0 for pnl in pnl_array)} out of {len(cycles)}")
    # print(f"Win rate: {(sum(pnl > 0 for pnl in pnl_array) / len(cycles) * 100):.1f}%")
