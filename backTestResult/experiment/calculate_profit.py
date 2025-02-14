import re

# Define the text input
text = """
open parameter beta: -8916.241331180605, intercept: 1715.1486135008704
Short TRX at price: 0.05536 for 1333050.4636516767 amount with leverage 1
Short ETH at price: 1337.72 for 149.50811829082318 amount with leverage 1
open parameter beta: -8916.241331180605, intercept: 1715.1486135008704
Cover short ETH at price: 1552.7 for 149.50811829082318 amount
Cover short TRX at price: 0.06061 for 1333050.4636516767 amount
"""

# Improved regex patterns
open_trade_pattern = r"^(Long|Short) (\w+) at price: ([\d.]+) for ([\d.]+) amount"
close_trade_pattern = r"^Cover (long|short) (\w+) at price: ([\d.]+) for ([\d.]+) amount"

# Extract open and close trades
open_trades = re.findall(open_trade_pattern, text, re.IGNORECASE | re.MULTILINE)
close_trades = re.findall(close_trade_pattern, text, re.IGNORECASE | re.MULTILINE)

# Debug: Print extracted trades
print("Open Trades:", open_trades)
print("Close Trades:", close_trades)

# Create dictionaries for quick matching
open_dict = {trade[1].lower(): trade for trade in open_trades}  # key = asset
close_dict = {trade[1].lower(): trade for trade in close_trades}  # key = asset

# Calculate profit/loss for each trade
profit_loss = []
for asset, open_trade in open_dict.items():
    action, _, open_price, amount = open_trade
    if asset in close_dict:
        _, _, close_price, _ = close_dict[asset]

        # Convert to float
        open_price = float(open_price)
        close_price = float(close_price)
        amount = float(amount)

        # Calculate P&L based on action
        if action.lower() == "long":
            pnl = (close_price - open_price) * amount
        elif action.lower() == "short":
            pnl = (open_price - close_price) * amount

        profit_loss.append({"Asset": asset, "P&L": pnl})

# Calculate total profit
total_profit = sum(p["P&L"] for p in profit_loss)

# Print the results
print("\nProfit and Loss per Asset:")
for p in profit_loss:
    print(f"{p['Asset'].upper()}: {p['P&L']:.2f}")

print(f"\nTotal Profit: {total_profit:.2f}")
