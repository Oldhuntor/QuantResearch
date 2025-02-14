import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read data
# br_data = pd.read_csv('coint_result_0.99_with_stop_loss_us_stocks(bayesianRegression).csv')
gam_data = pd.read_csv('coint_result_0.99_with_stop_loss_us_stocks(GAM2).csv')
gps_data = pd.read_csv('coint_result_0.99_with_stop_loss_us_stocks(GPs2).csv')
br_data = pd.read_csv('coint_result_0.99_with_stop_loss_us_stocks(BR2).csv')

metrics = ['Cumulative Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown']

# Function to calculate statistics
def get_stats(data, metric):
    return {
        'min': data[metric].min(),
        'q25': data[metric].quantile(0.25),
        'median': data[metric].quantile(0.5),
        'q75': data[metric].quantile(0.75),
        'max': data[metric].max()
    }

# Calculate statistics for each model and metric
results = {}
for metric in metrics:
    results[metric] = {
        'BR': get_stats(br_data, metric),
        'GAM': get_stats(gam_data, metric),
        'GPs': get_stats(gps_data, metric),
        # 'BRR': get_stats(brr_data, metric),
    }

# Print results
for metric in metrics:
    print(f"\n{metric}:")
    for model in ['BR', 'GAM', 'GPs']:
        print(f"{model}:", results[metric][model])

# Plotting
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Model Comparison - Box Plots')

for i, metric in enumerate(metrics):
    ax = axes[i//2, i%2]
    data = [
        br_data[metric],
        gam_data[metric],
        gps_data[metric]
    ]
    ax.boxplot(data, labels=['BR', 'GAM', 'GPs'])
    ax.set_title(metric)
    ax.grid(True)

plt.tight_layout()
plt.show()