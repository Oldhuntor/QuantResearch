import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model_new.bayesian_regression import bayesian_rolling_window
from model_new.jointCov2 import fit_HGP
from model_new.BayesGAM import bayesian_gam_with_splines
from model_new.BSTS import bsts_fit


def plot_symbols_time_series(data, symbol_list):
    """
    Plot the time series of specified symbols with improved x-axis readability.

    Parameters:
    - data: pd.DataFrame, original time series data. The first column is the symbol, and the rest are time points.
    - symbol_list: list of str, list of symbols to plot.

    Returns:
    - None. Displays the plot.
    """
    # Filter the data for the selected symbols
    selected_data = data[data.iloc[:, 0].isin(symbol_list)]

    # Plot each time series
    plt.figure(figsize=(12, 6))
    for _, row in selected_data.iterrows():
        symbol = row.iloc[0]
        time_series = row.iloc[1:].values
        plt.plot(data.columns[1:], time_series, label=symbol)

    plt.title("Time Series within clusters")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.legend(loc="upper right", fontsize="small")
    plt.grid()

    # Rotate and format x-axis labels for better readability
    plt.xticks(
        ticks=range(0, len(data.columns[1:]), max(len(data.columns[1:]) // 10, 1)),  # Show fewer ticks
        labels=[data.columns[i + 1] for i in range(0, len(data.columns[1:]), max(len(data.columns[1:]) // 10, 1))],
        rotation=45,  # Rotate the labels
        fontsize="small"
    )
    plt.tight_layout()
    plt.show()


def fitting_graphs_latent(data: dict, y_obs, window_size=False):
    '''
    :param data:
        data = {
        y : {
            mean:array,
             upper:array,
             lower:array
        },
        beta: {
            mean:array,
             upper:array,
             lower:array
        },
        mu:{
            mean:array,
             upper:array,
             lower:array
        },
        epsilon:{
            mean:array,
             upper:array,
             lower:array
        },
    }
    '''


    T = np.arange(0, len(y_obs))  # Time variable
    if window_size:
        window_size = 30
        T = np.arange(0, len(y_obs)-window_size)
    else:
        window_size = 0

    y_values = data['y']
    beta_values = data['beta']
    mu_values = data['mu']
    epsilon_values = data['epsilon']

    y_obs = y_obs[window_size:]
    y_pred_mean = y_values['mean'][window_size:]
    y_pred_upper = y_values['upper'][window_size:]
    y_pred_lower = y_values['lower'][window_size:]

    beta_posterior = beta_values['mean'][window_size:]
    beta_upper = beta_values['upper'][window_size:]
    beta_lower = beta_values['lower'][window_size:]

    mu_posterior = mu_values['mean'][window_size:]
    mu_upper = mu_values['upper'][window_size:]
    mu_lower = mu_values['lower'][window_size:]

    epsilon_mean = epsilon_values['mean'][window_size:]
    epsilon_upper = epsilon_values['upper'][window_size:]
    epsilon_lower = epsilon_values['lower'][window_size:]

    # Create subplots
    fig, axes = plt.subplots(4, 1, figsize=(20, 15))  # 4 rows, 1 column

    # Set the font sizes
    plt.rcParams['font.size'] = 16  # Base font size
    TITLE_SIZE = 24  # Title font size
    LABEL_SIZE = 20  # Axis label font size
    LEGEND_SIZE = 16  # Legend font size

    # Define a cohesive color palette
    primary_color = '#2E86AB'  # A pleasant blue
    secondary_color = '#F28F3B'  # A warm orange
    interval_color = '#C8E7FF'  # Light blue for intervals
    true_color = '#FF6B6B'  # Coral red for true values
    error_color = '#67B99A'  # Sage green for errors

    # ---- 1. Time Series ----
    axes[0].plot(T, y_obs.flatten(), label='Observed Y', color=primary_color, linewidth=2)
    axes[0].plot(T, y_pred_mean, label='Predicted Mean', color=secondary_color, linewidth=2)
    axes[0].fill_between(T, y_pred_lower, y_pred_upper, alpha=0.5, color=interval_color,
                         label='95% PI')
    axes[0].set_title('Time Series Posterior Predictive Check', fontsize=TITLE_SIZE)
    axes[0].set_xlabel('Time Step', fontsize=LABEL_SIZE)
    axes[0].set_ylabel('Y', fontsize=LABEL_SIZE)
    axes[0].legend(fontsize=LEGEND_SIZE)
    axes[0].grid(True, alpha=0.15)
    axes[0].tick_params(axis='both', labelsize=LABEL_SIZE)

    # ---- 2. Beta ----
    # axes[1].plot(T, beta_true, label="True Beta", color=true_color, linewidth=2)
    axes[1].plot(T, beta_posterior, label="Estimated Beta", color=primary_color, linewidth=2)
    axes[1].fill_between(T, beta_lower, beta_upper, alpha=0.5, color=interval_color, label='95% CI')
    axes[1].set_title("Beta Over Time", fontsize=TITLE_SIZE)
    axes[1].set_xlabel("Time Step", fontsize=LABEL_SIZE)
    axes[1].set_ylabel("Beta", fontsize=LABEL_SIZE)
    axes[1].legend(fontsize=LEGEND_SIZE)
    axes[1].grid(True, alpha=0.15)
    axes[1].tick_params(axis='both', labelsize=LABEL_SIZE)

    # ---- 3. Mu ----
    # axes[2].plot(T, mu_true, label="True Mu", color=true_color, linewidth=2)
    axes[2].plot(T, mu_posterior, label="Estimated Mu", color=primary_color, linewidth=2)
    axes[2].fill_between(T, mu_lower, mu_upper, alpha=0.5, color=interval_color, label='95% CI')
    axes[2].set_title("Mu Over Time", fontsize=TITLE_SIZE)
    axes[2].set_xlabel("Time Step", fontsize=LABEL_SIZE)
    axes[2].set_ylabel("Mu", fontsize=LABEL_SIZE)
    axes[2].legend(fontsize=LEGEND_SIZE)
    axes[2].grid(True, alpha=0.15)
    axes[2].tick_params(axis='both', labelsize=LABEL_SIZE)

    # ---- 4. Residual Errors (Epsilon) ----
    axes[3].plot(T, epsilon_mean, label="Mean Epsilon", color=error_color, linewidth=2)
    # axes[3].plot(T, epsilon_true, label='True Epsilon', color=true_color, linewidth=2)
    axes[3].fill_between(T, epsilon_lower, epsilon_upper, alpha=0.5, color=error_color,
                         label="95% PI")
    axes[3].set_title("Residual Errors (Epsilon) with 95% Prediction Interval", fontsize=TITLE_SIZE)
    axes[3].set_xlabel("Time Step", fontsize=LABEL_SIZE)
    axes[3].set_ylabel("Epsilon", fontsize=LABEL_SIZE)
    axes[3].legend(fontsize=LEGEND_SIZE)
    axes[3].grid(True, alpha=0.15)
    axes[3].tick_params(axis='both', labelsize=LABEL_SIZE)

    # Adjust layout for readability
    plt.tight_layout()
    # plt.savefig(f'{name}')
    plt.show()

if __name__ == '__main__':

    data = pd.read_csv('/Users/hxh/PycharmProjects/Quantitative_Research/model_new/bayesian_clustering/transposed_stock_data.csv')
    symbol_list = ['NI', 'PPL']
    # plot_symbols_time_series(data, symbol_list)
    selected_data = data[data.iloc[:, 0].isin(symbol_list)]

    time_series = []
    for _, row in selected_data.iterrows():
        symbol = row.iloc[0]
        time_series.append(row.iloc[1:].values)

    Xt = np.float64(time_series[0])
    Yt = np.float64(time_series[1])
    T = 300
    Xt = Xt[:T]
    Yt = Yt[:T]
    # result = bayesian_gam_with_splines(Xt, Yt)
    #
    # fitting_graphs_latent(result, Yt)

    result = bayesian_rolling_window(Xt, Yt)

    fitting_graphs_latent(result, Yt, window_size=True)

    # result = fit_HGP(Xt, Yt)
    #
    # fitting_graphs_latent(result, Yt)
    #
    # result = bsts_fit(Xt, Yt)
    #
    # fitting_graphs_latent(result, Yt)




