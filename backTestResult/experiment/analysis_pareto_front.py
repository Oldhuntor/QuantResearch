import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

metric1 = 'Cumulative Return'
metric2 = 'Max Drawdown'

# metric1 = 'pnl_ratio'
# metric2 = 'win_rate'
# dataPoint01 = pd.read_csv('coint_resul_0.01.csv')
# dataPoint05 = pd.read_csv('coint_resul_0.05.csv')
# dataPoint1 = pd.read_csv('backup/coint_result_0.1_with_stop_loss.csv')
# data1 = pd.read_csv('coint_result_without_coint_test.csv')
# data1 = pd.read_csv('backup/coint_result_1_with_stop_loss.csv')
# data_b = pd.read_csv('coint_result_0.05_with_beta_adjusted.csv')
# data_b1 = pd.read_csv('coint_result_0.1_with_stop_loss.csv')
# us = pd.read_csv('coint_result_us_stocks_p0.05_o0.04_c0.03.csv')
# us2 = pd.read_csv('coint_result_us_stocks_p0.05_o0.05_c0.02.csv')
# dataPoint05sl = pd.read_csv('backup/coint_result_0.05_with_stop_loss.csv')
# columns = ['Cumulative Return', 'Max Drawdown']

A = pd.read_csv('coint_result_0.05.csv')
B = pd.read_csv('coint_result_0.15.csv')
C = pd.read_csv('coint_result_1.csv')
D = pd.read_csv('ADF_result_0.05_with_stop_loss0.5and_pos_stop_loss.csv')
E = pd.read_csv('ADF_result_0.15_with_stop_loss0.5and_pos_stop_loss.csv')
F = pd.read_csv('ADF_result_1_with_stop_loss0.5and_pos_stop_loss.csv')
G = pd.read_csv('slides/coint_result_0.99_with_stop_loss_us_stocks(bayesianRegression).csv')
H = pd.read_csv('slides/coint_result_0.99_with_stop_loss_us_stocks(GAM2).csv')
I = pd.read_csv('slides/coint_result_0.99_with_stop_loss_us_stocks(GPs2).csv')
J = pd.read_csv('slides/coint_result_0.99_with_stop_loss_us_stocks(BR2).csv')


#
# dataPoint05 = dataPoint05[columns]
# dataPoint1 = dataPoint1[columns]
# data1 = data1[columns]

# # 合并所有数据
# combined_df = pd.concat([dataPoint05, dataPoint1, data1], ignore_index=True)
#
# # 绘制 Pareto 前沿图
# plt.figure(figsize=(10, 6))
# plt.scatter(combined_df['Max Drawdown'], combined_df['Cumulative Return'], label='All Strategies', color='b')
#
# # 寻找帕累托前沿的点
# pareto_front = combined_df.loc[
#     combined_df.apply(lambda x: not any(
#         (combined_df['Cumulative Return'] > x['Cumulative Return']) &
#         (combined_df['Max Drawdown'] < x['Max Drawdown'])
#     ), axis=1)
# ]
#
# # 绘制帕累托前沿
# plt.scatter(pareto_front['Max Drawdown'], pareto_front['Cumulative Return'], color='r', label='Pareto Front')
# plt.xlabel('Max Drawdown')
# plt.ylabel('Cumulative Return')
# plt.title('Pareto Front Analysis')
# plt.legend()
# plt.grid(True)
# plt.show()


# 计算并绘制每个数据集的帕累托前沿

def plot_pareto_front(df, label, color, edge_color):
    # 寻找帕累托前沿的点
    pareto_front = df.loc[
        df.apply(lambda x: not any(
            (df[metric1] > x[metric1]) &
            (df[metric2] > x[metric2])
        ), axis=1)
    ]

    # 绘制每个数据集的散点图和帕累托前沿，确保颜色一致chi
    # plt.scatter(df[metric2], df[metric1], color=color,alpha=0.5)
    plt.scatter(pareto_front[metric2], pareto_front[metric1], color=color, edgecolors=edge_color,
                label=f'{label}', s=100)
    return pareto_front

plt.rcParams.update({'font.size': 20})  # Global font size

plt.figure(figsize=(12, 8))

# 绘制每个数据集的帕累托前沿，确保使用一致的颜色
# pareto_front_1 = plot_pareto_front(dataPoint05, 'P value = 0.05 (0.05)', color='skyblue', edge_color='red')
# pareto_front_2 = plot_pareto_front(dataPoint1, 'P value = 0.1 ', color='skyblue', edge_color='black')
# pareto_front_3 = plot_pareto_front(data1, 'P value = 1 (Without Test)', color='lightgreen', edge_color='blue')
# pareto_front_4 = plot_pareto_front(dataPoint01, 'P value = 0.01 (0.01)', color='red', edge_color='black')
# pareto_front_5 = plot_pareto_front(data_b, 'P value = 0.05 with beta adjusted', color='orange', edge_color='yellow')
# pareto_front_5 = plot_pareto_front(data_b1, 'P value = 0.1 with beta adjusted', color='blue', edge_color='black')
# pareto_front_6 = plot_pareto_front(us2, 'P value = 0.05 with beta adjusted (US stocks)', color='blue', edge_color='black')
# pareto_front_7 = plot_pareto_front(dataPoint05sl, 'P value = 0.05 ', color='green', edge_color='black')
# pareto_front_8 = plot_pareto_front(data1, 'P value = 1 with stop loss 0.4', color='orange', edge_color='black')
# pareto_front_C = plot_pareto_front(C, 'P value = 1', color='green', edge_color='black')
# pareto_front_B = plot_pareto_front(B, 'P value = 0.15 ', color='orange', edge_color='black')
# pareto_front_A = plot_pareto_front(A, 'P value = 0.05 (COINT)', color='blue', edge_color='black')

# pareto_front_F = plot_pareto_front(F, 'P value = 1 (ADF) ', color='lightgreen', edge_color='black')

# pareto_front_E = plot_pareto_front(E, 'P value = 0.15 (ADF) ', color='lightblue', edge_color='black')

# pareto_front_D = plot_pareto_front(D, 'P value = 0.05 (ADF) ', color='orange', edge_color='black')

# pareto_front_G = plot_pareto_front(G, 'BR ', color='orange', edge_color='black')
pareto_front_H = plot_pareto_front(H, 'GAM ', color='lightblue', edge_color='black')
pareto_front_I = plot_pareto_front(I, 'GPs ', color='green', edge_color='black')
pareto_front_J = plot_pareto_front(J, 'BRR ', color='pink', edge_color='black')



# 设置图表属性
plt.xlabel(metric2)
plt.ylabel(metric1)
plt.title('Pareto Front Analysis')
plt.legend()
plt.grid(True)
plt.show()