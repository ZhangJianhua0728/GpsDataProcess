import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import norm


sns.set(style='white')
plt.rc('font',family='Times New Roman')
plt.rcParams['axes.formatter.use_mathtext'] = False

def show_acc_lat(data):
    tem = data.iloc[5:-3,:]
    # Create figure and axes objects
    fig, ax1 = plt.subplots(figsize=(10,4),dpi=600)
    ax2 = ax1.twinx()
    # Plot data on primary axis
    ax1.plot(tem['long'], tem['lat'], 'b-', label='Latitude')
    ax1.scatter(tem['long'], tem['lat'],s=10, c='b')
    ax1.set_xlabel('Longitude', fontsize=12)
    ax1.set_ylabel('Latitude', fontsize=12)
    if tem['long'].iloc[-1]>tem['long'].iloc[0]:
        direction = 'LR'
    else:
        direction = 'RL'
    # Plot data on secondary axis
    ax2.plot(tem['long'], tem['acc_lat'], 'r-', label='Lateral Acceleration')
    ax2.scatter(tem['long'], tem['acc_lat'], s=10, c='r')
    ax2.axhline(y=0, color='#D5ADC7', linestyle='--', linewidth=1)
    ax2.set_ylabel('Lateral Acceleration', fontsize=12)
    # Add legend to the plot
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, fontsize=10)
    plt.tight_layout()
    if not os.path.exists('Figure'):
        os.makedirs('Figure')
    plt.savefig('./Figure/{}_行程{}_横向加速度_{}.png'.format(tem['phone'].iloc[0],tem['trip_num'].iloc[0],direction), dpi=600)


def gaussian_fit(df, column_name):
    print(data[column_name].describe())
    x = df[column_name].values
    mu, std = norm.fit(x)
    plt.hist(x, bins=500, density=True, alpha=0.6, color='g', label='Data')
    sns.kdeplot(df[column_name], shade=False, label='Density')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xlim((-5,5))
    plt.legend()
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    csvfile = '/Users/zhangjianhua/Desktop/明觉:公路院项目/A_Code_MingJueProject/gps数据处理/14462543665_行程指标.csv'
    data = pd.read_csv(csvfile)
    # data.groupby('trip_num').apply(show_acc_lat)
    data = data[(data['acc_lat']>-15)&(data['acc_lat']<15)]
    gaussian_fit(data, column_name='acc_lat')

