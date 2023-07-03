import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')


df = pd.read_csv('14462543665_行程指标.CSv')
def show_DTV_V(df):
    # 绘制折线图
    plt.figure(figsize=(4,3))
    plt.plot(df.index[6000:9000], df['v'].iloc[6000:9000], c='#5516BF', Linewidth=1, label='Velocity')
    plt.axhline(y=80,c='#C66BA0', linewidth=2, linestyle='--', label='Threshold')
    plt.xticks([6000, 7000, 8000, 9000],[0, 1000, 2000, 3000], fontsize=10.5)
    plt.yticks(fontsize=10.5)
    plt.xlabel('Times(s)', fontsize=12)
    plt.xlabel('Velocity(m/s)', fontsize=12)
    plt.leqend()
    plt.tight_layout()
    # 显示图像
    plt.savefig('./速度DTV.png',dpi=1200)
    plt.show()