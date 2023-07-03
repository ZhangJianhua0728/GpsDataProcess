import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font', family=' Times New Roman')



def get_3s_mode(data:pd.DataFrame, col_name):
   
    data['mode1'] = data[col_name]
    data['mode2'] = data[col_name].shift(-1).fillna(method='ffill')
    data['mode3'] = data[col_name].shift(-2).fillna(method='ffill')
    data['mode'] = data['mode1'].str.cat([data['mode2'], data['mode3']], sep='')
    return data

def show_mode_freq(data):
    plt.figure(figsize=(8,3))
    x = [[i+1-0.2, i+1, i+1+0.2] for i in range (30)]
    for i in range (30):
        plt.plot(x[i],data[['coli','col2' 'col3']].iloc[i], c='black' , lw=0.75)
        plt.scatter(x[i], data[['col1''col2','col3']].iloc[i], c='black' , s=5)
        plt.axhline(y=4, c=' red')
    # 是示图形
    plt.yticks(fontsize=9)
    plt.xticks(fontsize=9)
    plt.xlabel('Mode', fontsige=10.5)
    plt.ylabel('Grade', fontsize=10.5)
    plt.xlim(0.6,30.4)
    plt.ylim(0.8, 7.2)
    plt.tight_layout()