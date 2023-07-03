import pandas as pd
import numpy as np
from sklearn.cluster import KMeans


def cls_acc(value, bins):
    idx = np.searchsorted(bins, value, side='right')
    if idx == 0:
        loc = '-inf~-5.5'
    elif idx == len(bins):
        loc = '5.5~inf'
    else:
        loc = '{}~{}'.format(bins[idx-1], bins[idx])
    return loc


def calc_entropy(data, cls_name):
    # 统计类别列中每个元素出现的次数
    counts = data[cls_name].value_counts()
    # 计算每个元素出现的概率
    probs = counts / counts.sum()
    # 计算信息熵
    entropy = -(probs * np.log2(probs)).sum()
    # 输出信息熵
    print(entropy)
    return entropy


def cls_entropy(data, x_cols):
    kmeans = KMeans(n_clusters=3)
    y_pred = kmeans.fit_predict(data[x_cols])
    return y_pred


def main(data):
    bins = [-5.5, -5, -4.5, -4, -3.5, -3, -2.5, -
            2, -1, 0, 1, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5]
    # 删除横向加速度大于15m/s^2的数据
    data = data[data['acc_lat'] < 15]

    data['acc_lon_cls'] = data.apply(
        lambda df: cls_acc(df['acc_lon'], bins), axis=1)
    acc_lon_entropy = calc_entropy(data[data['acc_lon'] >= 0], 'acc_lon_cls')
    dec_lon_entropy = calc_entropy(data[data['acc_lon'] < 0], 'acc_lon_cls')
    data['acc_lat_cls'] = data.apply(
        lambda df: cls_acc(df['acc_lat'], bins), axis=1)
    acc_lat_entropy = calc_entropy(data[data['acc_lat'] >= 0], 'acc_lat_cls')
    dec_lat_entropy = calc_entropy(data[data['acc_lat'] < 0], 'acc_lat_cls')
    entropy = [acc_lon_entropy, dec_lon_entropy,
               acc_lat_entropy, dec_lat_entropy]
    print(entropy)


if __name__ == "__main__":
    data = pd.read_csv('14462543665_行程指标.csv')
    main(data)
