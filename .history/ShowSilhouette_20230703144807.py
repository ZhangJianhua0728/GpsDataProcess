
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import silhouette_score, silhouette_samples

plt.rc('font',family='Times New Roman')


def show_silhouette_plot(data,x_cols: list, y_col: str, num_cls: int, cls_list: list, 
                         colors: list=['#8a2e3b', '#b2d235', '#dea32c', '#6950a1']):
    # 计算 Silhouette 分数和样本轮廓系数
    silhouette_avg = silhouette_score(data[x_cols], data[y_col])
    sample_silhouette_values = silhouette_samples(data[x_cols], data[y_col])
    # 绘制 Silhouette 图
    fig, ax = plt.subplots(figsize=(4, 3))
    y_lower, y_upper = 0, 0
    for i in range(num_cls):
        ith_cluster_silhouette_values = sample_silhouette_values[data[y_col] == cls_list[i]] # type: ignore
        ith_cluster_silhouette_values.sort()
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper += size_cluster_i
        ax.barh(range(y_lower, y_upper), ith_cluster_silhouette_values, height=1, color=colors[i])
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cls_list[i]))
        y_lower += size_cluster_i

    ax.axvline(x=silhouette_avg, color="#2468a2", linestyle="--", linewidth=0.8)
    # 设置坐标轴、标题等
    ax.set_xlabel('Silhouette value',fontsize = 12 )
    ax.set_ylabel('Clusters',fontsize = 12 )
    # 设置 x 轴和 y 轴的刻度标签字体
    ax.tick_params(axis='x', labelsize=10.5)
    ax.tick_params(axis='y', labelsize=10.5)
    ax.set_yticks([])
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.tight_layout()

    # 打印每个簇的平均轮廓系数
    print("The average silhouette score of all samples: ", silhouette_avg)
    for i in range(num_cls):
        ith_cluster_silhouette_values = sample_silhouette_values[data[y_col] == cls_list[i]]
        print("The average silhouette score of cluster {} : {:.2f}".format(i, np.mean(ith_cluster_silhouette_values)))