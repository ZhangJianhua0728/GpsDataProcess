import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.rc('font',family='Times New Roman')

def show_class_result(data:pd.DataFrame, PC_list:list, label:str, pca, num_cls:int, feature_num:int, arrow_text_list:list, xy_labels:list, 
                      cls_list:list, cls_legend:list, cls_colors = ['#bb505d','#7fb80e','#4e72b8'], vector_scale:float = 1):
    """绘制降维后的散点图
    Args:
        data (pd.DataFrame): 需要绘制散点的数据集
        PC_list (list): 散点图x,y轴对应数据data中的列名
        label (str): 散点图分类标签对应数据data中的列名
        pca (_type_): 降维散点图中所使用的pca模型
        num_cls (_int_): 散点分类的个数
        feature_num (int): 降维前的数据原始特征数
        arrow_text_list (list): 原始特征的标签
        xy_labels (list): 绘图坐标轴的标题
        cls_list (list): 分类标签的列表
        cls_legend (list): 绘图中所使用的分类图例
        cls_colors (list, optional): 散点分类所使用的颜色. Defaults to ['#bb505d','#7fb80e','#4e72b8'].
        vector_scale (float, optional): 矢量图缩放比例. Defaults to 1.
    """    
    fig,ax = plt.subplots(figsize=(4,3))

    # 绘制原点的水平线和垂直线
    ax.axhline(y=0, color='#3e4145', linestyle='--', linewidth=0.8)
    ax.axvline(x=0, color='#3e4145', linestyle='--', linewidth=0.8)
    
    # 可视化分类结果
    # 按照label列分组，将不同的标签对应的数据分别绘制在散点图上，并使用不同颜色表示
    marker_list = ['o', '^', 'x', 'D']
    for i in range(num_cls):
        x = data.loc[data[label] == cls_list[i], PC_list[0]]
        y = data.loc[data[label] == cls_list[i], PC_list[1]]
        plt.scatter(x, y, c=cls_colors[i], label=cls_legend[i], marker=marker_list[i], s=1, alpha=0.5)
    
    # 绘制原始特征的向量线
    features = np.eye(feature_num)
    feature_vectors = pca.transform(features)
    for i, (x, y) in enumerate(feature_vectors[:,[0,1]]):
        plt.arrow(0, 0, x*feature_num*vector_scale, y*feature_num*vector_scale, color='#130c0e',width=0.01, head_width=0.3, 
                head_length=0.3, length_includes_head=True, zorder=100)
        plt.text(x*feature_num*vector_scale, y*feature_num*vector_scale, arrow_text_list[i], fontsize=10, color='k')
    
    # 设置轴标题
    plt.xlabel(xy_labels[0], fontsize=12)
    plt.ylabel(xy_labels[1], fontsize=12)
    

    plt.xlim([-7.5,20])
    plt.ylim([-5,10])
    
    # 设置轴刻度属性
    plt.xticks(fontsize=10.5)
    plt.yticks(fontsize=10.5)
    
    # 设置图例
    plt.legend(title='Clusters', fontsize=10.5, loc='upper right')
    plt.tight_layout()
    