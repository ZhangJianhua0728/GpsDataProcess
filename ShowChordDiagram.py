import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nxviz as nv
from nxviz import annotate
plt.rc('font',family='Times New Roman')


def plot_chord_diagram(data, sour_col='Node1_3', targ_col='Node2_3', size1_col='Size1', size2_col='Size2', 
                    weight_col='weight', group_col='Group_new'):
    fig = plt.figure(figsize=(4,4))
    # 创建一个空的NetworkX图
    G = nx.Graph()
    edge_list = []
    node_list = []
    # 添加节点和边到图中
    for index, row in data.iterrows():
        source = row[sour_col]
        target = row[targ_col]
        size1 = row[size1_col]
        size2 = row[size2_col]
        weight = row[weight_col]
        group = row[group_col]
        node_list.append([source,{'group':group, 'label':source, 'size':size1}])
        node_list.append([target,{'group':group, 'label':target, 'size':size2}])
        edge_list.append([source, target, {'weight':weight}])

    G.add_nodes_from(node_list)
    G.add_edges_from(edge_list)

    ax = nv.circos(G,group_by='group',node_color_by='group',edge_color_by='weight',sort_by='label')
    nv.annotate.circos_labels(G,layout='rotate',sort_by='label',fontdict={"size":8})


if __name__ =='__main__':
    data = pd.read_excel('/Users/zhangjianhua/Desktop/明觉:公路院项目/A_Code_MingJueProject/gps数据处理/加速事件方差分析/关系图分析.xlsx')
    plot_chord_diagram(data)
    plt.show()