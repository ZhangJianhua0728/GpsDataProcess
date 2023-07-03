#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2023/05/05 16:06:08
@Author      :Zhang Jianhua
@version      :1.0
'''
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rc('font',family='Times New Roman')

class Clustering:
    """
    @description  : 创建一个用于数据聚类的类
    ---------
    """
    def __init__(self, data: pd.DataFrame, max_cluster: int=10, 
                 calc_best_cluster_flag: bool=True, best_cluster: int=2):
        """
        @description  : 聚类的类的初始函数
        ---------
        @param data  : 需要聚类的数据集
        @param max_cluster  : 最大聚类个数,用于循环遍历获取最优聚类个数
        @param calc_best_cluster_flag  : 判定是否需要计算最优聚类数,如果不需要计算最优聚类数,则需要传入best_cluster参数
        @param best_cluster  : 手动设置最优聚类数
        """
        self.data = data
        self.max_cluster = max_cluster
        # 选择聚类个数范围
        self.n_clusters_range = range(2, self.max_cluster+1)
        self.calc_best_cluster_flag = calc_best_cluster_flag
        # 计算轮廓系数
        if self.calc_best_cluster_flag:
            self.silhouette_scores: list = []
            self._calc_silhouette_scores()
            # 选择最佳聚类数
            self.best_n_clusters = np.argmax(self.silhouette_scores) + 2
            print('Best number of clusters:', self.best_n_clusters)
        else:
            self.best_n_clusters = best_cluster
        # 计算最优聚类的分类标签
        self.best_kmeans = KMeans(n_clusters=self.best_n_clusters, random_state=42)
       
        self.best_kmeans.fit(self.data)
        self.best_labels = self.best_kmeans.labels_
    
    def _calc_silhouette_scores(self):
        """
        @description  : 计算每一个聚类个数下的平均轮廓系数, 并返回平均轮廓系数列表
        ---------
        """
        for n_clusters in self.n_clusters_range:
            print('执行聚类，聚类个数为{}'.format(n_clusters))
            # 初始化kmeans算法
            kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(self.data)
            # 聚类
            labels = kmeans.labels_
            # 计算平均轮廓系数
            silhouette_avg = silhouette_score(self.data, labels)
            self.silhouette_scores.append(silhouette_avg)
            print(f'For n_clusters = {n_clusters}, the average silhouette_score is : {silhouette_avg}')
    def show_silhouette_clusters(self):
        """
        @description  : 绘制聚类个数与平均轮廓系数的关系图
        ---------
        """
        fig,ax = plt.subplots(figsize=(4,3))
        plt.plot([i for i in range(self.max_cluster+1)], [0].extend(self.silhouette_scores),'-o')
        plt.xlabel('Number of Clusters', fontsize=12)
        plt.ylabel('Average Silhouette Score', fontsize=12)
        plt.xticks([i for i in range(self.max_cluster+1)],fontsize=10.5)
        plt.yticks(fontsize=10.5)
        plt.grid(linestyle='--',linewidth=0.5)
        plt.tight_layout()
