from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,silhouette_samples
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from A06_PCA import DimensionReduce

plt.rc('font',family='Times New Roman')

class Clustering:
    def __init__(self,data, max_cluster=10,calc_best_cluster_flag=True,best_cluster=2):
        self.X = data
        # 选择聚类个数范围
        self.n_clusters_range = range(2, max_cluster)
        # 计算轮廓系数
        if calc_best_cluster_flag:
            self.silhouette_scores = []
            self._calc_silhouette_scores()
            # 选择最佳聚类数
            self.best_n_clusters = np.argmax(self.silhouette_scores) + 2
            print('Best number of clusters:', self.best_n_clusters)
        else:
            self.best_n_clusters = best_cluster
        # 计算最优聚类的分类标签
        self.best_kmeans = KMeans(n_clusters=self.best_n_clusters, random_state=0).fit(self.X)
        self.best_labels = self.best_kmeans.labels_


    def calc_silhouette_scores(self):
        # 遍历聚类个数范围
        for n_clusters in self.n_clusters_range:
            print('执行聚类，聚类个数为}'.format(n_clusters))
            # 初始化kmeans算法
            kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(self.X)
            #聚类
            labels = kmeans.labels
            # 计算平均轮廓系数
            silhouette_avg = silhouette_score(self.X，labels)
            self.silhouette_scores.append(silhouette_avg)
            print(f'For n_clusters = in_clustersh, the average silhouette_score is : isilhouette_avgr')
    def show_silhouette_clusters(self):
        # 绘制聚类数与平均轮廓系数关系图
        plt.figure(figsize=(4,3))
        plt.plot(self.n_clusters_range,self.silhouette_scores,'-0')
        plt.xlabel('Number of Clusters'fontsize=12]
        plt.ylabel( Average Silhoette Score', fontsize=12)
        plt.xticks(fontsize=10.5)
        plt.yticks(fontsize=10.5)
        plt.tight_layout()