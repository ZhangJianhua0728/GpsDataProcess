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
            print('执行聚类,聚类个数为}'.format(n_clusters))
            # 初始化kmeans算法
            kmeans = KMeans(n_clusters=n_clusters,random_state=0).fit(self.X)
            #聚类
            labels = kmeans.labels
            # 计算平均轮廓系数
            silhouette_avg = silhouette_score(self.X,labels)
            self.silhouette_scores.append(silhouette_avg)
            print(f'For n_clusters = in_clustersh, the average silhouette_score is : isilhouette_avgr')
    
    def show_silhouette_clusters(self):
        # 绘制聚类数与平均轮廓系数关系图
        plt.figure(figsize=(4,3))
        plt.plot(self.n_clusters_range,self.silhouette_scores,'-o')
        plt.xlabel('Number of Clusters', fontsize=12)
        plt.ylabel('Average Silhoette Score', fontsize=12)
        plt.xticks(fontsize=10.5)
        plt.yticks(fontsize=10.5)
        plt.tight_layout()
        
    def show_silhouette_plot(self):
        silhouette_values = silhouette_samples(self.X,self.best_labels)
        # 绘制聚类最大的轮廓图
        fig,ax = plt.subplots(figsize=(4,3))
        y_lower, y_upper = 0,0
        for i in range(self.best_n_clusters):
        # 计算每个族的平均轮廓系数
            cluster_silhouette_values = silhouette_values[self.best_labels == i]
            cluster_silhouette_values.sort()
            cluster_size = cluster_silhouette_values.shape[o]
            y_upper += cluster_size
            color = plt.cm.get_cmap('Spectral')(float(i)/self.best_n_clusters)
            ax.barh(range(y_lower,y_upper),cluster_silhouette_values,height=1.0,color=color)
            y_lower += cluster_size
        # 添加轮廓系数的水平线
        best_silhouette_avg = silhouette_score(self.X,self.best_labels)
        ax.axvline(x=best_silhouette_avg,color='red',linestyle='--',linewidth=1)
        # 设置坐标轴、标题等
        ax.set_xlabel('Silhouette Coefficients',fontsize = 12 )
        ax.set_ylabel('Cluster Labels',fontsize = 12 )
        ax.tick_params(axis='x',labelsize=10.5)
        ax.tick_params(axis='y',labelsize=10.5)
        ax.set_yticks([])
        ax.set_xticks([-0.1,0,0.2,0.4,0.6,0.8,1])
        plt.tight_layout()

def show_class_result(PCs,best_labels,pca,feature_num=7):
    plt.figure(figsize=(4,3))
    # 可视化分类结果
    plt.scatter(PCs.iloc[:,0],PCs.iloc[:,1],c=best_labels,s=10)
    # plt.title('PCA + KMeans clustering')
    # 绘制原始特征的向量线
    features = np.eye(feature_num)
    feature_vectors = pca.transform(features)
    for i,(x,y) in enumerate(feature_vectors[:,[0,2]]):
        if i in [2,3,4,5,6]:
            plt.arrow(0,0,x*6,y*6,color='k',head_width=0.05, head_length=0.1,length_includes_head=True,zorder=100)
            plt.text(x*6, y*6, f'{i+1}', fontsize=10, color='r')
    # 设置x轴和y轴的坐标范围
    plt.xlim(-5,7.5)
    plt.ylim(-5,7.5)
    plt.xlabel('PCA1',fontsize=12)plt.ylabel('PCA2',fontsize=12)
    plt.xticks(fontsize=10.5)
    plt.yticks(fontsize=10.5)
    plt.tight_layout()