#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2023/05/05 14:18:42
@Author      : Zhangjianhua
@version      :1.0
'''
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA, FactorAnalysis
import matplotlib.pyplot as plt

plt.rc('font', family='Times New Roman')


class DimensionReduce:
    """构建一个用于数据降维的类
    """
    def __init__(self, data: pd.DataFrame, n_pc=None):
        """数据降维类的初始化函数

        Args:
            data (pd.DataFrame): 需要降维的数据
            n_pc (int, optional): 主成分个数
        """
        self.data = data
        self._PCA(n_pc)

    def _PCA(self, n_pc):
        """创建主成分分析对象，并对标准化后的数据进行主成分分析

        Args:
            n_pc (int): 主成分个数
        """
        if n_pc is None:
            self.pca = PCA()
        else:
            self.pca = PCA(n_components=n_pc)
        self.pca.fit(self.data)
        if n_pc is None:
            self.pcs = pd.DataFrame(
                self.pca.transform(self.data),
                columns=['PC'+str(i+1) for i in range(self.data.shape[1])])
        else:
            self.pcs = pd.DataFrame(
                self.pca.transform(self.data),
                columns=['PC'+str(i+1) for i in range(n_pc)])

    def calc_total_variance_explain(self) -> pd.DataFrame:
        """ 计算每个主成分的特征值、方差贡献率和累计贡献率

        Returns:
            pd.DataFrame: 返回一个总方差解释表
        """
        # 获取特征值和方差贡献率
        eigenvalues = self.pca.explained_variance_
        variance_ratios = self.pca.explained_variance_ratio_
        # 计算累计贡献率
        cumulative_variances = pd.Series(variance_ratios).cumsum()
        # 构建结果DataFrame
        total_variance_explain = pd.DataFrame({
            'Eigenvalues': eigenvalues,
            'Variance': variance_ratios,
            'Cumulative Variance': cumulative_variances
        })
        return total_variance_explain

    def calc_PC_loadings(self) -> pd.DataFrame:
        """计算各指标数据的主成分因子载荷

        Returns:
            pd.DataFrame: 返回主成分因子载荷 DataFrame

        """
        component_loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=['PC{}'.format(i+1) for i in range(self.data.shape[1])],
            index=self.data.columns)
        return component_loadings

    def calc_RC_loadings(self) -> pd.DataFrame:
        """
        @description  : 计算旋转后的载荷
        ---------
        @Returns  : 返回旋转载荷 DataFrame
        -------
        """
        fa = FactorAnalysis(
            n_components=self.data.shape[1], rotation='varimax')
        fa.fit(self.data)
        rotated_loadings = pd.DataFrame(
            fa.components_.T,
            columns=['RC{}'.format(i+1) for i in range(self.data.shape[1])],
            index=self.data.columns)
        return rotated_loadings


def show_scree_plot(pca_df: pd.DataFrame):
    """绘制主成分与解释方差的碎石图

    Args:
        pca_df (pd.DataFrame): 主成分可解释方差数据
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    # 绘制碎石图
    plt.plot(range(1, len(pca_df) + 1),
             pca_df['Variance'], '-o', linewidth=1, color='#1a2933')
    plt.xlabel('Component number', fontsize=12)
    plt.ylabel('Explained variance', fontsize=12)
    plt.xticks(range(len(pca_df) + 1), fontsize=10.5)
    plt.yticks(fontsize=10.5)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.tight_layout()


def show_correlation_plot(loading_df: pd.DataFrame, x_cols: list, text_labels: list, xy_labels: list):
    """绘制载荷与原始特征相关关系图

    Args:
        loading_df (pd.DataFrame): 载荷数据
        x_cols (list): 主成分列标签
        text_labels (list): 原始特征绘图的标签
        xy_labels (list): 绘图的xy轴标签
    """
    # 创建坐标轴
    fig, ax = plt.subplots(figsize=(4, 3))
    # 计算每个向量的长度
    lengths = np.linalg.norm(loading_df[x_cols], axis=1)
    # 绘制原点到向量的线，并在线的上方添加注释
    cmap = plt.cm.Spectral
    norm = plt.Normalize(vmin=0, vmax=1)
    scalar_map = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.quiver([0]*len(loading_df), [0]*len(loading_df), loading_df[x_cols[0]]*1.5, loading_df[x_cols[1]]*1.5, linewidth=0.8,
              angles='xy', scale_units='xy', scale=1, color=scalar_map.to_rgba(lengths), zorder=100,)
    for i in range(len(loading_df)):
        value = lengths[i]
        text_color = cmap(norm(value))
        ax.text(loading_df[x_cols[0]][i]*1.5, loading_df[x_cols[1]][i]*1.5, text_labels[i], fontsize=8,
                color=text_color, ha='center', va='center', zorder=100)
    # 创建colorbar
    plt.colorbar(scalar_map)
    # 生成单位圆上的点
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta)
    y = np.sin(theta)
    # 绘制单位圆
    ax.plot(x, y, 'k-', linewidth=1, alpha=0.5)
    # 设置坐标轴
    ax.set_aspect('equal', adjustable='box')
    # 设置坐标轴范围和标签
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel(xy_labels[0], fontsize=12)
    ax.set_ylabel(xy_labels[1], fontsize=12)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.tick_params(axis='x', labelsize=10.5)
    ax.tick_params(axis='y', labelsize=10.5)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.tight_layout()
