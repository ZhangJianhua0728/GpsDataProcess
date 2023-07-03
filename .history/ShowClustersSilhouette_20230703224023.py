import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.rc('font',family='Times New Roman')

data = pd.read_excel('./事件聚类/all事件_speed指标_聚类数与轮廓系数对应表.xlsx')
fig,ax = plt.subplots(figsize=(4,3))
plt.plot(data['cluster'], data['silhouette'],'-o', color='#120432',linewidth=1)
plt.plot([2,2],[0,data['silhouette'].iloc[1]],'--',color='#5E0583', linewidth=0.8)
plt.xlabel('Number of Clusters', fontsize=12)
plt.ylabel('Average Silhouette Score', fontsize=12)
plt.xticks(data['cluster'],fontsize=10.5)
plt.yticks(fontsize=10.5)
plt.grid(linestyle='--',linewidth=0.5)
plt.tight_layout()
plt.savefig('./事件聚类/all事件_speed指标_聚类数与轮廓系数对应关系.pdf')