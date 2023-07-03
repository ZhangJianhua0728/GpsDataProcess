import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rc('font',family='Times New Roman')

# 横向控制安全评价
# 纵向加速的绝对值超过2.5m/s^2为不安全 DTV_acc_Lat
def calc_DTV_acc_lat(data, thre_a=2.5, frame=1):
    DTV_acc_lat = len(data[data['acc_lat'].abs() > 2.5])*frame
    DTV_acc_lat_ratio = DTV_acc_lat(len(data) * frame)
    return DTV_acc_lat, DTV_acc_lat_ratio

# 纵向控制安全评价
# 计算纵向违背数据值(Data Threshold Violations，DTV): 超过限定阑值的时长
# 速度超过80为不安全 DTV_V:
def calc_DTV_v(data, thre_v=80, frame=1):
    DTV_v = len(data[data['v'] > thre_v])*frame
    DTV_v_ratio = DTV_v/(len(data)*frame)
    return DTV_V, DTV_v_ratio