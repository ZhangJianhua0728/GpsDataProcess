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
# 计算纵向违背数据值(Data Threshold Violations,DTV): 超过限定阑值的时长
# 速度超过80为不安全 DTV_V:
def calc_DTV_v(data, thre_v=80, frame=1):
    DTV_v = len(data[data['v'] > thre_v])*frame
    DTV_v_ratio = DTV_v/(len(data)*frame)
    return DTV_v, DTV_v_ratio

#纵向控制:带极限的相平面分析(PPAL),横轴为纵向加速度,纵轴为jerk;将PPAL划分为了个风险等级,分别对应的不同阑值
"""
                PPAL1               PPAL2               PPAL3
thre_acc_lon    0.892 m/s^2        1.561 m/s^2          2.230 m/s^2
thre_dec_Ton    -1.383 m/s^2       -2.305 m/s^2         -3.227 m/s^2
thre_jerk       1 m/s^3            2 m/s^3              3 m/s^3
纵向加速度->X轴, jerk->轴, 构建PPAL多层级椭圆
椭圆x轴的半径为 r_a_i=0.5(a_tL_i-d_tL_i), y轴的半径为 r_j_i =0.5(i_tl_i-(-i_tl_i))=j_tl_i
椭圆的中心点为(C_a_i,C_j_i), C_a_i=a_tl_i-r_a_i, C_j_i=j_tl_i-r_j_i=0
"""
thre_a = np.array([0.892,1.561,2.23])
thre_d = np.array([-1.383,-2.305,-3.227])
thre_jerk = np.array([1,2,3])
radius_a = 0.5*(thre_a-thre_d)
radius_j = thre_jerk
C_a = thre_a-radius_a
C_j = thre_jerk-radius_j

## 计算旋转后位置
def calc_rot_loc(x,y, theta,cx,cy,rx,ry):
    theta_rad = np.deg2rad(theta)
    loc = ((x-cx)*np.cos(theta_rad)+(y-cy)*np.sin(theta_rad))**2/rx**2+(-(x-cx)*np.sin(theta_rad)+(y-cy)*np.cos(theta_rad))**2/ry**2
    return loc

## 获取最优旋转角度
def get_opt_theta(data,cx=C_a[0],cy=C_j[0],rx=radius_a[0],ry=radius_j[0]):
    theta = np.array([i for i in range(1,180,1)])
    # 计算角度phi,使更多数据小于PPL_1
    rot_angle_inn_point = []
    for i in range(len(theta)):
        inner_point = data.apply(lambda df: 1 if calc_rot_loc(df['acc_lon'],df['jerk'],theta[i],cx, cy,rx, ry) <= 1 else 0, axis=1)
        rot_angle_inn_point.append([theta[i],inner_point.sum()])
        print(":[},}]".format(i, theta[i], inner_point.sum()))
    max_val = max(rot_angle_inn_point,key=lambda x: x[1])
    opt_theta = max_val[0]
    print('最佳旋转角度为'.format(opt_theta))
    return opt_theta

## 计算PPAL
def cls_PPAL(x, Y,opt_theta, cxs,cys, rxs, rys):
    # 分别获取PPAL1,PPAL2,PPAL了对应的点
    if calc_rot_loc(x,Y,opt_theta,cxs[o],cys[o],rxs[o],rys[o])<=1:
        PPAL =0
    elif (calc_rot_loc(x,y,opt_theta,cxs[0], cys[0],rxs[0],rys[0])>1) and (calc_rot_loc(x,y,opt_theta,cxs[1],cys[1],rxs[1],rys[1])<-1):
        PPAL =1
    elif (calc_rot_loc(x,y,opt_theta, cxs[1],cys[1],rxs[1],rys[1])>1) and (calc_rot_loc(x,y,opt_theta, cxs[2],cys[2],rxs[2],rys[2])-1):
        PPAL =2
    else:
        PPAL =3
    return PPAL