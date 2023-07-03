import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')


# 横向控制安全评价
## 纵向加速的绝对值超过2.5m/s^2为不安全 DTV_acc_lat
def calc_DTV_acc_lat(data, thre_a=2.5, frame=1):
    DTV_acc_lat = len(data[data['acc_lat'].abs() > 2.5])*frame
    DTV_acc_lat_ratio = DTV_acc_lat / (len(data) * frame)
    return DTV_acc_lat, DTV_acc_lat_ratio


# 纵向控制安全评价
# 计算纵向违背数据阈值(Data Threshold Violations, DTV)：超过限定阈值的时长
# 速度超过80为不安全 DTV_v：
def calc_DTV_v(data, thre_v=80, frame=1):
    DTV_v = len(data[data['v'] > thre_v])*frame
    DTV_v_ratio = DTV_v/(len(data)*frame)
    return DTV_v, DTV_v_ratio

# 纵向控制：带极限的相平面分析(PPAL)，横轴为纵向加速度，纵轴为jerk;将PPAL划分为3个风险等级，分别对应的不同阈值
"""
                    PPAL1               PPAL2               PPAL3
    thre_acc_lon    0.892 m/s^2         1.561 m/s^2         2.230 m/s^2
    thre_dec_lon    -1.383 m/s^2        -2.305 m/s^2        -3.227 m/s^2
    thre_jerk       1 m/s^3             2 m/s^3             3 m/s^3

    纵向加速度->x轴, jerk->y轴, 构建PPAL多层级椭圆
    椭圆x轴的半径为 r_a_i=0.5(a_tl_i-d_tl_i), y轴的半径为 r_j_i = 0.5(j_tl_i-(-j_tl_i))=j_tl_i
    椭圆的中心点为(C_a_i,C_j_i), C_a_i=a_tl_i-r_a_i, C_j_i=j_tl_i-r_j_i=0
"""
thre_a = np.array([0.892, 1.561, 2.23])
thre_d = np.array([-1.383, -2.305, -3.227])
thre_jerk = np.array([1, 2, 3])
radius_a = 0.5*(thre_a-thre_d)
radius_j = thre_jerk
C_a = thre_a-radius_a
C_j = thre_jerk-radius_j

## 计算旋转后位置
def calc_rot_loc(x, y, theta, cx, cy, rx, ry):
    theta_rad = np.deg2rad(theta)
    loc = ((x-cx)*np.cos(theta_rad)+(y-cy)*np.sin(theta_rad))**2/rx**2 + \
        (-(x-cx)*np.sin(theta_rad)+(y-cy)*np.cos(theta_rad))**2/ry**2
    return loc

## 获取最优旋转角度
def get_opt_theta(data, cx=C_a[0], cy=C_j[0], rx=radius_a[0], ry=radius_j[0]):
    theta = np.array([i for i in range(1, 180, 1)])
    # 计算角度phi,使更多数据小于PPL_1
    rot_angle_inn_point = []
    for i in range(len(theta)):
        inner_point = data.apply(lambda df: 1 if calc_rot_loc(
            df['acc_lon'], df['jerk'], theta[i], cx, cy, rx, ry) <= 1 else 0, axis=1)
        rot_angle_inn_point.append([theta[i], inner_point.sum()])
        print("{}:[{},{}]".format(i, theta[i], inner_point.sum()))
    max_val = max(rot_angle_inn_point, key=lambda x: x[1])
    opt_theta = max_val[0]
    print('最佳旋转角度为{}'.format(opt_theta))
    return opt_theta

## 计算PPAL
def cls_PPAL(x,y,opt_theta,cxs,cys,rxs,rys):
    # 分别获取PPAL1, PPAL2, PPAL3对应的点
    if calc_rot_loc(x,y,opt_theta,cxs[0],cys[0],rxs[0],rys[0])<=1:
        PPAL = 0
    elif (calc_rot_loc(x,y,opt_theta,cxs[0],cys[0],rxs[0],rys[0])>1) and (calc_rot_loc(x,y,opt_theta,cxs[1],cys[1],rxs[1],rys[1])<=1):
        PPAL = 1
    elif (calc_rot_loc(x,y,opt_theta,cxs[1],cys[1],rxs[1],rys[1])>1) and (calc_rot_loc(x,y,opt_theta,cxs[2],cys[2],rxs[2],rys[2])<=1):
        PPAL = 2
    else:
        PPAL = 3
    return PPAL

def calc_PPAL(data, opt_theta):
    data['PPAL'] = data.apply(lambda df: cls_PPAL(df['acc_lon'],df['jerk'],opt_theta, C_a, C_j, radius_a, radius_j),axis=1)
    PPAL1 = data[data['PPAL']==1]['PPAL'].count()
    PPAL2 = data[data['PPAL']==2]['PPAL'].count()
    PPAL3 = data[data['PPAL']==3]['PPAL'].count()
    return PPAL1, PPAL2, PPAL3

def calc_DTR_lg(alpha,DTV_v,PPAL1,PPAL2,PPAL3,total_mileage):
    """
    alpha=[alpha_v,alpha_PPAL1,alpha_PPAL2,alpha_PPAL3]
    """
    DTR_lg = (alpha[0]*DTV_v+alpha[1]*PPAL1+alpha[2]*PPAL2+alpha[3]*PPAL3)/ total_mileage
    return DTR_lg

def ellipse(cx,cy,rx,ry,angle):
    angle = np.deg2rad(angle)
    t = np.arange(0,2*np.pi,0.01)
    x = np.cos(t)*rx
    y = np.sin(t)*ry
    xx = np.cos(angle)*x - np.sin(angle)*y+cx
    yy = np.sin(angle)*x + np.cos(angle)*y+cy
    return xx, yy

## 显示PPAL
def show_PPAL(data,opt_theta):
    fig, ax = plt.subplots(figsize=(6,4))
    # 创建一个椭圆对象，设置中心点、长轴、短轴和旋转角度等参数
    xx1,yy1 = ellipse(C_a[0], C_j[0],radius_a[0],radius_j[0],angle=opt_theta)
    xx2,yy2 = ellipse(C_a[1], C_j[1],radius_a[1],radius_j[1],angle=opt_theta)
    xx3,yy3 = ellipse(C_a[2], C_j[2],radius_a[2],radius_j[2],angle=opt_theta)
    plt.plot(xx1,yy1,color='#224b8f',label='PPAL1')
    plt.plot(xx2,yy2,color='#b7ba6b',label='PPAL2')
    plt.plot(xx3,yy3,color='#f15b6c',label='PPAL3')
    # 提取绘图数据
    show_data = pd.concat([data[data['PPAL']==0].sample(n=70), data[data['PPAL']==1].sample(n=20), data[data['PPAL']==2].sample(n=10), data[data['PPAL']==3].sample(n=2)])
    # 设置不同类别的颜色和形状
    colors = ['#121a2a', '#224b8f', '#b7ba6b', '#f15b6c']
    markers = ['o', 's', '^','*']
    # 绘制散点图
    for i, group in show_data.groupby('PPAL'):
        plt.scatter(group['acc_lon'], group['jerk'], color=colors[i], marker=markers[i])
    # 添加图例，并设置图例的位置和显示方式
    ax.legend(loc='upper right', frameon=False)
    ax.set_xlabel('Acceleration', fontsize=12)
    ax.set_ylabel('Jerk', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)
    plt.tight_layout()
    plt.grid()
    plt.axis('equal')# 调整显示的横纵轴比例
    plt.savefig('./{}_PPAL_demo.png'.format(data['phone'].iloc[0]),dpi=1200)
    plt.show()

def main(data):
    # 通过所有数据计算最优旋转角度，这里只使用了一个驾驶员的数据，后期需要调整
    # opt_theta = get_opt_theta(data)
    opt_theta = 139
    # 计算每个驾驶员的DTV_v, DTV_v_ratio
    DTV_v, DTV_v_ratio = calc_DTV_v(data, thre_v=80)
    # 计算每个驾驶员的DTV_acc_lat, DTV_acc_lat_ratio
    DTV_acc_lat, DTV_acc_lat_ratio = calc_DTV_acc_lat(data,thre_a=2.5)
    # 计算每个驾驶员的 PPAL1, PPAL2, PPAL3
    PPAL1, PPAL2, PPAL3 = calc_PPAL(data,opt_theta)
    # 计算每个驾驶员的总行驶里程
    total_mileage = data.groupby('trip_num')['total_mileage'].mean().sum()
    # 记录每个驾驶员的DTV_PPAL对应的指标，以及驾驶员ID和总的行驶里程
    DTV_PPAL_res = []
    DTV_PPAL_res.append([data['phone'][0],DTV_v, DTV_v_ratio, DTV_acc_lat, DTV_acc_lat_ratio, PPAL1, PPAL2, PPAL3, total_mileage])
    columns = ['phone','DTV_v','DTV_v_ratio', 'DTV_acc_lat', 'DTV_acc_lat_ratio', 'PPAL1', 'PPAL2', 'PPAL3', 'total_mileage']
    DTV_PPAL_res_df = pd.DataFrame(DTV_PPAL_res,columns=columns)
    DTV_PPAL_res_df['DTR_lg'] = DTV_PPAL_res_df.apply(lambda df: calc_DTR_lg([1,2,3,5],df['DTV_v'],df['PPAL1'],df['PPAL2'],df['PPAL3'],df['total_mileage']),axis=1)
    # 显示PPAL图
    show_PPAL(data,opt_theta)
    print(DTV_PPAL_res_df)
     

if __name__ == '__main__':
    data = pd.read_csv('/Users/zhangjianhua/Desktop/明觉:公路院项目/A_Code_MingJueProject/gps数据处理/14462543665_行程指标.csv')
    main(data)