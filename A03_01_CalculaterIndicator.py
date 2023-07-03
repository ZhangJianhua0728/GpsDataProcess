import numpy as np
import pandas as pd
import copy


# 车辆横向控制指标: 横向加速度
## 计算方位角
def getDegree(latA, lonA, latB, lonB):
    """
    Args:
        point p1(latA, lonA)
        point p2(latB, lonB)
    Returns:
        bearing between the two GPS points,
        default: the basis of heading direction is north
    """
    radLatA = np.deg2rad(latA)
    radLonA = np.deg2rad(lonA)
    radLatB = np.deg2rad(latB)
    radLonB = np.deg2rad(lonB)
    dLon = radLonB - radLonA
    y = np.sin(dLon) * np.cos(radLatB)
    x = np.cos(radLatA) * np.sin(radLatB) - np.sin(radLatA) * np.cos(radLatB) * np.cos(dLon)
    brng = np.degrees(np.arctan2(y, x))
    brng = (brng + 360) % 360
    brng = np.deg2rad(brng)
    return brng


## 计算横向加速度
def calc_acc_lat(data):
    """
    计算横向加速度的正负值。向左为正，向右为负
    """
    tem = copy.deepcopy(data)
    # 将速度换算为m/s
    tem['v'] = tem['v'] / 3.6
    # 相对方位角计算
    # 前一个点
    tem[['lat-1', 'long-1']] = tem[['lat', 'long']].shift()
    tem['lat-1'].iloc[0] = tem['lat'].iloc[0]
    tem['long-1'].iloc[0] = tem['long'].iloc[0]
    # 前第二个点
    tem[['lat-2', 'long-2']] = tem[['lat-1', 'long-1']].shift()
    tem['lat-2'].iloc[0] = tem['lat-1'].iloc[0]
    tem['long-2'].iloc[0] = tem['long-1'].iloc[0]
    # 后一个点
    tem[['lat1', 'long1']] = tem[['lat', 'long']].shift(-1)
    tem['lat1'].iloc[-1] = tem['lat'].iloc[-1]
    tem['long1'].iloc[-1] = tem['long'].iloc[-1]
    tem['Bb'] = getDegree(tem['lat-2'], tem['long-2'], tem['lat-1'], tem['long-1'])
    tem['Baf'] = getDegree(tem['lat'], tem['long'], tem['lat1'], tem['long1'])
    tem['Bdiff'] = tem['Baf'] - tem['Bb']
    tem['Bdiff'][tem['Bdiff'] < -np.pi] = tem['Bdiff'][tem['Bdiff'] < -np.pi] + 2 * np.pi
    tem['Bdiff'][tem['Bdiff'] > np.pi] = tem['Bdiff'][tem['Bdiff'] > np.pi] - 2 * np.pi
    tem['Bdiff'][tem['Baf'] == 0] = 0
    tem['R'] = tem['v'] / tem['Bdiff']
    tem['R'] = tem['R'].fillna(0)
    tem['acc_lat'] = np.power(tem['v'], 2) / tem['R']
    tem['acc_lat'] = tem['acc_lat'].fillna(0)
    return pd.DataFrame(tem['acc_lat'])


def calc_acc_lon(data, frame=1):
    tem = copy.deepcopy(data)
    tem['v'] = tem['v'] / 3.6
    tem['next_v'] = tem['v'].shift(-1)
    tem['next_v'] = tem['next_v'].fillna(method='ffill')
    tem['acc_lon'] = (tem['next_v'] - tem['v']) / frame
    return pd.DataFrame(tem['acc_lon'])


def calc_jerk(data, frame=1):
    tem = copy.deepcopy(data)
    tem['next_acc_lon'] = tem['acc_lon'].shift(-1)
    tem['next_acc_lon'] = tem['next_acc_lon'].fillna(method='ffill')
    tem['jerk'] = (tem['next_acc_lon'] - tem['acc_lon']) / frame
    return pd.DataFrame(tem['jerk'])

 
def calc_yaw_angle(data,frame=1):
    tem = copy.deepcopy(data)
    tem['time'] = pd.to_datetime(tem['time'])
    tem['next_long'] = tem['long'].shift(-1).fillna(method='ffill')
    tem['next_lat'] = tem['lat'].shift(-1).fillna(method='ffill')
    tem['yaw_angle'] = np.arctan((tem['next_long']- tem['long'])/(tem['next_lat']-tem['lat'])*np.cos(tem['next_lat']))*(180/np.pi)
    tem['next_yaw_angle'] = tem['yaw_angle'].shift(-1).fillna(method='ffill')
    tem['yaw_rate']  = (tem['next_yaw_angle'] - tem['yaw_angle'])/frame
    return pd.DataFrame(tem['yaw_rate'])

def main(data):
    # 1. 计算纵向加速度
    data['acc_lon'] = np.array(data.groupby('trip_num').apply(calc_acc_lon))
    # 2. 计算jerk
    data['jerk'] = np.array(data.groupby('trip_num').apply(calc_jerk))
    # 3. 计算横向加速度
    data['acc_lat'] = np.array(data.groupby('trip_num').apply(calc_acc_lat))
    # 4. 计算横摆角速度
    data['yaw_rate'] = np.array(data.groupby('trip_num').apply(calc_yaw_angle))

    data.to_csv('./data_demo_new/{}_行程指标.csv'.format(data['phone'].iloc[0]), index=False)
    return data


if __name__ == '__main__':
    csvfile = 'data_demo_new/14462543665_行程数据.csv'
    data = pd.read_csv(csvfile)
    main(data)
