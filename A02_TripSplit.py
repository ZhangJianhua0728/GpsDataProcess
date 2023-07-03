import pandas as pd
import numpy as np
import copy


def delete_stop_data(data):
    """
    :param data: gps数据
    :return: 删除gps记录的速度为0以及周边3秒的数据,这类数据可能是车辆未启动、gps信号中断
    """
    # 找到 speed=0 的行
    zero_speed = data[data['v'] == 0].index
    # 找到上下 3 行的索引
    zero_index = []
    for i in zero_speed:
        zero_index.extend(range(i-3, i+3))
    # 去重，排序
    zero_index = np.array(sorted(list(set(zero_index))))
    # 删除，超过指定索引的值
    zero_index = zero_index[zero_index <= data.index[-1]]
    new_data = data.loc[~data.index.isin(zero_index)]
    new_data = new_data.reset_index(drop=True)
    return new_data


def get_trip_SE(data, T_thre=60):
    """
    获取行程起终点
    """
    data['time'] = pd.to_datetime(data['time'])
    trip_SE = pd.DataFrame()
    trip_SE['F_time_diff'] = (
        data['time']-data['time'].shift(1)).dt.total_seconds()
    trip_SE['F_time_diff'] = trip_SE['F_time_diff'].fillna(1)
    trip_SE['R_time_diff'] = (
        data['time'].shift(-1)-data['time']).dt.total_seconds()
    trip_SE['R_time_diff'] = trip_SE['R_time_diff'].fillna(1)
    start_index = np.array(trip_SE[trip_SE['F_time_diff'] != 1].index)
    start_index = np.sort(np.append(start_index, trip_SE.index[0]))
    end_index = np.array(trip_SE[trip_SE['R_time_diff'] != 1].index)
    end_index = np.sort(np.append(end_index, trip_SE.index[-1]))
    trip_index_df = pd.DataFrame(np.array([start_index, end_index]).T, columns=[
                                 'start_point', 'end_point'])
    trip_index_df['duration'] = trip_index_df['end_point'] - \
        trip_index_df['start_point']+1
    return trip_index_df[trip_index_df['duration'] > T_thre]


def get_trip_data(data, trip_index):
    all_trip = []
    for i in range(len(trip_index)):
        trip = copy.deepcopy(data[trip_index.iloc[i, 0]:trip_index.iloc[i, 1]])
        trip.loc[:, 'trip_num'] = i+1
        trip.loc[:, 'total_mileage'] = trip['mileage'].iloc[-1] - \
            trip['mileage'].iloc[0]
        all_trip.append(trip)
    new_data = pd.concat(all_trip)
    new_data = new_data.reset_index(drop=True)
    return new_data


def main(data):
    data = delete_stop_data(data)
    trip_index = get_trip_SE(data)
    trip_data = get_trip_data(data, trip_index)
    trip_data.to_csv(
        './data_demo_new/{}_行程数据.csv'.format(trip_data['phone'].iloc[0]), index=False)


if __name__ == '__main__':
    data = pd.read_csv(
        'data_demo_new/14462543665_预处理.csv')
    main(data)
