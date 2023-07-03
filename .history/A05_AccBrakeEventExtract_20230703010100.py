import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


class EventExtract:
    def __init__(self, data) -> None:
        self.gps = data
        pass
    def acc_lon_event(self):
        # 提取gps中的速度低于15km/h，纵向加速度大于0的数据
        df = self.gps[(self.gps['v'] >= 15) & (self.gps['acc_lon'] > 0)]
        # 将df中字段 time 转换为时间格式，并计算时间差
        grouped = self._groupby_time(df)
        # 提取纵向加速事件的指标
        result = pd.DataFrame()
        self._basic_indicators(grouped, result)
        self._speed_indicators(grouped, result)
        self._acc_lon_indicators(grouped, result)
        self._jerk_indicators(grouped, result)
        # 确保每个分组的最大v和最小v的差值不小于5，返回纵向加速事件指标结果
        v_diff = result['max_v'] - result['min_v']
        invalid_groups = v_diff < 5
        result.loc[invalid_groups, :] = np.nan
        result = result.dropna()
        result = result.reset_index(drop=True)
        return result
    
    def dec_lon_event(self):
        # 提取gps中的速度低于15km/h，纵向加速度小于0的数据
        df = self.gps[(self.gps['v'] >= 15) & (self.gps['acc_lon'] < -0.1)]
        # 将减速度转换为正值
        df.loc[:, 'dec_lon'] = np.abs(df['acc_lon']) 
        # 将df中字段 time 转换为时间格式，并计算时间差
        grouped = self._groupby_time(df)

        # 提取纵向减速事件的指标
        result = pd.DataFrame()
        self._basic_indicators(grouped, result)
        self._speed_indicators(grouped, result)
        self._dec_lon_indicators(grouped, result)
        self._jerk_indicators(grouped, result)

        # 确保每个分组的最大v和最小v的差值不小于5
        v_diff = result['max_v'] - result['min_v']
        invalid_groups = (v_diff < 5)
        result.loc[invalid_groups, :] = np.nan
        result = result.dropna()
        result = result.reset_index(drop=True)
        return result

    def _groupby_time(self, df):
        df.loc[:, 'time'] = pd.to_datetime(df['time'])
        time_diff = (df['time'] - df['time'].shift()).dt.total_seconds().fillna(0)
        groups = time_diff > 1
        group_ids = groups.cumsum()
        grouped = df.groupby(group_ids)
        return grouped
    
    def _basic_indicators(self,grouped, result):
        result['phone'] = grouped['phone'].apply(lambda x: x.iloc[0])
        result['start_time'] = grouped['time'].apply(lambda x: x.iloc[0])
        result['end_time'] = grouped['time'].apply(lambda x: x.iloc[-1])
        result['duration'] = grouped['time'].apply(lambda x: x.max() - x.min()).dt.total_seconds()+1

    def _speed_indicators(self,grouped, result):
        result['min_v'] = grouped['v'].min()
        result['max_v'] = grouped['v'].max()
        result['mean_v'] = grouped['v'].mean()
        result['std_v'] = grouped['v'].std()

    def _acc_lon_indicators(self,grouped, result):
        result['max_acc_lon'] = grouped['acc_lon'].max()
        result['mean_acc_lon'] = grouped['acc_lon'].mean()
        result['std_acc_lon'] = grouped['acc_lon'].std()
        result['min_acc_lon'] = grouped['acc_lon'].min()

    def _dec_lon_indicators(self, grouped, result):
        result['max_dec_lon'] = grouped['dec_lon'].max()
        result['mean_dec_lon'] = grouped['dec_lon'].mean()
        result['std_dec_lon'] = grouped['dec_lon'].std()
        result['min_dec_lon'] = grouped['dec_lon'].min()
    
    def _jerk_indicators(self, grouped, result):
        result['max_jerk'] = grouped['jerk'].max()
        result['mean_jerk'] = grouped['jerk'].mean()
        result['std_jerk'] = grouped['jerk'].std()
        result['min_jerk'] = grouped['jerk'].min()

if __name__ == "__main__":
    filename = 'data_demo_new/14462543665_行程指标.csv'
    data = pd.read_csv(filename)
    EveExt = EventExtract(data)
    acc_event = EveExt.acc_lon_event()
    dec_event = EveExt.dec_lon_event()
    acc_event.to_csv('data_demo_new/14462543665_纵向加速事件.csv',index=False)
    dec_event.to_csv('data_demo_new/14462543665_纵向减速事件.csv',index=False)