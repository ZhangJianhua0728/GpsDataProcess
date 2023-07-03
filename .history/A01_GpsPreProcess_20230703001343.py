import csv
import pandas as pd

def gps_outliers_remove(csvfile, fixed_column_count=14):
    """
    剔除导出数据中存在的异常数据行
    csvfile -> gps文件名
    fixed_column_count -> 规定固定列数
    return: 返回剔除异常值后的数据
    """
    data = []
    raw_row = 0
    with open(csvfile, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            raw_row += 1
            if len(row) == fixed_column_count:
                data.append(row)
    raw_row -= 1
    data = pd.DataFrame(data,columns=data[0])
    data = data.iloc[1:,:]
    adjusted_row, _ = data.shape
    outliers_row = raw_row-adjusted_row
    print("原始数据:{}行，剔除异常值后的数据:{},共删除异常数据:{}行".format(raw_row, adjusted_row, outliers_row))
    return data

def gps_preprocess(csvfile, fixed_column_count=14):
    # 删除gps数据中的异常行数据
    data = gps_outliers_remove(csvfile, fixed_column_count)
    # 删除时间长度不等于12的数据
    data = data[data['date_time'].str.len()==12]
    # 基于时间，删除重复行
    data.drop_duplicates(subset='date_time', keep='first', inplace=True)
    print("基于时间列，删除重复数据后，剩余:{}行".format(len(data)))
    # 基于时间，对数据进行排序
    data.sort_values('date_time', ascending=True, inplace=True)
    # 基于时间，对数据的索引进行重新编号
    data = data.reset_index(drop=True)
    # 将日期转为时间，速度/10转为浮点数，经度|纬度/1000000转为浮点数，方向转为整数，海拔高度转为整数，里程/10转为浮点数
    # 部分文件的mileage列数据存在''空字符串
    for i, row in data.iterrows():
        if row['mileage'] == '':
            if i-1>=0:
                data.loc[i, 'mileage'] = data.loc[i-1, 'mileage']
            else:
                data.loc[i, 'mileage'] = data.loc[i+1, 'mileage']
    
    data[['speed', 'longitude', 'latitude', 'direction', 'altitude', 'mileage']]=data[['speed', 'longitude', 'latitude', 'direction', 'altitude', 'mileage']].astype(int)
    data['date_time']=data['date_time'].apply(lambda t: '20'+t)
    data['date_time'] = pd.to_datetime(data['date_time'])
    data['speed'] = data['speed']/10
    data['longitude'] = data['longitude']/1000000
    data['latitude'] = data['latitude']/1000000
    data['mileage'] = data['mileage']/10
    columns = ['phone', 'date_time', 'speed', 'longitude', 'latitude', 'direction', 'altitude', 'mileage']
    re_columns = ['phone','time','v','long','lat','direct','alti','mileage']
    data = data[columns]
    data.columns = re_columns
    return data

def main(file):
    data = gps_preprocess(file, fixed_column_count=14)
    data.to_csv('./data_demo_new/{}_预处理.csv'.format(data['phone'].iloc[0]), index=False)

if __name__ == '__main__':
    csvfile ='data_demo_new/14462543665.csv'
    main(csvfile)