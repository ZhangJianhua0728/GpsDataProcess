import pandas as pd

def match_vehicle_info(data,phone_vehicle, vehicle_info):
    
    # 将两个表通过id列进行关联，并将df2的age列添加到df1中
    df_merged = pd.merge(data, phone_vehicle[['接入号', '车牌']], left_on='phone', right_on='接入号', how='left')
    df_merged = df_merged.drop('接入号', axis=1)


    df_merged = pd.merge(df_merged,vehicle_info[['car_no', 'car_type']],left_on='车牌', right_on='car_no', how='left')
    df_merged = df_merged.drop('car_no', axis=1)

    # 打印结果
    print(df_merged)
    return df_merged

if __name__ == "__main__":
    acc_event = pd.read_csv('加减速事件/加速事件.csv')
    dec_event = pd.read_csv('加减速事件/减速事件.csv')
    phone_vehicle = pd.read_excel('加减速事件/车牌与企业匹配表.xlsx')
    vehicle_info = pd.read_excel('加减速事件/car_industry_info.xlsx')
    
    acc_event_merged = match_vehicle_info(acc_event,phone_vehicle,vehicle_info)
    acc_event_merged.to_csv('加减速事件/加速事件附车辆信息.csv', index=False)

    dec_event_merged = match_vehicle_info(dec_event,phone_vehicle,vehicle_info)
    dec_event_merged.to_csv('加减速事件/减速事件附车辆信息.csv',index=False)
