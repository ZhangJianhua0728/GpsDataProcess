import os
import pandas as pd

def get_csvfile_name(folder_path):
    all_csvfile_name=[]
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            csvfile_name = filename.split('.')[0]
            all_csvfile_name.append(csvfile_name)
    return all_csvfile_name

def save_multidf_to_xlsx(df_dict,out_filename='output.xlsx',index_flag=False):
    # 创建一个ExcelWriter对象
    writer = pd.ExcelWriter(out_filename, engine='xlsxwriter')
    for key, df in df_dict.items():  
        df.to_excel(writer, sheet_name=key, index=index_flag)
    # 保存Excel文件
    writer.save()
