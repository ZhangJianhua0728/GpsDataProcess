import pandas as pd
import numpy as np


def filter_outlier(data: pd.DataFrame, columns=None):
    if columns is None:
        df = data
    else:
        df = data[columns]
    # 对于每一列，计算第一四分位数（Q1）、第三四分位数（Q3）和四分位距（IQR）
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    # 定义过滤条件，即每列数据必须在Q1-3IQR和Q3+3IQR之间
    filter = (df >= Q1 - 3 * IQR) & (df <= Q3 + 3 * IQR)
    # 删除不符合条件的行，并生成新的DataFrame
    data_filtered = data[filter.all(axis=1)].reset_index(drop=True)
    return data_filtered