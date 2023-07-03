#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Description:       :
@Date     :2023/05/09 15:46:36
@Author      :Zhang Jianhua
@version      :1.0
'''
from sklearn.preprocessing import StandardScaler
import pandas as pd
class Standard:
    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data
        self._Z_score()
        pass

    def _Z_score(self):
        self.scaler = StandardScaler()
        self.scaler.fit(self.data)

    def out_scaled_result(self):
        """
        @description  : 实现对dataframe数据的的标准化
        ---------
        @param  data: 需要标准化的DataFrame数据
        -------
        @Returns  scaled_data: 返回完成标准化后的DataFrame数据
        -------
        """
        scaled_data = self.scaler.transform(self.data)
        scaled_data = pd.DataFrame(scaled_data, columns=self.data.columns)
        return scaled_data