# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2022/11/2 19:45
# License: bupt
import pandas as pd

import pandas as pd
import numpy as np
def convert_datetime(col_series: pd.Series):
    """series datetime64[ns] 转 字符串日期"""
    if col_series.dtype == "datetime64[ns]":
        return col_series.dt.strftime('%Y-%m-%d %H:%M:%S')
    else:
        return col_series
def main():
    df = pd.read_csv('./data/contact_all.csv')
    print(df.info())
    df['时间'] = pd.to_datetime(df['时间'])
    print(df.info())
    df['时间'] = pd.Timestamp(df['时间'])
    # print(time_df.dtypes, "\n ==============")
    print(df['时间'].dtypes)
    # df.to_csv('./data/contact_all_new.csv', index=None)
if __name__ == '__main__':
    main()

# print(df.info())
# df.to_csv('./data/contact_all_new.csv', index=None)