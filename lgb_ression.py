# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2022/11/2 11:27
# License: bupt

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from 第三届江苏大数据开发与应用大赛_时序预测.tree.trainer import Trainer


def get_lag_rolling_feature(data, roll_columns, lags, periods, id_col, agg_funs, feature_cols=[]):
    # stat between (lag, lag+period）
    for lag in lags:
        for agg_fun in agg_funs:
            for roll_column in roll_columns:
                for period in periods:
                    feature_col = roll_column + '_lag{}_roll{}_{}_{}'.format(lag, period, agg_fun, period)
                    feature_cols.append(feature_col)
                    if id_col is not None:
                        data[feature_col] = data.groupby(id_col)[roll_column].transform(
                            lambda x: x.shift(lag + 1).rolling(period).agg(agg_fun))
                    else:
                        data[feature_col] = data[roll_column].shift(lag + 1).rolling(period).agg(agg_fun)
    return data


def get_lag_feature(data, lag_columns, lags, feature_cols=[]):
    for col in lag_columns:
        for lag in lags:
            feature_col = col + '_lag{}'.format(lag)
            feature_cols.append(feature_col)
            data[feature_col] = data[col].shift(lag)
    return data


def get_rolling_feature(data, roll_columns, periods, agg_funs=['mean'], feature_cols=[], prefix=None):
    for col in roll_columns:
        for period in periods:
            for agg_fun in agg_funs:
                if prefix is None:
                    prefix = col
                feature_col = prefix + '_roll{}_'.format(period) + str(agg_fun)
                feature_cols.append(feature_col)
                data[feature_col] = data[col].transform(lambda x: x.rolling(period).agg(agg_fun))
    return data


df = pd.read_csv('./data/contact_all.csv')
# df['时间'] = pd.to_datetime(df['时间'])
# df['Hour'] = df['时间'].dt.hour
# df['Minute'] = df['时间'].dt.minute
# df['Second'] = df['时间'].dt.second
# df['MinuteofDay'] = df['Hour'] * 60 + df['Minute']
# df['SecondofDay'] = df['Hour'] * 60 + df['Minute'] * 60 + df['Second']
# #
# df['min_sin'] = np.sin(df['Minute'] / 60 * 2 * np.pi)
# df['min_cos'] = np.cos(df['Minute'] / 60 * 2 * np.pi)
# df['hour_sin'] = np.sin(df['Hour'] / 24 * 2 * np.pi)
# df['hour_cos'] = np.cos(df['Hour'] / 24 * 2 * np.pi)

# roll_columns = ['给水流量', '炉排实际运行指令', '引风机转速', '二次风量']
roll_columns = [i for i in df.columns if i not in [
    '时间', '推料器启停', '推料器自动投退信号', '炉排启停', '炉排自动投退信号', '主蒸汽流量',
    '主蒸汽流量设定值', '氧量设定值'
]]
feature_cols = []
#
# df = get_lag_rolling_feature(df, roll_columns=roll_columns, periods=[60, 120, 300, 600], lags=[60, 300, 600, 1800], id_col=None,
#                              agg_funs=['mean', 'std', 'max', 'min'], feature_cols=feature_cols)

# df = get_lag_feature(df, lag_columns=roll_columns,  lags=[600, 1800, 3600], feature_cols=feature_cols)
#
# df = get_rolling_feature(df, roll_columns=roll_columns, periods=[60, 1800], agg_funs=['mean', 'max', 'min', 'std', 'median'], feature_cols=feature_cols)


LABEL = '主蒸汽流量'

df_train = df[df[LABEL].notna()].reset_index(drop=True)
df_test = df[df[LABEL].isna()].reset_index(drop=True)
# df_test['ID'] = range(1, 1801)
# print(df_test['ID'].value_counts())
feats = [f for f in df_test if f not in [LABEL, '时间']]
print(feats)
print(df_train[feats].shape, df_test[feats].shape)
lgb_params = {
    'objective': 'rmse',
    'boosting_type': 'gbdt',
    'subsample': 0.8,
    'subsample_freq': 1,
    'learning_rate': 0.01,
    'n_estimators': 8000,
    'num_leaves': 2 ** 11 - 1,
    'min_data_in_leaf': 2 ** 12 - 1,
    'bagging_fraction': 0.8,
    'feature_fraction': 1.0,
    'seed': 2022,
    'max_depth': 5,
}

fold_num = 10
seeds = [2222]
oof = np.zeros(len(df_train))
importance = 0
pred_y = pd.DataFrame()
score = []
for seed in seeds:
    # kf = StratifiedKFold(n_splits=fold_num, shuffle=True, random_state=seed)
    # kf = KFold(n_splits=fold_num, shuffle=True, random_state=seed)

    fit_params = {
        'eval_metric': ['rmse'],
        # 'sample_weight': weight_train,
        # 'eval_sample_weight': [weight_valid],
        'verbose': 200,
        # 'early_stopping_rounds': 100,
    }
    model = LGBMRegressor(**lgb_params)
    trainer = Trainer(model)
    x_train = df.loc[df[0: 172800].index.tolist(), feats]
    y_train = df.loc[df[0: 172800].index.tolist(), LABEL]
    x_valid = df.loc[df[172800: 257400].index.tolist(), feats]
    y_valid = df.loc[df[172800: 257400].index.tolist(), LABEL]
    trainer.train(x_train, y_train, x_valid, y_valid, categorical_feature=None, fit_params=fit_params,
                  importance_method='auto')
    y_valid_pred = trainer.predict(df_test[feats])
    df_test['Steam_flow'] = y_valid_pred
    df_test['ID'] = range(1, 1801)
    df_test.rename(columns={'时间': 'Time'}, inplace=True)
    df_test = df_test[['ID', 'Time', 'Steam_flow']]
    # df_test = df_test.loc[:, ['ID', 'Time', 'Steam_flow']]
    df_test.to_csv('./data/test_prediction.csv', index=False)
