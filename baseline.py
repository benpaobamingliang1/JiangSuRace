# coding=utf-8
# Author: gml <28422785281@qq.com>
# Modified by: gml
# datetime： 2022/11/3 22:34
import datetime

import matplotlib.pyplot as plt
import pandas as pd
from hyperts import make_experiment
from hyperts.framework.search_space import DLForecastSearchSpace
from hyperts.toolbox import temporal_train_test_split

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

df_train = pd.read_csv('./data/contact_train.csv')
df_test = pd.read_csv('./contact_test.csv')

df_train = df_train[['时间', '给水流量', '主蒸汽流量']]
df_test = df_test[['时间', '给水流量', '主蒸汽流量']]
# roll_columns = ['给水流量', '炉排实际运行指令', '引风机转速', '二次风量']
# roll_columns = ['给水流量']
# df_train = get_lag_rolling_feature(df_train, roll_columns=roll_columns, periods=[30, 60], lags=[60, 120], id_col=None,
#                                    agg_funs=['mean', 'max', 'min'], feature_cols=[])
train_data, valid_data = temporal_train_test_split(df_train, test_horizon=1800)

plt.figure(figsize=(16, 6))
plt.plot(pd.to_datetime(train_data['时间']), train_data['主蒸汽流量'], c='k', label='train data')
plt.plot(pd.to_datetime(valid_data['时间']), valid_data['主蒸汽流量'], c='r', label='test data')
# plt.legend()
# plt.show()

covariates = [i for i in df_train.columns if i in ['给水流量']]
print(covariates)
custom_search_space = DLForecastSearchSpace(enable_deepar=False,
                                            enable_hybirdrnn=False,
                                            enable_lstnet=False,
                                            enable_nbeats=True)
experiment = make_experiment(train_data=df_train.copy(),
                             task='forecast',
                             mode='dl',
                             timestamp='时间',
                             tf_gpu_usage_strategy=1,
                             covariates=covariates,
                             reward_metric='rmse',
                             # max_trials=1,
                             eval_data=valid_data.copy(),
                             dl_forecast_window=[60],
                             dl_forecast_horizon=7200,
                             search_space=custom_search_space,
                             early_stopping_time_limit=60,
                             # ensemble_size=5,
                             random_state=2022)
model = experiment.run()
# model.save('./model/')
# model.get_pipeline_params()
print(model)
# X_test, y_test = model.split_X_y(test_data.copy())
# X_test_columns = [i for i in df_train.columns if i not in ['主蒸汽流量']]
X_test_columns = [i for i in df_train.columns if i in ['时间', '给水流量']]
X_valid = valid_data[X_test_columns]
Y_valid = valid_data['主蒸汽流量']
forecast = model.predict(X_valid)
print(forecast.head())
results = model.evaluate(y_true=Y_valid, y_pred=forecast)

X_test = df_test[X_test_columns]
forecast = model.predict(X_test)
print(forecast.head())
df_test['Steam_flow'] = forecast['主蒸汽流量']
df_test['ID'] = range(1, 1801)
df_test = df_test[['ID', '时间', 'Steam_flow']]
# df_test = df_test.loc[:, ['ID', 'Time', 'Steam_flow']]
df_test.to_csv('./test_prediction.csv', index=False)
print(results.head())

time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
