import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def build_scaler(values):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaled_values = scaler.fit_transform(values)
    return scaler, scaled_values


def difference(dataset, interval=1):
    """
        Create a differenced series.
        RNN is more likely to learn the 
        difference values of original data.
    """
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)

    
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    """
        Convert the problem to supervised learning problem.
        For the total data, the input is the n_in days and
        the output is the n_out days.
    """
    n_vars = 1 if type(data) is list else data.shape[1] # 1
    df = pd.DataFrame(data)
    
    cols, names = list(), list()
    # input sequence (t-n_in, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n_out)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg
    
    
def prepare_data(series, n_lag, n_seq):
    """
        Transform series into train and test sets for supervised learning
    """
    raw_values = series.values
    
    # transform data to be stationary
    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values.reshape(len(diff_series), 1)
    
    # rescale values to -1, 1
    scaler, scaled_values = build_scaler(diff_values)
    
    # transform into supervised learning problem X, y
    supervised = series_to_supervised(scaled_values, n_lag, n_seq)
    supervised_values = supervised.values
    
    # split into train and test sets
    #train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
    return scaler, supervised_values
    

# invert differenced forecast
def inverse_difference(last_ob, forecast):
    # invert first forecast
    inverted = list()
    inverted.append(forecast[0] + last_ob)
    # propagate difference forecast using inverted first value
    for i in range(1, len(forecast)):
        inverted.append(forecast[i] + inverted[i-1])
    return inverted


# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
    inverted = list()
    for i in range(len(forecasts)):
        # invert scaling
        forecast = np.array(forecasts[i]).reshape(1, len(forecasts[i]))
        inv_scale = scaler.inverse_transform(forecast)
        inv_scale = inv_scale[0, :]
        # invert differencing
        index = len(series) - n_test + i - 1
        last_ob = series.values[index]
        inv_diff = inverse_difference(last_ob, inv_scale)
        inverted.append(inv_diff)
    return inverted

#actual = inverse_transform_v2('actual', trav_mean, trav_std, actual, n_test+2)
def inverse_transform_v2(s_type, v1, v2, series, n_test):
    origin_df = pd.read_csv('traffic_data_nfb0008_dataframe.csv')
    origin_series = origin_df['traveltime'].astype(int)
    start = len(origin_series) - n_test + 1
    last_ob = origin_series.values[start]
    
    if s_type == "forecasts":
        for i in range(len(series)): 
            # series[i] = series[i] * (v2-v1) + v1 # min max scaling
            series[i] = series[i] * v2 + v1 # standarization scaling
            series[i] = last_ob + series[i] # without multi-step
            last_ob = series[i]
    elif s_type == "actual":
        series = list(origin_series[start:])
    
    return series
def inverse_transform_v3(trav_mean, trav_std, series):
    for i in range(len(series)):
        series[i] = series[i] * trav_std + trav_mean # standarization scaling
    return series
def inverse_transform_minmax(trav_max, trav_min, series):
    for i in range(len(series)):
        series[i] = series[i] * (trav_max - trav_min)+trav_min  # standarization scaling
    return series
def prepare_data_v2(series, n_seq=12, n_lag=12):
    raw_values = series.values

    diff_series = difference(raw_values, 1)
    diff_values = diff_series.values.reshape(len(diff_series), 1)

    scaler, scaled_values = build_scaler(diff_values)
    supervised = series_to_supervised_v2(scaled_values, n_seq, n_lag)


    agg.dropna(inplace=True)
