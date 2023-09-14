import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib 
matplotlib.use('Agg') 
# evaluation metrics
from sklearn import metrics
from math import sqrt

def smoothing(vec):
    for i in range(len(vec)):
        vec[i] = 0.00001 if vec[i] == 0 else vec[i]
    return vec

# metrics (input shape: 1D array (1,))	
def mean_squared_error(y_true, y_pred):
    return metrics.mean_squared_error(y_true, y_pred)

def root_mean_squared_error(y_true, y_pred):
    return sqrt(metrics.mean_squared_error(y_true, y_pred))

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_true = smoothing(y_true)
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100

def symmetric_mean_absolute_error(y_true, y_pred):
    output=list()
    for i in range(y_true.shape[1]):
      y_true[:,i] = np.array(y_true[:,i])
      y_pred[:,i] = np.array(y_pred[:,i])
      y_true[:,i] = smoothing(y_true[:,i])
      output.append(100/len(y_true[:,i]) * np.sum(2 * np.abs(y_pred[:,i] - y_true[:,i]) / (np.abs(y_true[:,i]) + np.abs(y_pred[:,i]))))
    return np.mean(output)
	
	
# plot the forecasts in the context of the original dataset
def plot_forecasts(forecasts, actual, title, file_path):
    plt.plot(forecasts[:288*7], label='forecast')
    plt.plot(actual[:288*7], label='actual')
    plt.title(title)
    plt.xlabel('steps')
    plt.ylabel('traveltime')
    plt.legend(loc='upper right')
    plt.savefig(file_path)


# plot data speed distribution
def plot_year_speed_distribution(series, roadId):
    year_list = [2013, 2014, 2015, 2016, 2017, 2018, 2019]
    for year in year_list:
        sub_series = series[series.index.year == year]
        plt.figure(figsize=(20, 6))
        plt.plot(sub_series)
        plt.savefig(f'figures/distribution/{roadId}_{str(year)}_speed_distribution.png')
