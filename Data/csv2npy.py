import pandas as pd   
import numpy as np
from tqdm import tqdm
import datetime


def zscore_scaling(df):
    df2 = df[df['traveltime'] != -99999]
    #print("traveltime mean: ", df2['traveltime'].mean(), " traveltime std: ", df2['traveltime'].std()) #traveltime, mean and std are 167.28 and 48.96.
    traveltime_mean, traveltime_std = df2['traveltime'].mean(), df2['traveltime'].std()
    tmp_df = df2.loc[:,['traveltime', 'value']]
    tmp_df = (tmp_df - tmp_df.mean(axis=0)) / tmp_df.std(axis=0)
    df2.loc[:,['traveltime', 'value']] = tmp_df
    df[df['traveltime'] != -99999] = df2
    return traveltime_mean, traveltime_std, df
def minmax_scaling(df):
    df2 = df[df['traveltime'] != -99999]
    #print("traveltime mean: ", df2['traveltime'].mean(), " traveltime std: ", df2['traveltime'].std()) #traveltime, mean and std are 167.28 and 48.96.
    traveltime_max, traveltime_min = df2['traveltime'].max(), df2['traveltime'].min()
    tmp_df = df2.loc[:,['traveltime', 'value']]
    tmp_df = (tmp_df - tmp_df.min(axis=0)) / (tmp_df.max(axis=0) - tmp_df.min(axis=0))
    df2.loc[:,['traveltime', 'value']] = tmp_df
    df[df['traveltime'] != -99999] = df2
    return traveltime_max, traveltime_min, df
    
def build_supervised_data(df, test_x,test_y, train1_x,train1_y,train2_x, train2_y, predict_time,last_time): #predict_time hour
    #df = pd.read_csv(in_file)
    print(f'predict_time: {predict_time}, last_time: {last_time}')
    print(f"Input file shape: {df.shape}") #(105696, 316)
    # normalize 
    print(f"\nnormalizing data...")
    trav_mean, trav_std, df = zscore_scaling(df) 
    #print(trav_mean, trav_std)  #162.00442165798611  ,  27.07442385772283
    #print(f"\nAfter normalization:\n {df.describe()}")
    
    # configuration
    timestamp = 5
    past_day = 7
    n_short_term_time_slot = 12
    time_slot_per_hour = int(60 / timestamp) # 12
    time_slot_per_day = int(60*24 / timestamp) # 288
    current_time = int(past_day*time_slot_per_day + n_short_term_time_slot) #7*288+12=2028
    predict_time = int(predict_time*time_slot_per_hour) # [0,24,48,..]*12
    
    print(f"\nbuilding supervised data...")
    interval0 = datetime.date(2020,4,1)-datetime.date(2019,10,1) #
    interval1 = datetime.date(2020,10,1)-datetime.date(2020,4,1) #
    interval2 = datetime.date(2021,1,31)-datetime.date(2020,10,1) #datetime.date(2021,2,28)-datetime.date(2020,10,1)
    #print(interval0,interval1,interval2) #183 183,150
    train_df_1, test_df = df[:time_slot_per_day*int(interval0.days)].reset_index(drop=True), df[(-time_slot_per_day*int(interval2.days)):].reset_index(drop=True)
    train_df_2 = df[time_slot_per_day*int(interval0.days):time_slot_per_day*int(interval0.days)+time_slot_per_day*int(interval1.days)].reset_index(drop=True)
    print(f"train start date: {2019,10,1} / test start date: {2020,10,1}")
    print(train_df_1.shape,train_df_2.shape,test_df.shape) #(52704, 317) (52704, 317) (43200, 317)

    sample_list_index, start = list(), 0
    for i in range(past_day+1):
        if i == past_day:
            sample_list_index.extend(list(range(start-time_slot_per_hour,start)))
            sample_list_index.extend(list(range(start+predict_time,start+predict_time+last_time*time_slot_per_hour))) # target predict 1 hour in the future, last 1 hour time
            break
        else:
            sample_list_index.extend(list(range(start,start+1*time_slot_per_hour))) ###### 1 hour segment or 2 hour? from historical data#############
            start += time_slot_per_day
    train_list_index_1 = sample_list_index
    train_list_index_2 = sample_list_index
    test_list_index = sample_list_index
    
    start_time_index=int(7*time_slot_per_day+predict_time+time_slot_per_hour-1)
    #print('train columns',train_df.columns)
    #print(train_df.shape)
    
    # ==============================
    # building test supervised data
    # ==============================
    # append samples into list to make supervise data 
    
    file_list_x = list()
    file_list_y = list()
    for step in tqdm(range(start_time_index, len(test_df))):
        if test_df.loc[test_list_index,:].isna().sum().sum() == 0: # ignore missing values
            sub_df = test_df.iloc[test_list_index,1:] 
            sub_df_x=sub_df.values[:96,:]  #180=7x24+12+1(current); 96=7x12+12
            #-----------only travel time [96:,0]; full temporal data [96:,:]------------------------
            sub_df_y=sub_df.values[96:,:] 
            file_list_x.append(sub_df_x.tolist())
            file_list_y.append(sub_df_y.tolist())
        test_list_index = [x+1 for x in test_list_index]
    test_supervised_values_x = np.array(file_list_x,dtype='float16')
    test_supervised_values_y = np.array(file_list_y,dtype='float16')
    np.save(test_x, test_supervised_values_x)
    np.save(test_y, test_supervised_values_y)    
    print(f"final test supervised value x shape: {test_supervised_values_x.shape}")  #(41161, x, 316)
    print(f"final test supervised value y shape: {test_supervised_values_y.shape}")  
        
    # ==============================
    # building train supervised data
    # ==============================        
    # append samples into list to make supervise data
    
    file_list_x = list()
    file_list_y = list()
    for step in tqdm(range(start_time_index, len(train_df_1))):
        if train_df_1.loc[train_list_index_1,:].isna().sum().sum() == 0: # ignore missing values
            sub_df = train_df_1.iloc[train_list_index_1,1:]
            sub_df_x=sub_df.values[:96,:]
            sub_df_y=sub_df.values[96:,:]
            file_list_x.append(sub_df_x.tolist())
            file_list_y.append(sub_df_y.tolist())
        train_list_index_1 = [x+1 for x in train_list_index_1]

    train_supervised_values_x = np.array(file_list_x,dtype='float16')
    train_supervised_values_y = np.array(file_list_y,dtype='float16')
    np.save(train1_x, train_supervised_values_x)
    np.save(train1_y, train_supervised_values_y)
    print(f"final train supervised value x shape for part1: {train_supervised_values_x.shape}")     
    print(f"final train supervised value y shape for part1: {train_supervised_values_y.shape}")    
    
    file_list_x = list()
    file_list_y = list()
    for step in tqdm(range(start_time_index, len(train_df_2))):
        if train_df_2.loc[train_list_index_2,:].isna().sum().sum() == 0: # ignore missing values
            sub_df = train_df_2.iloc[train_list_index_2,1:]
            sub_df_x=sub_df.values[:96,:]
            sub_df_y=sub_df.values[96:,:]
            file_list_x.append(sub_df_x.tolist())
            file_list_y.append(sub_df_y.tolist())
        train_list_index_2 = [x+1 for x in train_list_index_2]

    train_supervised_values_x = np.array(file_list_x,dtype='float16')
    train_supervised_values_y = np.array(file_list_y,dtype='float16')
    np.save(train2_x, train_supervised_values_x)
    np.save(train2_y, train_supervised_values_y)
    print(f"final train supervised value x shape for part2: {train_supervised_values_x.shape}")     
    print(f"final train supervised value y shape for part2: {train_supervised_values_y.shape}")    
    
    # save np file
    print(f"done")
    
root_path = "Data/"
roads=['nfb0370'] 
#north : 'nfb0370', 'nfb0431', 'nfb0019','nfb0033','nfb0425',#8/16/2023 9:33AM
#middle: 'nfb0064', 'nfb0063', 'nfb0247', 'nfb0248', 'nfb0061'#8/16/2023 11:05AM
#south: 'nfb0327', 'nfb0328', 'nfb0117', 'nfb0124', 'nfb0123'
for road in roads:
  print('For road:',road)
  df=pd.read_csv(root_path+'traffic_data_'+road+'_dataframe.csv')
  df=pd.get_dummies(df, columns=['month','dayofweek','holiday','time_slot','peak'])
  #print("csv columns",df.columns) # 317=date+traveltime+value+one-hot_314(288 12 7 4 3)
  
  last_time=1
   #for predict time = 0, generate both x & y 
   #for 1-7 predict time only generate y, Manual mark the generating code for x, because they share the same input x as the first 0 day.
  for predict_time in range(0,1):
      selecteddata = '_'+str(predict_time)+'d_last'+str(last_time)+'h'
      predict_time=predict_time*24     
      build_supervised_data(df, root_path+road+'/test_data_x'+road+'_periodic.npy',root_path+road+'/test_data_y'+road+'_periodic'+selecteddata+'.npy',
                        root_path+road+'/train_data1_x'+road+'_periodic.npy',root_path+road+'/train_data1_y'+road+'_periodic'+selecteddata+'.npy',
                        root_path+road+'/train_data2_x'+road+'_periodic.npy',root_path+road+'/train_data2_y'+road+'_periodic'+selecteddata+'.npy',
                        predict_time=predict_time,last_time=last_time)

