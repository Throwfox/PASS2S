import pandas as pd   
import numpy as np
from tqdm import tqdm
import datetime

north_roads=[ 'nfb0370', 'nfb0431', 'nfb0019', 'nfb0033', 'nfb0425']
middle_roads=[ 'nfb0064', 'nfb0063', 'nfb0247', 'nfb0248', 'nfb0061']
south_roads=[ 'nfb0327', 'nfb0328', 'nfb0117', 'nfb0124', 'nfb0123']

filepath='Data/roads/nouth_all_segments/'
    #traffic_data_nfb0008.csv to traffic_data_nfb0008_dataframe.csv, 
for road in south_roads:
    print(f'For road:{road}')
    holiday = ['2019-10-10','2019-10-11','2019-10-12','2019-10-13','2020-01-01','2020-02-28',
     '2020-02-29','2020-03-01','2020-04-02','2020-04-03','2020-04-04','2020-04-05',
     '2020-06-25','2020-06-26','2020-06-27', '2020-06-28','2020-10-01','2020-10-02',
     '2020-10-03','2020-10-04','2020-10-09','2020-10-10','2020-10-11','2021-01-01',
     '2021-01-02', '2021-01-03']
    nonholiday =['2019-10-05','2020-02-15','2020-06-20','2020-09-26']
    df = pd.read_csv(filepath+'traffic_data_'+road+'.csv')
    print(f"Input file shape: {df.shape}")
    ##### sorted
    df['datacollecttime'] = pd.to_datetime(df['datacollecttime'], infer_datetime_format = True)
    df.sort_values(by = ['datacollecttime'], ascending = True, inplace = True)
    df=df.reset_index(drop=True)
    
    print('\nprocessing data...')
    start_time, end_time = '2019-10-01 00:00:00', '2021-01-31 23:55:00'
    timestamp_list = pd.date_range(start=start_time, end=end_time, freq='5min').tolist()
    df_timestamp_list = list(df['datacollecttime'].values)
    print(f"\nfile series length: {len(df_timestamp_list)} / full series length should be {len(timestamp_list)}")
    
    print("\nhandling missing values...")
    new_df = pd.DataFrame(timestamp_list, columns=['datacollecttime'])
    new_df = pd.merge(df, new_df, on='datacollecttime', how='outer').sort_values('datacollecttime')
    new_df=new_df.reset_index(drop=True)
    new_df['value'] = new_df['value'].fillna(0)
    new_df['traveltime'] = new_df['traveltime'].fillna(0)
    for i in tqdm(range(len(new_df))):
        if new_df.loc[i,'traveltime'] == 0 and i >288 :        
            before, after = i-288, i+288  #(1 day, before or after)
            while new_df.loc[before,'traveltime'] == 0: 
                before -= 288
            while new_df.loc[after,'traveltime'] == 0:
                after += 288
            new_df.loc[i,'traveltime'] = (new_df.loc[before,'traveltime']+new_df.loc[after,'traveltime']) / 2
            new_df.loc[i,'value'] = (new_df.loc[before,'value']+new_df.loc[after,'value']) / 2
        elif new_df.loc[i,'traveltime'] == 0 and i <288 :  
            after = i+288  #(1 day, before or after)
            while new_df.loc[after,'traveltime'] == 0:
                after += 288
            new_df.loc[i,'traveltime'] = new_df.loc[after,'traveltime'] 
            new_df.loc[i,'value'] = new_df.loc[after,'value']    
    print("generating features...")
    new_df['datacollecttime'] = pd.to_datetime(new_df['datacollecttime'])
    new_df['month'] = new_df['datacollecttime'].astype(str).str[5:7].astype(int) # 1 ~ 12
    new_df['dayofweek'] = new_df['datacollecttime'].dt.dayofweek# Mon=0,Tus=1,Wed=2,Thur=3,Fri=4,Sat=5,Sun = 6 
    #new_df['dayofweek_name'] = new_df['datacollecttime'].dt.day_name()
    new_df['holiday'] = 0
    x=np.arange(1,289)
    interval = datetime.date(2021,1,31)-datetime.date(2019,10,1)
    for i in range(int(interval.days)):
        b=np.arange(1,289)
        x=np.concatenate((x, b), axis=0)
    new_df['time_slot']=x
    new_df['peak'] = (np.where((new_df['datacollecttime'].astype(str).str[11:13].astype(int)>=6) & (new_df['datacollecttime'].astype(str).str[11:13].astype(int)<=9), 
           '1', np.where((new_df['datacollecttime'].astype(str).str[11:13].astype(int)>=18) 
           & (new_df['datacollecttime'].astype(str).str[11:13].astype(int)<=22), '2', 
           np.where((new_df['datacollecttime'].astype(str).str[11:13].astype(int)>=10) & (new_df['datacollecttime'].astype(str).str[11:13].astype(int)<=13), '3', '0'))))
    ############################################ peak #################################################
    #23:00-05:55, 14:00-17:55 0,Non-peak
    #06:00-09:55 1, morning peak
    #18:00-22:55 2, night peak
    #10:00—13:55 3,noon peak
    ####################################################################################################
    #weekday 0 weekend 1 holiday 2
    
    print('Holiday making')
    # holiday
    for i in tqdm(range(len(new_df))):
        if(str(new_df['datacollecttime'].dt.date[i]) in holiday):
            new_df.loc[i,'holiday'] = 2
        elif((new_df.loc[i,'dayofweek'] == 5 or new_df.loc[i,'dayofweek'] == 6) and str(new_df['datacollecttime'].dt.date[i]) not in nonholiday):
            new_df.loc[i,'holiday'] = 1
    new_df.to_csv('Data/traffic_data_'+road+'_dataframe.csv',index=False)





