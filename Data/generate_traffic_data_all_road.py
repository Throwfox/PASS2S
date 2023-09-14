import os
import pandas as pd
from datetime import timedelta


csv_file = "csv"
traffic_all_road_csv = open('traffic_all_road.csv','w')
first = 0
count = 0
for filename in os.listdir(csv_file):
	
	data = pd.read_csv(csv_file + filename, dtype=str)
	route_data = data.loc[:, ['routeid','datacollecttime', 'value']]
	if(len(route_data) > 0):
		time = filename[0:4] + "/" + filename[4:6] + "/" + filename[6:8] + " " + filename[9:11] + ":" + filename[11:13]
		route_data.ix[route_data.index.values,'datacollecttime'] = pd.to_datetime(time, format='%Y/%m/%d %H:%M')

	count = count + 1
	if(count % 1000 == 0):
		print(count)
	if(first == 0):
		route_data = route_data.loc[route_data['value'].astype(int) >= 0,:]
		route_data.to_csv(traffic_all_road_csv, sep=',',index = False)
		first = 1
	else:
		route_data = route_data.loc[route_data['value'].astype(int) >= 0,:]
		route_data.to_csv(traffic_all_road_csv, sep=',',index = False,header = False)
		first = 1
traffic_all_road_csv.close()
		