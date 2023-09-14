import os
import pandas as pd
from datetime import timedelta

'''
#middle 142 north 166 south 124
middle= ["nfb0061", "nfb0063", "nfb0445", "nfb0067", "nfb0443","nfb0069", "nfb0071", "nfb0073", "nfb0077", "nfb0079", "nfb0081", "nfb0083", 
 "nfb0085", "nfb0087", "nfb0089", "nfb0091", "nfb0093", "nfb0095", "nfb0099", "nfb0101", "nfb0103", "nfb0423", "nfb0105", "nfb0107", 
"nfb0062", "nfb0064", "nfb0446", "nfb0068", "nfb0444","nfb0070",  "nfb0072","nfb0074", "nfb0078", "nfb0080", "nfb0082", "nfb0084", 
"nfb0086", "nfb0088", "nfb0090", "nfb0092", "nfb0094", "nfb0096", "nfb0100", "nfb0102", "nfb0104", "nfb0424", "nfb0106", "nfb0108", 
"nfb0229", "nfb0231", "nfb0233", "nfb0235", "nfb0237", "nfb0241", "nfb0243", "nfb0245", "nfb0247", "nfb0249", "nfb0253", "nfb0255", 
"nfb0257", "nfb0259", "nfb0261", "nfb0263", "nfb0265", "nfb0267", "nfb0269","nfb0271", "nfb0273", "nfb0411", "nfb0275", "nfb0277", 
"nfb0279", "nfb0451", "nfb0281", "nfb0285", "nfb0287", "nfb0461", "nfb0289", "nfb0463", "nfb0291", "nfb0465", "nfb0295", "nfb0230", 
"nfb0232",  "nfb0234", "nfb0236", "nfb0238", "nfb0242", "nfb0244", "nfb0246", "nfb0248", "nfb0250","nfb0254", "nfb0256", "nfb0258", 
 "nfb0260", "nfb0262", "nfb0264", "nfb0266", "nfb0268", "nfb0270","nfb0272", "nfb0274", "nfb0412", "nfb0276", "nfb0278", "nfb0280", 
"nfb0452", "nfb0282", "nfb0286", "nfb0288", "nfb0462", "nfb0290", "nfb0464", "nfb0292", "nfb0466", "nfb0296", "nfb0355", "nfb0357", 
"nfb0359", "nfb0361", "nfb0363", "nfb0356", "nfb0358", "nfb0360", "nfb0362", "nfb0364", "nfb0379", "nfb0417", "nfb0381", "nfb0383", 
"nfb0447", "nfb0385", "nfb0387", "nfb0380", "nfb0418", "nfb0382", "nfb0384", "nfb0448", "nfb0386","nfb0388" ]


south=[ "nfb0111", "nfb0113", "nfb0115", "nfb0117", "nfb0421" , "nfb0121", "nfb0123", "nfb0125", "nfb0127", "nfb0415","nfb0467",
"nfb0131","nfb0133","nfb0135","nfb0137","nfb0139","nfb0413", "nfb0141","nfb0145","nfb0147","nfb0149","nfb0151","nfb0153",
"nfb0155","nfb0157","nfb0455","nfb0457", "nfb0112","nfb0114","nfb0116","nfb0118","nfb0422","nfb0122","nfb0124","nfb0126",
"nfb0128","nfb0416","nfb0132","nfb0134","nfb0136","nfb0138","nfb0468","nfb0140","nfb0142","nfb0146","nfb0148","nfb0150",
"nfb0414","nfb0152" ,"nfb0154" ,"nfb0156", "nfb0158" ,"nfb0456","nfb0458","nfb0297" ,"nfb0299" ,"nfb0301" ,"nfb0303" ,"nfb0305",
"nfb0309"  ,"nfb0449", "nfb0311", "nfb0313", "nfb0315", "nfb0319","nfb0321", "nfb0323", "nfb0325", "nfb0327",  "nfb0331", "nfb0333", 
"nfb0335", "nfb0337", "nfb0341", "nfb0343", "nfb0345", "nfb0347", "nfb0298", "nfb0300", "nfb0302", "nfb0304","nfb0306", "nfb0310", 
 "nfb0450", "nfb0312", "nfb0314", "nfb0316", "nfb0320", "nfb0322", "nfb0324", "nfb0326", "nfb0328", "nfb0332", "nfb0334", "nfb0336", 
"nfb0338", "nfb0342", "nfb0344", "nfb0346", "nfb0348", "nfb0389", "nfb0391", "nfb0393", "nfb0395", "nfb0397", "nfb0390","nfb0392", "nfb0394", 
 "nfb0396", "nfb0398", "nfb0399", "nfb0401", "nfb0403", "nfb0405", "nfb0407", "nfb0409", "nfb0453", "nfb0400", "nfb0402", "nfb0404", "nfb0406", 
"nfb0408", "nfb0410", "nfb0454"]

north=["nfb0001","nfb0003","nfb0005","nfb0419","nfb0007", "nfb0013","nfb0015", "nfb0017", "nfb0019", "nfb0021" ,"nfb0023" ,"nfb0025","nfb0027", 
"nfb0029", "nfb0031", "nfb0425", "nfb0033" ,"nfb0035","nfb0037", "nfb0039" ,"nfb0041" ,"nfb0427", "nfb0429" , "nfb0043", "nfb0045", "nfb0047",
"nfb0049" ,"nfb0431" ,"nfb0053" ,"nfb0055" ,"nfb0057", "nfb0059", "nfb0002" ,"nfb0004", "nfb0006", "nfb0420", "nfb0008" ,"nfb0012", 
"nfb0014", "nfb0016", "nfb0018" ,"nfb0020" ,"nfb0022", "nfb0024" ,"nfb0026", "nfb0028", "nfb0030" ,"nfb0032","nfb0426", "nfb0034", "nfb0036" ,
"nfb0038" ,"nfb00404" ,"nfb0042" ,"nfb0428" ,"nfb0428" ,"nfb0044" ,"nfb0046" ,"nfb0048", "nfb0050", "nfb0432", "nfb0054" ,"nfb0056" ,"nfb0058" ,
"nfb0060" ,"nfb0159", "nfb0161", "nfb0161", "nfb0165" ,"nfb0433" ,"nfb0435" ,"nfb0437", "nfb0439" ,"nfb0471", "nfb0441", "nfb0160", "nfb0162" ,
"nfb0164" ,"nfb0166" ,"nfb0434" ,"nfb0436", "nfb0438", "nfb0440" ,"nfb0472" ,"nfb0442" ,"nfb0167" ,"nfb0169" ,"nfb0171" ,"nfb0173" ,"nfb0175" ,"nfb0177", 
"nfb0168" ,"nfb0170","nfb0170","nfb0170","nfb0170","nfb0170","nfb0179", "nfb0181","nfb0185","nfb0187","nfb0189", "nfb0191","nfb0193","nfb0195","nfb0197",
"nfb0199","nfb0201","nfb0203","nfb0205","nfb0469" ,"nfb0209" ,"nfb0211","nfb0213","nfb0215","nfb0219","nfb0221" "nfb0223" "nfb0225" "nfb0227" ,
"nfb0180", "nfb0182", "nfb0186", "nfb0188", "nfb0190", "nfb0192", "nfb0194", "nfb0196", "nfb0198", "nfb0200", "nfb0202", "nfb0204", "nfb0206", "nfb0470" ,
"nfb0210" ,"nfb0212","nfb0214" ,"nfb0216" ,"nfb0220","nfb0222",  "nfb0224",  "nfb0226",  "nfb0228",  "nfb0349", "nfb0351","nfb0353","nfb0350","nfb0352",
"nfb0354","nfb1001", "nfb1002", "nfb1003", "nfb1004", "nfb1005", "nfb1006", "nfb1007", "nfb1008", "nfb0365" ,"nfb0366", "nfb0367" ,"nfb0368" ,"nfb0369", 
"nfb0370", "nfb0373" ,"nfb0375" ,"nfb0377" ,"nfb0374" ,"nfb0376" ,"nfb0378" ]
'''

road_name =[]


csv_file_dict = 'Data/csv/' 

for roadname in road_name:
	print(roadname)
	csv_file = csv_file_dict
	roadname_csv = open('roads/south_all_segments/'+'traffic_data_' + roadname + '.csv','w') 
	first = 0
	count = 0
	for filename in os.listdir(csv_file):
		data = pd.read_csv(csv_file + filename, dtype=str)
		route_data = data.loc[data['routeid'] == roadname, ['datacollecttime', 'traveltime', 'value']]
		if(len(route_data) > 0):
			time = filename[0:4] + "/" + filename[4:6] + "/" + filename[6:8] + " " + filename[9:11] + ":" + filename[11:13]
			route_data.ix[route_data.index.values,'datacollecttime'] = pd.to_datetime(time, format='%Y/%m/%d %H:%M')
		count = count + 1
		if(count % 1000 == 0):
			print(count)
		route_data = route_data.loc[route_data['traveltime'].astype(int) >= 0,:]
		route_data = route_data.loc[route_data['value'].astype(int) >= 0,:]
		if(first == 0):
			route_data.to_csv(roadname_csv, sep=',',index = False)
			first = 1
		else:
			route_data.to_csv(roadname_csv, sep=',',index = False,header = False)
	roadname_csv.close()
