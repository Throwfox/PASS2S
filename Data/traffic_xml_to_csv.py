import xml.etree.ElementTree as ET
import csv
import os

xml_file_dict ='xml/'
csv_file_dict = 'csv/'
for filename in os.listdir(xml_file_dict):
    try:
        if('.xml' in filename):
          xml_fname = xml_file_dict + filename
          csv_fname = csv_file_dict + filename.split(".")[0] + ".csv"      
          fields = ["routeid","level","value","traveltime","datacollecttime"]
          tree = ET.parse(xml_fname)
          root = tree.getroot()
          
          with open(csv_fname, "w", newline='') as f:
          	writer = csv.DictWriter(f, fieldnames=fields)
          	writer.writeheader()
          	for child in root:
          		for child2 in child:
          			writer.writerow(child2.attrib)
    except:
        print(filename)
        continue
