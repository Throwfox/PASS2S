# -- coding: utf-8 --

import urllib.request
from html.parser import HTMLParser
import gzip  
import os  

filename="123"

form=""
inyear=0
inmonth=0
inday=0
wantfirst=2355
filepath='xml/'
g=open(filepath+"name.txt","w")
h=open(filepath+"lost.txt","w")

year="2021"                                    #starting time
month="2"
day="28"
while year!="2019" or month!="9" or day!="30": #end time
    class WeatherList(HTMLParser):
        name = []
        def handle_data(self, data):
            if data[0:16]=="roadlevel_value5":
                self.name.append(data)
        def __del__(self):
            print("bye-bye")
    if int(day,10)>=10:
        if int(month)>=10:
            form=year+month+day+'/'
        else:
            form=year+'0'+month+day+'/'
    else:
        if int(month)>=10:
            form=year+month+'0'+day+'/'
        else:
            form=year+'0'+month+'0'+day+'/'
    print('http://tisvcloud.freeway.gov.tw/history/roadlevel/'+form)
    content = urllib.request.urlopen('http://tisvcloud.freeway.gov.tw/history/roadlevel/'+form).read()
    Tempreature = WeatherList()
    Tempreature.feed(str(content))
    def un_gz(file_name,form): #'roadlevel_value5_0000.xml.gz'
        ff_name=file_name.replace(".gz","")
        fff_name=ff_name.replace("roadlevel_value5_","") #0000.xml
        f_name = file_name.replace(file_name, form+'_'+fff_name)#20200120_000.xml
        ffff_name=fff_name.replace(".xml","") #0000
        g.write(form+'_'+ffff_name+"\n") #20200120_0000
        g_file = gzip.GzipFile(filepath+file_name)
        open(filepath+f_name, "w+").write(str(g_file.read().decode('utf8')))    
        g_file.close()
        
    wantfirst=2355
    for i in Tempreature.name:
        while int(i[17:21],10) != wantfirst:
            h.write(year+'_'+month+'_'+day+'_'+str(wantfirst)+"\n")
            wantfirst=wantfirst-5
            if wantfirst%100==95:
                wantfirst=wantfirst-40
        wantfirst=wantfirst-5
        if wantfirst%100==95:
            wantfirst=wantfirst-40  
        filename=i
        add='http://tisvcloud.freeway.gov.tw/history/roadlevel/'+form+i
        #print(add)

        urllib.request.urlretrieve(add, filepath+filename)
        #print year+month+day
        un_gz(filename,form[:-1])
        #print filename
        os.remove(filepath+filename)
        #print('%s,done.'% filename)
        if i =='roadlevel_value5_0000.xml.gz':                            
            break
########### switch to yesterday###############
    if int(day,10)==1:
        if int(month,10)==1:
            inyear=int(year,10)
            inyear-=1
            year=str(inyear)
            day="32"
            month="12"
        elif int(month,10)==12:
            inmonth=int(month,10)
            inmonth-=1
            month=str(inmonth)
            day="31"
        elif int(month,10)==11:
            inmonth=int(month,10)
            inmonth-=1
            month=str(inmonth)
            day="32"
        elif int(month,10)==10:
            inmonth=int(month,10)
            inmonth-=1
            month=str(inmonth)
            day="31"
            print ("set day="+day)
        elif int(month,10)==9:
            inmonth=int(month,10)
            inmonth-=1
            month=str(inmonth)
            day="32"
        elif int(month,10)==8:
            inmonth=int(month,10)
            inmonth-=1
            month=str(inmonth)
            day="32"
        elif int(month,10)==7:
            inmonth=int(month,10)
            inmonth-=1
            month=str(inmonth)
            day="31"
        elif int(month,10)==6:
            inmonth=int(month,10)
            inmonth-=1
            month=str(inmonth)
            day="32"
        elif int(month,10)==5:
            inmonth=int(month,10)
            inmonth-=1
            month=str(inmonth)
            day="31"
        elif int(month,10)==4:
            inmonth=int(month,10)
            inmonth-=1
            month=str(inmonth)
            day="32"
        elif int(month,10)==3:
            inmonth=int(month,10)
            inmonth-=1
            month=str(inmonth)
            day="30" #2020 year feb have 29th day
        elif int(month,10)==2:
            inmonth=int(month,10)
            inmonth-=1
            month=str(inmonth)
            day="32"
    inday=int(day,10)
    inday-=1
    day=str(inday)
    print ("Next day No.%s" % day)

h.close()
g.close()
