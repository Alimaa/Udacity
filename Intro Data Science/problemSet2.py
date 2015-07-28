import pandas as pd
import pandasql
import csv
import urllib2
import os

cwd = os.getcwd()
os.chdir('%s\\Intro Data Science\\Problem Set 2' %(cwd))

#################################################################
##################### Problem Set #2 ############################
#################################################################

def num_rainy_days(filename):

    weather_data = pd.read_csv(filename)

    q = "SELECT COUNT(*) FROM weather_data WHERE rain=1"

    rainy_days = pandasql.sqldf(q.lower(), locals())

    return rainy_days

print num_rainy_days('weather_underground.csv')

def max_temp_aggregate_by_fog(filename):

    weather_data = pd.read_csv(filename)

    q = 'SELECT fog, MAX(maxtempi) FROM weather_data GROUP BY fog'

    foggy_days = pandasql.sqldf(q.lower(), locals())
    return foggy_days

print max_temp_aggregate_by_fog('weather_underground.csv')

def avg_weekend_temperature(filename):

    weather_data = pd.read_csv(filename)

    q = "select AVG(meantempi) from weather_data WHERE cast(strftime('%w',date) as integer) in (0,6)"

    mean_temp_weekends = pandasql.sqldf(q.lower(), locals())
    return mean_temp_weekends

print avg_weekend_temperature('weather_underground.csv')

def avg_min_temperature(filename):

    weather_data = pd.read_csv(filename)

    q = "SELECT AVG(mintempi) from weather_data WHERE rain=1 AND mintempi>55"

    avg_min_temp_rainy = pandasql.sqldf(q.lower(), locals())
    return avg_min_temp_rainy

print avg_min_temperature('weather_underground.csv')

'''

I used the code below to read the text file from web and save it to disk:

url = 'http://web.mta.info/developers/data/nyct/turnstile/turnstile_110507.txt'
response = urllib2.urlopen(url)

with open('test.txt','wb') as file:

    file.write(response.read())

'''
def fix_turnstile_data(filenames):

    for name in filenames:

        with open(name,'rb') as f:

            reader = csv.reader(f)

            list=[]

            for row in reader:

                start = row[:3]
                end = row[3:]

                for i in range(len(end)/5):

                    out = start + end[i*5:(i+1)*5]
                    list.append(out)

            with open('updated_%s' %(name),'wb') as f1:

                fw = csv.writer(f1,dialect='excel')
                fw.writerows(list)

fix_turnstile_data(['test.txt'])

def create_master_turnstile_file(filenames, output_file):

    with open(output_file, 'w') as master_file:

        master_file.write('C/A,UNIT,SCP,DATEn,TIMEn,DESCn,ENTRIESn,EXITSn\n')

        for filename in filenames:

            with open(filename,'rb') as f:

                for row in f:

                    master_file.write(row)

create_master_turnstile_file(['updated_test.txt'],'output.txt')

def filter_by_regular(filename):

    turnstile_data = pd.read_csv(filename)
    turnstile_data = turnstile_data[turnstile_data.DESCn=='REGULAR']

    return turnstile_data

def get_hourly_entries(df):

    df['ENTRIESn_hourly'] = df.ENTRIESn.shift(1)
    df['ENTRIESn_hourly'] = df.ENTRIESn - df['ENTRIESn_hourly']
    df['ENTRIESn_hourly'] = df['ENTRIESn_hourly'].fillna(1)

    return df

def get_hourly_exits(df):

    df['EXITSn_hourly'] = df.EXITSn.shift(1)
    df['EXITSn_hourly'] = df.EXITSn - df['EXITSn_hourly']
    df['EXITSn_hourly'] = df['EXITSn_hourly'].fillna(0)
    return df

def time_to_hour(time):

    hour = int(time[:2])
    return hour

import datetime

def reformat_subway_dates(date):

    date_formatted = '20%s-%s-%s' %(date[-2:],date[:2],date[3:5])

    return date_formatted

