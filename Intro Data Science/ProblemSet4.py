#################################################################
##################### Problem Set #4 ############################
#################################################################
from ggplot import *
import pandas as pd
import os
import datetime

cwd = os.getcwd()
os.chdir('%s\\Intro Data Science\\Problem Set 4' %(cwd))

df = pd.read_csv('turnstile_data_master_with_weather.csv')

def plot_weather_data1(turnstile_weather):

    turnstile_weather['DAY'] = [datetime.datetime.strptime(X,'%Y-%m-%d').strftime('%A') for X in turnstile_weather.DATEn]
    plot = ggplot(turnstile_weather, aes(x='Hour')) + geom_histogram(binwidth=1)

    return plot

def plot_weather_data2(turnstile_weather):

    df = turnstile_weather
    df.is_copy = False
    df['DAY'] = [datetime.datetime.strptime(X,'%Y-%m-%d').strftime('%A') for X in df.DATEn]
    plot = ggplot(df, aes(x='DAY')) + geom_histogram(binwidth=1)

    return plot