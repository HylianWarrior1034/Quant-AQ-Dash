from time import time
import quantaq
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np 
from tkinter import *

client = quantaq.QuantAQAPIClient('7NVIKOGY0DGZBPWJ2YJJ2KR9')

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """

    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print('Done!')


def download_data(start_day, end_day, dev):
    date_range = pd.date_range(start = start_day, end = end_day)[:-1]
    datas = []
    for i, day in enumerate(date_range):
        data = client.data.bydate(sn = dev, date = str(day.date()))
        datas.append(data)
        printProgressBar (i+1, len(date_range), prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r")

    pm1 = []
    pm10 = []
    pm25 = []
    o3 = [] 
    winddir = []
    windspeed = [] 
    co = [] 
    temperature = []
    timestamp = []
    rh = []
    no2 = [] 
    timestamp = []

    if dev in ['MOD-00021', 'MOD-00022', 'MOD-00047', 'MOD-00048', 'MOD-00049', 'MOD-00050', 'MOD-00051', \
        'MOD-00052', 'MOD-00053', 'MOD-00054']:
        # append only the wanted data from the obtained quant-aq data
        for data in datas: 
            for entry in data: 
                pm1.append(entry['pm1'])
                pm10.append(entry['pm10'])
                pm25.append(entry['pm25'])
                o3.append(entry['o3'])
                no2.append(entry['no2'])
                co.append(entry['co'])
                temperature.append(entry['temp'])
                rh.append(entry['rh'])
                winddir.append(entry['met']['wd'])
                windspeed.append(entry['met']['ws'])     
                timestamp.append(entry['timestamp'][:-3])

        # create a pandas dataframe from the lists 
        df = pd.DataFrame({
            'timestamp' : timestamp,
            'pm1': pm1,
            'pm10': pm10,
            'pm25': pm25,
            'o3': o3,
            'no2': no2,
            'co': co, 
            'temperature': temperature,
            'rh': rh, 
            'wind direction': winddir,
            'wind speed': windspeed
        }, index = range(len(timestamp)))

    else: 
        for data in datas: 
            for entry in data: 
                pm1.append(entry['pm1'])
                pm10.append(entry['pm10'])
                pm25.append(entry['pm25']) 
                timestamp.append(entry['timestamp'][:-3])

        # create a pandas dataframe from the lists 
        df = pd.DataFrame({
            'timestamp' : timestamp,
            'pm1': pm1,
            'pm10': pm10,
            'pm25': pm25,
        }, index = range(len(timestamp)))

    return df


# this is the only method that should be called, not the other one 
def download_csv(start_month, end_month, directory, devs): 
    for dev in devs: 
        print(f'Processing {dev}...')
        df = pd.DataFrame() 
        monthly = pd.date_range(start=f'{start_month}/1/2022', end=f'{end_month}/1/2022', freq='MS')  
        
        for i in range(len(monthly)-1): 
            month_df = download_data(monthly[i], monthly[i+1], dev)

            df = pd.concat([df, month_df])
            
            month_df.to_csv(f'{directory}/{dev}/Quant-aq_{dev}_{str(monthly[i].date())[5:7]}.csv'.format(directory, dev))

        year_df = pd.read_csv(f'{directory}/{dev}/Quant-aq_{dev}_year.csv')
        df = pd.concat([year_df, df])
        df.to_csv(f'{directory}/{dev}/Quant-aq_{dev}_year.csv')

import os

# this bottom segment of code was used to concatenate all the sub-csv's into one gigantic one (that's 6 million lines long)

###################################################################################
def concatenate():
    df = pd.DataFrame()

    for directory in os.scandir(r"data"): 
        name = directory.path[5:]
        for file in os.scandir(directory.path): 
            if 'year' in file.path: 
                df_sub = pd.read_csv(file.path).iloc[:, 1:]
                # if 'MOD-00021' or 'MOD-00022' in file.path:
                #     df_sub = df_sub[['timestamp', 'pm1', 'pm10', 'pm25']]

                df_sub['module'] = [name] * len(df_sub['timestamp'])
                df = pd.concat([df, df_sub])

    # df = pd.concat([df, pd.read_csv("data\MOD-00047\Quant-aq_MOD-00047_09.csv"), pd.read_csv("data\MOD-00048\Quant-aq_MOD-00048_09.csv"), \
    #     pd.read_csv("data\MOD-00050\Quant-aq_MOD-00050_09.csv"), pd.read_csv("data\MOD-00051\Quant-aq_MOD-00051_09.csv"), \
    #         pd.read_csv("data\MOD-00052\Quant-aq_MOD-00052_09.csv"), pd.read_csv("data\MOD-00053\Quant-aq_MOD-00053_09.csv"), \
    #             pd.read_csv("data\MOD-00054\Quant-aq_MOD-00054_09.csv")])

    df = df.reset_index(drop = True)
    df = df.drop(["Unnamed: 0"], axis = 1)

    df.to_csv('Quant-aq_2022_test.csv')

###################################################################################

# this is just to get the exact coordinates for all the sensors 

# import json 

# coordinates = {}

# for directory in os.scandir(r"data"): 

#     name = directory.path[5:]
#     data = client.data.bydate(sn = name, date = '2022-09-07')[0]
#     coordinates[name] = data['geo']

# print(json.dumps(coordinates))
# with open('coords.txt', 'w+') as f:
#     f.write(json.dumps(coordinates)) 

       


if __name__ == "__main__": 
    download_csv(9, 10, r"C:\Users\Dm101\Desktop\plotly\data", [
        "MOD-00021", 
        "MOD-00022", 
        "MOD-00047",
        "MOD-00048",
        "MOD-00049",
        "MOD-00051",
        "MOD-00052",
        "MOD-00053",
        "MOD-00054",
        "MOD-PM-00194",
        "MOD-PM-00195",
        "MOD-PM-00196",
        "MOD-PM-00197",
        "MOD-PM-00270",
        "MOD-PM-00273",
        "MOD-PM-00274",
        "MOD-PM-00275", 
        "MOD-PM-00276", 
        "MOD-PM-00277",
        ])
    concatenate()


# code block to add all the new shit 

# start_day = '2022-09-14'
# end_day = '2022-10-01'

# devs = ['MOD-00047', 'MOD-00048', 'MOD-00050', 'MOD-00051', 'MOD-00052', 'MOD-00053', 'MOD-00054']

# date_range = pd.date_range(start = start_day, end = end_day)
# for dev in devs: 
#     datas = []
#     for i, day in enumerate(date_range):
#         data = client.data.bydate(sn = dev, date = str(day.date()))
#         datas.append(data)
#         printProgressBar (i+1, len(date_range), prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r")

#     pm1 = []
#     pm10 = []
#     pm25 = []
#     o3 = [] 
#     winddir = []
#     windspeed = [] 
#     co = [] 
#     temperature = []
#     timestamp = []
#     rh = []
#     no2 = [] 
#     timestamp = []

#     # append only the wanted data from the obtained quant-aq data
#     for data in datas: 
#         for entry in data: 
#             pm1.append(entry['pm1'])
#             pm10.append(entry['pm10'])
#             pm25.append(entry['pm25'])
#             o3.append(entry['o3'])
#             no2.append(entry['no2'])
#             co.append(entry['co'])
#             temperature.append(entry['temp'])
#             rh.append(entry['rh'])
#             winddir.append(entry['met']['wd'])
#             windspeed.append(entry['met']['ws'])   
#             # The timestamp for the new sensors have an extra character at the end... don't ask me why this makes it really annoying  
#             timestamp.append(entry['timestamp'][:-3])

#     # create a pandas dataframe from the lists 
#     df = pd.DataFrame({
#         'timestamp' : timestamp,
#         'pm1': pm1,
#         'pm10': pm10,
#         'pm25': pm25,
#         'o3': o3,
#         'no2': no2,
#         'co': co, 
#         'temperature': temperature,
#         'rh': rh, 
#         'wind direction': winddir,
#         'wind speed': windspeed
#     }, index = range(len(timestamp)))

#     df.to_csv(f"data/{dev}/Quant-aq_{dev}_year.csv")
