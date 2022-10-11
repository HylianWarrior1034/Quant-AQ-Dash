from ast import Pass
import quantaq
import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.lines import Line2D
import time 
from pathlib import Path
import plotly.express as px

client = quantaq.QuantAQAPIClient('7NVIKOGY0DGZBPWJ2YJJ2KR9')

#this function generates graph, csv, and wind graphs by the week based on given inputs
def generate(start_date, end_date, device):

    date_range = pd.date_range(start = start_date, end = end_date)
    date_range_list = date_range.strftime("%Y-%m-%d").tolist()
    print(len(date_range_list))

    k = 0

    while k < len(date_range_list):
        pm1 = []
        pm25 = [] 
        pm10 = []
        o3 = [] 
        winddir = []
        windspeed = [] 
        co = [] 
        temperature = []
        timestamp = []
        rh = []
        no2 = [] 

        datas = []

        start = date_range_list[k]
        end = date_range_list[k+6]

        date_range = pd.date_range(start = start, end = end)

        for day in date_range:
            data = client.data.bydate(sn = device, date = str(day.date()))
            datas.append(data)

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

        # convert C to F 
        df['temperature'] = df['temperature'].apply(lambda x: (x * 9/5) + 32)


        # this loop creates dataframes for each day within the date range for each of the modules in the inputted list 
        temp = []
        df_day = [] 

        # split by days 
        for i in range(len(df)-1):
            if df['timestamp'].values[i][:11] != df['timestamp'].values[i+1][:11]:
                temp.append(i+1)

        a = 0
        for i in temp: 
            b = i
            df_day.append(df.iloc[a:b])
            a = b
        df_day.append(df.iloc[temp[-1]:])

        # split by hour 
        df_hours = []
        for day in df_day: 
            temp_day = []
            j = 0
            while j < len(day)-1: 
                hour1 = day['timestamp'].values[j][-5:-3]
                hour2 = day['timestamp'].values[j+1][-5:-3]
                if hour1 != hour2:
                    temp_day.append(j+1)
                    j += 30
                else: 
                    j += 1

            a = 0 
            for i in temp_day: 
                b = i
                df_hours.append(day.iloc[a:b])
                a = b
            df_hours.append(day.iloc[temp_day[-1]:])

        k += 7

        scatter_plot(df_hours, start_date = start, end_date = end)
        generate_hourlycsv(df_hours, start_date = start, end_date = end, directory = f"plotly\{start_date}_7_{device}.csv")
        mean_plot(df_hours, start_date = start, end_date = end)

        print('{} processed'.format(start))


def csv_call(csv_path): 
    df = pd.read_csv(csv_path)

    start_date = df['timestamp'].values[0][:10]
    end_date = df['timestamp'].values[-1][:10]

    temp = []
    df_day = [] 

    for i in range(len(df)-1):
        if df['timestamp'].values[i][-5:] == "00:00" and df['timestamp'].values[i+1][-5:] != "00:00":
            temp.append(i)

    a = 0

    for i in temp[1:]: 
        b = i
        df_day.append(df.iloc[a:b, :])
        a = b

    df_day.append(df.iloc[temp[-1]:, :])

    # split by hour 
    df_hours = []
    for day in df_day: 
        temp_day = []
        j = 0
        while j < len(day)-1: 
            hour1 = day['timestamp'].values[j][-5:-3]
            hour2 = day['timestamp'].values[j+1][-5:-3]
            if hour1 != hour2:
                temp_day.append(j+1)
                j += 30
            else: 
                j += 1

        a = 0 
        for i in temp_day: 
            b = i
            df_hours.append(day.iloc[a:b, :])
            a = b
        df_hours.append(day.iloc[temp_day[-1]:, :])

    return [df_day, df_hours, start_date, end_date]


def api_call(start_date, end_date, dev = 'MOD-00022'):

    date_range = pd.date_range(start = start_date, end = end_date)
    # date_range_list = date_range.strftime("%Y-%m-%d").tolist()

    pm1 = []
    pm25 = [] 
    pm10 = []
    o3 = [] 
    winddir = []
    windspeed = [] 
    co = [] 
    temperature = []
    timestamp = []
    rh = []
    no2 = [] 

    datas = []

    for i, day in enumerate(date_range):
        data = client.data.bydate(sn = dev, date = str(day.date()))
        datas.append(data)
        print(f'day {i + 1} processed')

    print(f'data downloaded for {dev}')

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

    # convert C to F 
    df['temperature'] = df['temperature'].apply(lambda x: (x * 9/5) + 32)


    # this loop creates dataframes for each day within the date range for each of the modules in the inputted list 
    temp = []
    df_day = [] 

    # split by days 
    for i in range(len(df)-1):
        if df['timestamp'].values[i][:11] != df['timestamp'].values[i+1][:11]:
            temp.append(i+1)

    a = 0
    for i in temp: 
        b = i
        df_day.append(df.iloc[a:b])
        a = b
    df_day.append(df.iloc[temp[-1]:])

    # split by hour 
    df_hours = []
    for day in df_day: 
        temp_day = []
        j = 0
        while j < len(day)-1: 
            hour1 = day['timestamp'].values[j][-5:-3]
            hour2 = day['timestamp'].values[j+1][-5:-3]
            if hour1 != hour2:
                temp_day.append(j+1)
                j += 30
            else: 
                j += 1

        a = 0 
        for i in temp_day: 
            b = i
            df_hours.append(day.iloc[a:b])
            a = b
        df_hours.append(day.iloc[temp_day[-1]:])

    return [df_day, df_hours]

# function to find the average wind vector 
def wind_average(df):
    # every element in wind_averages is a list of two elements containing average wind speed and direction
    wind_averages = []    

    for i in df: 
        wind_dir = i['wind direction'].tolist()
        wind_speed = i['wind speed'].tolist()

        north_vector = [round(x[1] * math.sin(x[0] * math.pi/180), 3) for x in list(zip(wind_dir, wind_speed))]
        east_vector = [round(x[1] * math.cos(x[0] * math.pi/180), 3) for x in list(zip(wind_dir, wind_speed))]

        # contains normalized north and east vectors 
        normal_vectors = [round(sum(north_vector)/len(north_vector), 3), round(sum(east_vector)/len(east_vector), 3)]

        # finding the average wind speed and wind direction 
        avg_speed = round(math.sqrt(normal_vectors[0]**2 + normal_vectors[1]**2), 3) 
        
        if normal_vectors[0] == 0.000: 
            angle = math.atan(normal_vectors[1] / (0.001 + normal_vectors[0]))
        else: 
            angle = math.atan(normal_vectors[1] / + normal_vectors[0])
        
        # add pi radians to the angle if the north vector is negative
        if normal_vectors[0] < 0: 
            angle += math.pi 


        avg_direction = round(angle * 180/math.pi, 3)
        if avg_direction < 0: 
            avg_direction += 360

        wind_averages.append([avg_speed, round(avg_direction, 3)])
    return wind_averages

# find means 
def means(df): 
    wind_speed = [wind_average(df)[i][0] for i in range(len(df))]
    wind_dir = [wind_average(df)[i][1] for i in range(len(df))]
    mean_pm1 = []
    mean_pm25 = []
    mean_pm10 = []
    mean_o3 = []
    mean_no2 = []
    mean_co = []
    mean_temp = []
    mean_rh = []

    for df1 in df: 
        mean_pm1.append(round(df1['pm1'].mean(), 3))
        mean_pm25.append(round(df1['pm25'].mean(), 3))
        mean_pm10.append(round(df1['pm10'].mean(), 3))
        mean_o3.append(round(df1['o3'].mean(), 3))
        mean_no2.append(round(df1['no2'].mean(), 3))
        mean_co.append(round(df1['co'].mean(), 3))
        mean_temp.append(round(df1['temperature'].mean(), 3))
        mean_rh.append(round(df1['rh'].mean(), 3))

    return [mean_pm1, mean_pm25, mean_pm10, mean_o3, mean_no2, mean_co, mean_temp, mean_rh, wind_speed, wind_dir]

def quantiles25(df): 
    top10 = (round(df['pm25'].quantile(0.9), 3))
    top25 = (round(df['pm25'].quantile(0.75), 3))
    mean = (round(df['pm25'].mean(), 3))
    top75 = (round(df['pm25'].quantile(0.25), 3))
    top90 = (round(df['pm25'].quantile(0.1), 3))
    median = (round(df['pm25'].median(), 3))
    
    return([top10, top25, mean, top75, top90, median])

def scatter_plot(df_hour, start_date, end_date): 
    means1 = means(df_hour)
    # choice0 = pm2.5, choice1 = pm10, choice2 = both 

    # HERES ALL THE PARAMETERS STATED OUT FOR THE SCATTER PLOT 
    p = px.scatter(x = means1[9], y = means1[2], color = means1[8], custom_data = [means1[8]], \
        title = f'PM10 Conc. Based on Wind Velocity {start_date}~{end_date}').update_layout(xaxis_title = 'Wind Direction', \
            yaxis_title = 'PM10 Levels', xaxis = dict(tickmode = 'array', tickvals = [0, 45, 90, 135, 180, 225, 270, 315, 360], \
                ticktext = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'N']), xaxis_range = [0, 360], yaxis = dict(rangemode = 'tozero'), \
                    coloraxis_colorbar_title_text = 'Wind Velocity')

    # update the hover descriptions 
    p.update_traces(hovertemplate = 'Wind Direction = %{x}°<br>PM10 Levels = %{y}<br>Wind Speed = %{customdata[0]} m/s')
    p.show()
    # p.write_html(r'C:\Users\Dm101\Desktop\plotly\index.html', include_plotlyjs = 'cdn')

    #!!! fix the axes plot sooner or later
    # fixed hahahahahahaha
    # if choice != 0:
    #     points = plt.scatter(means1[9], means1[2], c = means1[8], cmap= plt.get_cmap('cool'))
    #     plt.clf()
    #     plt.colorbar(points)        
    #     p = sns.scatterplot(x = means1[9], y = means1[2], hue = means1[8], palette = plt.get_cmap('cool'))
    #     p.set_xlabel('Wind Direction')
    #     p.set_ylabel('PM10 Levels')
    #     p.set_xlim(0, 360)
    #     p.set_xticks([x for x in range(0, 360, 60)] + [360])
    #     p.set_title('PM10 Conc. Based on Wind Velocity {}~{}'.format(start_date, end_date))
    #     plt.legend(title = 'Wind Speed')
    #     plt.savefig('Louisiana\Weekly Dataset Graphs\PM10 Wind\{}_{}_MOD-00022_PM10'.format(start_date, 7))
    #     # plt.show()

    # if choice != 1: 
    #     points = plt.scatter(means1[9], means1[1], c = means1[8], cmap= plt.get_cmap('cool'))
    #     plt.clf()
    #     plt.colorbar(points)        
    #     p = sns.scatterplot(x = means1[9], y = means1[1], hue = means1[8], palette = plt.get_cmap('cool'))
    #     p.set_xlabel('Wind Direction')
    #     p.set_ylabel('PM2.5 Levels')
    #     p.set_xlim(0, 360)
    #     p.set_xticks([x for x in range(0, 360, 60)] + [360])
    #     p.set_title('PM2.5 Conc. Based on Wind Velocity {}~{}'.format(start_date, end_date))
    #     plt.legend(title = 'Wind Speed')
    #     plt.savefig('Louisiana\Weekly Dataset Graphs\PM2.5 Wind\{}_{}_MOD-00022_PM25'.format(start_date, 7))

def hourplot(df_hour, start_date, end_date): 
    list_by_hour = []
    #create list of lists
    lists_temp = [[] for _ in range(24)]
    for hour in df_hour: 
        #find which hour the hour dataframe is currently in
        index_hour = int(hour['timestamp'].values[0][-5:-3]) # means first row of the timestamp column
        lists_temp[index_hour].append(hour)
    

    for list_hour in lists_temp: 
        s1 = [] 
        s1 = pd.concat(list_hour, ignore_index = True)
        list_by_hour.append(s1)

    top10 = []
    top25 = []
    mean = []
    top75 = []
    top90 = []
    median = []

    for hour in list_by_hour: 
        quantiles1 = quantiles25(hour)
        top10.append(quantiles1[0])
        top25.append(quantiles1[1])
        mean.append(quantiles1[2])
        top75.append(quantiles1[3])
        top90.append(quantiles1[4])
        median.append(quantiles1[5])

    fig, ax = plt.subplots(1, figsize = (13, 13))
    x_coord = range(24)

    ax.plot(x_coord, top10)
    ax.plot(x_coord, top25)
    ax.plot(x_coord, mean)
    ax.plot(x_coord, median)
    ax.plot(x_coord, top75)
    ax.plot(x_coord, top90)

    ax.fill_between(x_coord, top25, top75, color = 'lightblue')
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 4), range(0, 24, 4))
    ax.set_xticks(range(23))

    # just the usuals 
    ax.set_ylabel('PM 2.5 Concentration')
    ax.set_xlabel('Hour of Day')
    ax.set_title('PM 2.5 Concentration Each Hour from {}~{}'.format(start_date, end_date))

    ax.legend(['top 10th', 'top 25th', 'mean', 'median', 'botton 25th', 'bottom 10th'])
    plt.show()

def color_hourplot(df_hour, start_date, end_date): 
    list_by_hour = []
    #create list of lists
    lists_temp = [[] for _ in range(24)]
    for hour in df_hour: 
        #find which hour the hour dataframe is currently in
        index_hour = int(hour['timestamp'].values[0][-5:-3]) # means first row of the timestamp column
        lists_temp[index_hour].append(hour)
    

    for list_hour in lists_temp: 
        s1 = [] 
        s1 = pd.concat(list_hour, ignore_index = True)
        list_by_hour.append(s1)

    top10 = []
    top25 = []
    mean = []
    top75 = []
    top90 = []
    median = []

    for hour in list_by_hour: 
        quantiles1 = quantiles25(hour)
        top10.append(quantiles1[0])
        top25.append(quantiles1[1])
        mean.append(quantiles1[2])
        top75.append(quantiles1[3])
        top90.append(quantiles1[4])
        median.append(quantiles1[5])

    fig, ax = plt.subplots(1, figsize = (13, 13))
    x_coord = range(24)

    # add the background colors for the healthy - unhealthy ranges and add text 
    #healthy 
    ax.axhspan(0, 20, facecolor = 'lime', label = '_nolegend_')
    ax.axhspan(20, 40, facecolor = 'yellow', label = '_nolegend_')
    ax.axhspan(40, 55, facecolor = 'orange', label = '_nolegend_')
    ax.axhspan(55, 150, facecolor = 'red', label = '_nolegend_')
    ax.axhspan(150, 1000, facecolor = 'purple', label = '_nolegend_')

    custom_lines = [Line2D([0], [0], color='lime', lw=4),
                    Line2D([0], [0], color='yellow', lw=4),
                    Line2D([0], [0], color='orange', lw=4),
                    Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='purple', lw=4)]

    custom_legend = ax.legend(custom_lines, ['healthy', '1', '2', '3', '4'], bbox_to_anchor = (1, 1), loc='upper left', ncol=1)
    plt.gca().add_artist(custom_legend)

    ax.plot(x_coord, top10, linestyle = 'dotted', color = 'black')
    ax.plot(x_coord, top25, linestyle = 'solid', color = 'black')
    ax.plot(x_coord, top75, linestyle = 'solid', color = 'black')
    ax.plot(x_coord, top90, linestyle = 'dashed', color = 'black')

    ax.fill_between(x_coord, top25, top75, color = 'grey')
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 4), range(0, 24, 4))
    ax.set_xticks(range(23))

    # just the usuals 
    ax.set_ylabel('PM 2.5 Concentration')
    ax.set_xlabel('Hour of Day')
    ax.set_title('PM 2.5 Concentration Each Hour from {}~{}'.format(start_date, end_date))

    ax.set_ylim(0, max(top90) * 1.5)

    ax.legend(['top 10th', 'top 25th', 'botton 25th', 'bottom 10th'], loc = 'upper left')
    plt.show()

def color_dayplot(df_day, start_date, end_date):
    start_month = start_date[5:7]
    end_month = end_date[5:7]

    list_by_day = []
    for i in range(7):
        day = i
        s1 = df_day[day]
        for j in range(len(df_day)//7-1): 
            day += 7
            s2 = df_day[day]
            s1 = pd.concat([s1, s2], ignore_index=True)
        list_by_day.append(s1)

    top10 = []
    top25 = []
    mean = []
    top75 = []
    top90 = []
    median = []

    for day in list_by_day: 
        quantiles1 = quantiles25(day)
        top10.append(quantiles1[0])
        top25.append(quantiles1[1])
        mean.append(quantiles1[2])
        top75.append(quantiles1[3])
        top90.append(quantiles1[4])
        median.append(quantiles1[5])

    fig, ax = plt.subplots(1, figsize = (13, 13))
    x_coord = range(7)

    # add the background colors for the healthy - unhealthy ranges and add text 
    #healthy 
    ax.axhspan(0, 20, facecolor = 'lime', label = '_nolegend_')
    ax.axhspan(20, 40, facecolor = 'yellow', label = '_nolegend_')
    ax.axhspan(40, 55, facecolor = 'orange', label = '_nolegend_')
    ax.axhspan(55, 150, facecolor = 'red', label = '_nolegend_')
    ax.axhspan(150, 1000, facecolor = 'purple', label = '_nolegend_')

    custom_lines = [Line2D([0], [0], color='lime', lw=4),
                    Line2D([0], [0], color='yellow', lw=4),
                    Line2D([0], [0], color='orange', lw=4),
                    Line2D([0], [0], color='red', lw=4),
                    Line2D([0], [0], color='purple', lw=4)]

    custom_legend = ax.legend(custom_lines, ['healthy', '1', '2', '3', '4'], bbox_to_anchor = (1, 1), loc='upper left', ncol=1)
    plt.gca().add_artist(custom_legend)

    ax.plot(x_coord, top10, linestyle = 'dotted', color = 'black')
    ax.plot(x_coord, top25, linestyle = 'solid', color = 'black')
    ax.plot(x_coord, top75, linestyle = 'solid', color = 'black')
    ax.plot(x_coord, top90, linestyle = 'dashed', color = 'black')

    ax.fill_between(x_coord, top25, top75, color = 'grey')
    ax.set_xlim(0, 7)
    ax.set_xticks(range(0, 7), ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

    # just the usuals 
    ax.set_ylabel('PM 2.5 Concentration')
    ax.set_xlabel('Day of Week')
    ax.set_title('Avg. PM 2.5 Concentration Each Day of Week from {}~{}'.format(start_month, end_month))

    ax.set_xlim(0, 6)
    ax.set_ylim(0, max(top10) * 1.25)
    ax.legend(['top 10th', 'top 25th', 'botton 25th', 'bottom 10th'])

    plt.show()

def dayplot(df_day):
    pass 

def mean_plot(df, start_date, end_date): 
    means1 = means(df)
    pm1 = means1[0]
    pm25 = means1[1]
    pm10 = means1[2]
    o3 = means1[3]
    no2 = means1[4]
    co = means1[5]
    temperature = means1[6]
    rh = means1[7]
    wind_speed = means1[8]
    wind_direction = means1[9]

    fig, ax = plt.subplots(5, figsize = (13,13))

    date_range = pd.date_range(start = start_date, end = end_date)
    date_range_list = date_range.strftime("%Y-%m-%d").tolist()

    # set up x-coords 
    xcoords = range(len(df))

    # Parameters for graph 0
    ax[0].plot(xcoords, temperature, color = 'red')
    ax[0].set_ylabel('Temperature (F)', color = 'red')
    ax[0].tick_params('y', colors = 'red')
    ax[0].set_xticks(np.arange(0, len(df), 6))
    ax[0].axes.xaxis.set_ticklabels([])

    ax[0].grid(color='#EEEEEE', axis = 'x')
    ax[0].set_axisbelow(True)  

    # Overlap another graph on graph 0 to add a secondary y-axis on the right
    ax_dup0 = ax[0].twinx()
    ax_dup0.plot(xcoords, rh, color = 'lightblue')
    ax_dup0.set_ylabel('RH(%)', color = 'lightblue')
    ax_dup0.tick_params('y', colors = 'lightblue')
    ax_dup0.set_xticks(np.arange(0, len(df), 6))
    ax_dup0.axes.xaxis.set_ticklabels([])

    # ax[0].grid(color='#EEEEEE')
    # ax[0].set_axisbelow(True)  
    
    # Parameters for graph 1 
    ax[1].scatter(xcoords, wind_direction, s = 10, color = 'blue')
    ax[1].set_ylabel('Wind Dir. (degrees)', color = 'blue')
    ax[1].set_ylim(0, 380)
    ax[1].set_yticks(np.arange(0, 370, 90))
    ax[1].tick_params('y', colors = 'blue')
    ax[1].set_xticks(np.arange(0, len(df), 6))
    ax[1].axes.xaxis.set_ticklabels([])

    ax[1].grid(color='#EEEEEE', axis = 'x')
    ax[1].set_axisbelow(True)  

    ax_dup = ax[1].twinx()
    ax_dup.plot(xcoords, wind_speed, 'lightcoral')
    ax_dup.set_xticks(np.arange(0, len(df), 6))
    ax_dup.axes.xaxis.set_ticklabels([])
    ax_dup.set_ylabel('Wind Speed (m/s)', color = 'lightcoral')
    ax_dup.set_ylim(ymin = 0)
    ax_dup.tick_params('y', colors = 'lightcoral')

    # Parameters for graph 2
    ax[2].plot(xcoords, o3, color = 'orange')
    ax[2].set_ylabel('O3 (pbb)', color = 'orange')
    ax[2].tick_params('y', colors = 'orange')
    ax[2].set_xticks(np.arange(0, len(df), 6))
    ax[2].axes.xaxis.set_ticklabels([])
    ax[2].set_ylim(ymin = 0)

    ax[2].grid(color='#EEEEEE', axis = 'x')
    ax[2].set_axisbelow(True)  

    ax_dup2 = ax[2].twinx()
    ax_dup2.plot(xcoords, no2, color = 'forestgreen')
    ax_dup2.set_ylabel('NO2 (pbb)', color = 'forestgreen')
    ax_dup2.tick_params('y', colors = 'forestgreen')
    ax_dup2.set_xticks(np.arange(0, len(df), 6))
    ax_dup2.axes.xaxis.set_ticklabels([])
    # ax_dup2.grid(color='#a6d0a6')
    # ax_dup2.set_axisbelow(True)  

    # Parameters for graph 3 
    ax[3].plot(xcoords, co, color = 'maroon')
    ax[3].set_ylabel('CO (pbb)', color = 'maroon')
    ax[3].tick_params('y', colors = 'maroon')
    ax[3].set_xticks(np.arange(0, len(df), 6))
    ax[3].axes.xaxis.set_ticklabels([])
    ax[3].set_ylim(ymin = 0)
    ax[3].grid(color='#EEEEEE', axis = 'x')
    ax[3].set_axisbelow(True)  
    
    # Parameters for graph 4
    y_temp = -10 * np.ones(len(xcoords))
    ax[4].plot(xcoords, pm1, color = 'lightgray')
    ax[4].plot(xcoords, pm25, color = 'grey')
    ax[4].plot(xcoords, y_temp, color = 'black')
    ax[4].set_ylim(ymin = 0)

    ax_dup4 = ax[4].twinx()
    ax_dup4.plot(xcoords, pm10, color = 'black')
    ax_dup4.set_ylabel('PM10 Levels (µg/m³)', color = 'black')
    ax_dup4.tick_params('y', colors = 'black')
    ax_dup4.set_xticks(np.arange(0, len(df), 6))
    ax_dup4.axes.xaxis.set_ticklabels([])
    ax_dup4.set_ylim(ymin = 0)

    ax[4].set_ylabel('PM1, PM2.5 Levels (µg/m³)')
    ax[4].legend(['PM1', 'PM2.5', 'PM10'],  bbox_to_anchor = (1.11, 1.15), loc = 'lower right')
    ax[4].grid(color='#EEEEEE', axis = 'x')
    ax[4].set_axisbelow(True)  
    ax[4].set_xticks(np.arange(0, len(df), 24), date_range_list, size = 9)
    ax[4].set_xticks(np.arange(0, len(df), 6))

    # shared features for all 5 graphs
    fig.suptitle('Data Observed by MOD-AIR 022 {}~{}'.format(start_date, end_date))
    plt.setp(ax, xlim = (0, len(df)-1))
    plt.xlabel('Date')
    plt.savefig('Louisiana\Weekly Dataset Graphs\Cumulative\{}_{}_MOD-00022_Cumulative'.format(start_date, 7))
    # plt.show()

def generate_hourlycsv(df, start_date, end_date, directory): 
    means1 = means(df)
    mean_pm1 = means1[0]
    mean_pm25 = means1[1]
    mean_pm10 = means1[2]
    mean_o3 = means1[3]
    mean_no2 = means1[4]
    mean_co = means1[5]
    mean_temp = means1[6]
    mean_rh = means1[7]
    wind_speed = means1[8]
    wind_dir = means1[9]

    date_range = pd.date_range(start = start_date, end = end_date)
    date_range_list = date_range.strftime("%Y-%m-%d").tolist()

    ###################THE PART WHERE I DETERMINE THE LENGTH OF INDEX##############################
    hours = []

    # find the index where day changes 
    for hour in df: 
        hours.append('{}:00'.format(hour['timestamp'].values[0][-5:-3]))

    index = [] 

    for i in range(len(hours)): 
        if hours[i] == '00:00':
            index.append(i)

    timestamp2 = [] 

    # split the indeces 
    for i in range(len(date_range_list)-1): 
        for j in range(index[i+1]-index[i]): 
            timestamp2.append(date_range_list[i])
    
    for i in range(len(hours) - index[-1]):
        timestamp2.append(date_range_list[-1])
    ################################################################################################

    df_mean = pd.DataFrame({
    'Date': timestamp2,
    'Hour': hours,
    'pm1': mean_pm1,
    'pm2.5': mean_pm25,
    'pm10': mean_pm10,
    'o3': mean_o3,
    'no2': mean_no2, 
    'co': mean_co,
    'temp': mean_temp,
    'RH': mean_rh, 
    'Wind Speed': wind_speed,
    'Wind Dir': wind_dir
    }, index = range(len(mean_pm1)))

    df_mean.index.name = 'Index'
    
    filepath = Path(directory)  

    df_mean.to_csv(filepath)

def generate_hourlycsv2(df, start_date, end_date, directory): 
    means1 = means(df)
    mean_pm1 = means1[0]
    mean_pm25 = means1[1]
    mean_pm10 = means1[2]
    mean_o3 = means1[3]
    mean_no2 = means1[4]
    mean_co = means1[5]
    mean_temp = means1[6]
    mean_rh = means1[7]
    wind_speed = means1[8]
    wind_dir = means1[9]

    date_range = pd.date_range(start = start_date, end = end_date)
    date_range_list = date_range.strftime("%Y-%m-%d").tolist()

    ###################THE PART WHERE I DETERMINE THE LENGTH OF INDEX##############################
    hours = []

    # find the index where day changes 
    for hour in df: 
        hours.append('{}:00'.format(hour['timestamp'].values[0][-5:-3]))

    index = [] 

    for i in range(len(hours)): 
        if hours[i] == '00:00':
            index.append(i)

    timestamp2 = [] 

    # split the indeces 
    for i in range(len(date_range_list)-1): 
        for j in range(index[i+1]-index[i]): 
            timestamp2.append(date_range_list[i])
    
    for i in range(len(hours) - index[-1]):
        timestamp2.append(date_range_list[-1])
    ################################################################################################

    df_mean = pd.DataFrame({
    'Date': timestamp2,
    'Hour': hours,
    'pm1': mean_pm1,
    'pm2.5': mean_pm25,
    'pm10': mean_pm10,
    'o3': mean_o3,
    'no2': mean_no2, 
    'co': mean_co,
    'temp': mean_temp,
    'RH': mean_rh, 
    'Wind Speed': wind_speed,
    'Wind Dir': wind_dir
    }, index = range(len(mean_pm1)))

    df_mean.index.name = 'Index'
    
    filepath = Path(directory)  

    df_mean.to_csv(filepath)

if __name__ == '__main__': 
    # # this api_call step takes the longest, 7 seconds per day to call
    # # splicing the data and generating the graphs take like 2 seconds max

    # dfs = api_call('2022-08-14', '2022-08-20')
    # print(dfs[1])
    # generate_hourlycsv(dfs[1], '2022-08-14', '2022-08-20', directory = f"Quant_aq_MOD-00022.csv")

    dfs = csv_call('Quant-aq_MOD-00022.csv')
    # generate_hourlycsv(dfs[1], '2022-08-07', '2022-08-13', directory = f"Louisiana\Weekly Dataset Graphs\CSVs\2022-08-07_7_MOD-00022.csv")

    # start_date = dfs[2]
    # end_date = dfs[3]
    
    # df_day = dfs[0]
    df_hours = dfs[1]


    # color_dayplot(df_day, start_date, end_date)

    # generate('2022-08-14', '2022-08-20', 'MOD-00022')
    scatter_plot(dfs[1], '2022-08-14', '2022-08-20')
