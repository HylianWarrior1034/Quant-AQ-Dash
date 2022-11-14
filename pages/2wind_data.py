# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import html, dcc, Input, Output, ctx, register_page, callback
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
from datetime import date
import plotly.express as px
import plotly.graph_objects as go
import time 
from flask_caching import Cache
import pandas as pd
import numpy as np 
import functools 
import operator 
import matplotlib.pyplot as plt 
from dask import dataframe as dd

register_page(__name__)

df = pd.read_csv("Quant-aq_2022.csv")
all_sensors = df['module'].unique()

filtered = filter(lambda x: "PM" not in x, all_sensors)

layout = \
    html.Div([
        html.Div([
            html.Div([
                dcc.Dropdown(value = ['MOD-00021'], id='crossfilter-graph-module', multi = True, options = filtered)
            ], className = 'pm-type'),

            html.Div([
                html.Div([
                dcc.RadioItems(['Minute', 'Hour', 'Day'], 'Day', id = 'crossfilter-graph-freq', labelStyle = {'display': 'inline-block', 'marginTop': '5px', 'padding': '10px'})
                ]),

                html.Div([
                    dcc.RadioItems(options = [
                        {'label': 'PM1', 'value': 'pm1'}, 
                        {'label': 'PM2.5', 'value': 'pm25'}, 
                        {'label': 'PM10', 'value': 'pm10'}], 
                        value = "pm1",
                        id = 'particle-type',
                        labelStyle = {'display': 'inline-block', 'marginTop': '5px', 'padding': '10px'})
                ])
            ], className = 'radios', style = {'padding': '0px'}),

            html.Div([
                dcc.Graph(id='main-graph2', style={'height': '65vh'})
            ], className = 'graph'),

        ], className = 'left'),

        html.Div([
            html.Div([
                dcc.DatePickerRange('2022-01-01', '2022-12-31', min_date_allowed = date(2022, 1, 1), max_date_allowed = date(2022, 12, 31), id = 'datepicker')
            ], className = 'daterange'),

            html.Div([
                html.Button("Download Raw Data", id = "btn-download-raw-csv", className = "downloader_btn"),
                dcc.Download(id="download-raw-csv2")
            ], className = 'downloader'),

            html.Div([
                html.Button("Download Averaged Data", id = "btn-download-csv", className = "downloader_btn"),
                dcc.Download(id="download-csv2")
            ], className = 'downloader'),

            dcc.Store(id = 'storage2')
            # html.Div(id = 'my-output')
        ], className = 'right')
    ], className = 'mainframe')

# this is basically storing the modified dataframe (based on input values) into dcc.Store
# dcc.Store does not take the type of dataframe, so the df has to be changed to JSON and back to dataframe whenever we use it
@callback(
    Output("storage2", "data"),
    Input('crossfilter-graph-module', 'value'),
    Input('crossfilter-graph-freq', 'value'),
    Input('particle-type', 'value'),
    Input('datepicker', 'start_date'),
    Input('datepicker', 'end_date'),
)
def store(modules, freq, types, start, end):
    dff = df.loc[df['module'].isin(modules)]

    dff = dff.drop(['Unnamed: 0'], axis = 1)

    date_range = pd.date_range(start = start, end = end)
    date_range_list = date_range.strftime("%Y-%m-%d").tolist()

    if freq == 'Minute':
        dff = dff.loc[dff['timestamp_local'].apply(lambda x:x[:-6]).isin(date_range_list)]

    if freq == 'Hour': 
        # groupby each hour 
        dff['timestamp_local'] = dff['timestamp_local'].apply(lambda x: x[:-3])
        dff = dff.groupby(['timestamp_local', 'module']).mean().reset_index()

        # we only need the "apply" in the next line to drop the hours and to just compare dates
        dff = dff.loc[dff['timestamp_local'].apply(lambda x:x[:-3]).isin(date_range_list)]

    if freq == 'Day': 
        # groupby each day
        dff['timestamp_local'] = dff['timestamp_local'].apply(lambda x: x[:-6])
        dff = dff.groupby(['timestamp_local', 'module']).mean().reset_index() 

        dff = dff.loc[dff['timestamp_local'].isin(date_range_list)]

    return dff.to_json(orient = 'split')

# # Changes the option of sensors based on graph type chosen
# @callback(
#     Output("crossfilter-graph-module", "options"),
#     Input("crossfilter-graph-type", "value")
# )
# def sensor_options(graph_type): 
#     if graph_type == "PM Data":
#         return [{'label': i, 'value': i} for i in all_sensors]

#     if graph_type == 'Wind Data':
#         filtered = filter(lambda x: "PM" not in x, all_sensors)
#         return [{'label':i , 'value' :i} for i in filtered]

# This is the download (raw files) button
# The changed_input = ctx.triggered_id checks if the "download data" button was actually clicked
# without this parameter, the Dash downloads a new CSV everytime one of the input parameters is changed 
@callback(
    Output("download-raw-csv2", "data"),
    Input("btn-download-raw-csv", "n_clicks"),
    Input('crossfilter-graph-module', 'value'),
    Input('particle-type', 'value'),
    Input('datepicker', 'start_date'),
    Input('datepicker', 'end_date'),
)
def download_raw(n_clicks, modules, types, start, end):
    changed_input = ctx.triggered_id

    if changed_input == 'btn-download-raw-csv':
        dff = df[[types] + ['timestamp_local', 'module']]
        dff = dff.loc[dff['module'].isin(modules)]

        date_range = pd.date_range(start = start, end = end)
        date_range_list = date_range.strftime("%Y-%m-%d").tolist()

        dff = dff.loc[dff['timestamp_local'].apply(lambda x:x[:-6]).isin(date_range_list)]
        return dcc.send_data_frame(dff.to_csv, "customRaw.csv")    

# This is the download button
# The changed_input = ctx.triggered_id checks if the "download data" button was actually clicked
# without this parameter, the Dash downloads a new CSV everytime one of the input parameters is changed 
@callback(
    Output("download-csv2", "data"),
    # Output("my-output", "children"),
    Input("btn-download-csv", "n_clicks"),  
    Input("storage2", "data"),
    prevent_initial_call = True,
)
def download(n_clicks, dff_json):
    changed_input = ctx.triggered_id

    if changed_input == 'btn-download-csv':
        dff = pd.read_json(dff_json, orient = "split")
        dff = dff.sort_values(['module', 'timestamp_local'], ascending = [True, True]).reset_index(drop = True)
        return dcc.send_data_frame(dff.to_csv, "customData.csv")

# # Changes the checkbox parameters based on graph types chosen
# @callback(
#     Output("particle-type", "options"),
#     Input("crossfilter-graph-type", "value"),
#     Input("particle-type", "value")
# )
# def select_particle_type(graph_type, value):
#     if graph_type == 'PM Data':
#         return [{'label': 'PM1', 'value': 'pm1'}, {'label': 'PM2.5', 'value': 'pm25'}, {'label': 'PM10', 'value': 'pm10'}]

#     if graph_type == 'Wind Data':
#         # This is only a function to limit the number of checked boxes on Wind Data to ONE ONLY 
#         options = [
#             {'label': 'PM1', 'value': 'pm1'}, 
#             {'label': 'PM2.5', 'value': 'pm25'}, 
#             {'label': 'PM10', 'value': 'pm10'}
#         ]
#         if len(value) >= 1:
#             options = [
#                 {
#                     "label": option["label"],
#                     "value": option["value"],
#                     "disabled": option["value"] not in value,
#                 }
#                 for option in options
#             ]
        
#         return options

# This is only a function to limit the number of checked boxes on Wind Data to ONE ONLY 
# @app.callback(
#     Output("particle-type", "options"),
#     Input("crossfilter-graph-type", "value"),
#     Input("particle-type", "value")
# )
# def update_options(value):
#     options = [{'label': 'PM1', 'value': 'pm1'}, 
#     {'label': 'PM2.5', 'value': 'pm25'}, 
#     {'label': 'PM10', 'value': 'pm10'}]

#     if len(value) > 1:
#         options = [
#             {
#                 "label": option["label"],
#                 "value": option["value"],
#                 "disabled": option["value"] not in value,
#             }
#             for option in options
#         ]
    
#     return options

# Dropdown: return "value", type: list, 
# Checkbox: returns "value", type: list,
# datepicker: returns "start_date", type: date,
#           : return "end_date", type: date,
# radioItems: returns "value", type: whatever you want it to 

# This is the graph that is generated with the modified dataframe based on input parameters 
@callback(
    Output('main-graph2', 'figure'),
    Input('storage2', 'data'),
    Input('crossfilter-graph-module', 'value'),
    Input('crossfilter-graph-freq', 'value'),
    Input('particle-type', 'value'),
    Input('datepicker', 'start_date'),
    Input('datepicker', 'end_date'),
)
def update_graph(df_json, modules, freq, types, start, end):
    # since id = particle-type is a radioitem, types returns a string, not a list
    dff = pd.read_json(df_json, orient='split')
    print(dff.head())
    print(dff['module'].unique())

    # if graph_type == "PM Data":
    # dff = dff[types + ['timestamp_local', 'module']]

    # if graph_type == 'Wind Data':
    dff = dff[[types] + ['timestamp_local', 'module', 'wind direction', 'wind speed']]

    figure_list = []
    # symbol list to distinguish the different particles 
    symbol = ['circle', 'triangle-up', 'square']

    for i, type1 in enumerate([types]): 
        # draw a scatter plot for every module for each TYPE THAT IS CHECKED ON THE CHECKBOX
        # the labels = {} updates the hover labels 
        sub_fig = px.scatter_3d(dff, x = "wind direction", y = "wind speed", z = type1, color = 'module', \
        labels = {
            "wind direction": "Wind Direction", 
            "wind speed": "Wind Speed",
            f"{type}": f"{type}".upper()
        }) \
        .update_layout(paper_bgcolor = "#F2EBCE", legend_title = "Modules")

        # update the scatter symbol (the shapes of the dots basically) based on particle matter type 
        sub_fig.update_traces(marker_symbol = symbol[i])

        # update the scatter marker size
        sub_fig.update_traces(marker_size = 5)

        # update the variable name on the legend so it's actually distinguishable 
        # unnecessary for this page since you can only select one type 
        # sub_fig.for_each_trace(lambda t: t.update(name = t.name + ' PM2.5') if type1 == 'pm25' else t.update(name = t.name + ' ' + type1.upper()))
        
        # update the figure onto the figure_list to concatenate later 
        figure_list.append(sub_fig)
    
    # add all the subfigures into one giant figure     
    fig = go.Figure(data=functools.reduce(operator.add, [_.data for _ in figure_list]))

    # change pm25 -> pm2.5
    if 'pm25' in [types]: 
        index = [types].index('pm25')
        [types][index] = 'pm2.5'

    # add title to graph and makes the particle types all upper case
    fig.update_layout(title = f"{', '.join([i.upper() for i in sorted([types])])} Conc. Based on Wind Velocity", transition_duration=500)
    fig.update_layout(scene = dict(xaxis_title = "Wind Direction (Degrees)", yaxis_title = "Wind Speed (m/s)", zaxis_title = "PM Levels (μg/m³)"))

    return fig
