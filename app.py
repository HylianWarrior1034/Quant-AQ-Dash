# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc, Input, Output, ctx, page_container, page_registry
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

app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], use_pages=True)
load_figure_template("LUX")
app._favicon = (r"quantaq.ico")

# set up server for caching the csv
# server = app.server

# CACHE_CONFIG = {
#     # try 'FileSystemCache' if you don't want to setup redis
#     'CACHE_TYPE': 'redis',
#     'CACHE_REDIS_URL': os.environ.get('REDIS_URL', 'redis://localhost:6379')
# }
# cache = Cache()
# cache.init_app(app.server, config=CACHE_CONFIG)

### import df ###
df = pd.read_csv("Quant-aq_2022.csv")
all_sensors = df['module'].unique()
# df = df[['timestamp', 'module', 'pm1', 'pm25', 'pm10']]

# groupby each hour 
# df['timestamp'] = df['timestamp'].apply(lambda x: x[:-3])
# df = df.groupby(['timestamp', 'module']).mean().reset_index() 

# or you can groupby each day by doing this: 
# df['timestamp'] = df['timestamp'].apply(lambda x: x[:-6])
# df = df.groupby(['timestamp', 'module']).mean().reset_index() 
#################


# module = 'MOD-00022'

# fig = px.scatter(df, x="timestamp", y="pm1", color = 'module', title = f'PM10 Levels for {module}', \
#     labels = {
#         "timestamp": "Timestamp", 
#         "pm1": "PM1",
#         "module": "Module",
#     }) \
#     .update_layout(paper_bgcolor = "#F2EBCE", legend_title = "Modules")
# fig.update_traces(hovertemplate = "Timestamp = %{x}<br>PM Levels = %{y}<br>Module = %{color}")

link_names = {
    "1pm data": "PM Data",
    "2wind data": "Wind Data"
}

app.layout = \
html.Div([
    html.Div([
        html.Div([
            dcc.Link(children= link_names[page['name']] + " | ", href = page['path']) 
            for page in page_registry.values()
        ], className = 'graph-type'),

        html.Div(className = 'header', children='Sensor Data for 2022')
    ], className = 'top'),

    page_container,

    # html.Div([
    #     html.Div([
    #         html.Div([
    #             dcc.Dropdown(value = ['MOD-00021'], id='crossfilter-graph-module', multi = True)
    #         ], className = 'pm-type'),

    #         html.Div([
    #             html.Div([
    #             dcc.RadioItems(['Minute', 'Hour', 'Day'], 'Day', id = 'crossfilter-graph-freq', labelStyle = {'display': 'inline-block', 'marginTop': '5px', 'padding': '10px'})
    #             ]),

    #             html.Div([
    #                 dcc.Checklist(id = 'particle-type', labelStyle = {'display': 'inline-block', 'marginTop': '5px', 'padding': '10px'})
    #             ])
    #         ], className = 'radios', style = {'padding': '0px'}),

    #         html.Div([
    #             dcc.Graph(id='main-graph', style={'height': '65vh'})
    #         ], className = 'graph'),

    #     ], className = 'left'),

    #     html.Div([
    #         html.Div([
    #             dcc.DatePickerRange('2022-01-01', '2022-12-31', min_date_allowed = date(2022, 1, 1), max_date_allowed = date(2022, 12, 31), id = 'datepicker')
    #         ], className = 'daterange'),

    #         html.Div([
    #             html.Button("Download Raw Data", id = "btn-download-raw-csv", className = "downloader_btn"),
    #             dcc.Download(id="download-raw-csv")
    #         ], className = 'downloader'),

    #         html.Div([
    #             html.Button("Download Averaged Data", id = "btn-download-csv", className = "downloader_btn"),
    #             dcc.Download(id="download-csv")
    #         ], className = 'downloader'),

    #         # html.Div(id = 'my-output')
    #     ], className = 'right')
    # ], className = 'mainframe'),

    html.Div(className = 'footer', children = 'Powered by Quant-AQ.inc'),
])

# # this was to check what the outputs of the components were
# @app.callback(
#     Output(component_id = 'my-output', component_property = 'children'),
#     Input(component_id='btn-download-csv', component_property = 'n_clicks')
# )
# def update_output(types):
#     return f'Output: {types}'


# LAYOUT CUZ DASH LAYOUT IS BAD 
# class -> className in Dash 

# <body>
#     <div class = 'header'>
#     <div class = 'left'>
#         <div class = 'graph type'>
#         <div class = 'hour or day'> 
#         <div class = 'graph'>
#          & <div class = 'particle size'>
#     </div> 
#     <div class = 'right'> 
#         <div class = 'datepickerrange'>
#         <div class = 'data download'> 
#             <button("Download Data for the Current Graph"), id = 'btn-download-csv'>
#     </div> 
# </body>

# ALL THE POSSIBLE PARAMETERS: 
# particle size
# date range 
# which modules 



if __name__ == '__main__':
    app.run_server(debug=True, port=8080)