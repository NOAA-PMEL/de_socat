from datetime import date, datetime, timezone
import hashlib
import io
import constants
from itertools import compress
import json
import os
import pprint
import urllib
import math
from io import StringIO

import colorcet as cc
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    ctx,
    dcc,
    exceptions,
    html,
    no_update,
)
import dash_ag_grid as dag
import dash_design_kit as ddk
import datashader as ds
import datashader.transfer_functions as tf
import db
import maputil
import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import plotly.colors as colors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyproj import Transformer
import redis
from sdig.erddap.info import Info
from sqlalchemy import all_, create_engine, engine
from sqlalchemy.pool import NullPool
from theme import theme, tabs_styles, tab_style, tab_selected_style
import util
from datetime import datetime


def get_blank(message):
    plot_bg = 'rgba(1.0, 1.0, 1.0 ,1.0)'
    blank_graph = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], showlegend=False))
    blank_graph.add_trace(go.Scatter(x=[0, 1], y=[0, 1], showlegend=False))
    blank_graph.update_traces(visible=False)
    blank_graph.update_layout(
        height=map_height,
        xaxis={"visible": False},
        yaxis={"visible": False},
        title=message,
        plot_bgcolor=plot_bg,
        annotations=[
            {
                "text": message,
                "xref": "paper",
                "yref": "paper",
                "showarrow": False,
                "font": {
                    "size": 14
                }
            },
        ]
    )
    return blank_graph


# When there will be more than 50,000 (???) points on the property property panel
# either use the decimated data set or
# segement by time to display to show the first 50000 with a time selector menu to see the remaning segments
#


# Create a SQLAlchemy connection string from the environment variable `DATABASE_URL`
# automatically created in your dash app when it is linked to a postgres container
# on Dash Enterprise. If you're running locally and `DATABASE_URL` is not defined,
# then this will fall back to a connection string for a local postgres instance
#  with username='postgres' and password='password'
connection_string = "postgresql+pg8000" + os.environ.get(
    "DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432"
).lstrip("postgresql")

# Create a SQLAlchemy engine object. This object initiates a connection pool
# so we create it once here and import into app.py.
# `poolclass=NullPool` prevents the Engine from using any connection more than once. You'll find more info here:
# https://docs.sqlalchemy.org/en/14/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
postgres_engine = create_engine(connection_string, poolclass=NullPool)

pp = pprint.PrettyPrinter(indent=4)

marker_size = 6

# Sample every N hours
hours_interval = 168

redis_instance = redis.StrictRedis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))

visible = {'visibility': 'visible'}
hidden = {'visibility': 'hidden'}

no_display = {'display': 'none'}
display_block = {'display': ''}

center = {'lon': 0.0, 'lat': 0.0}
zoom = 1.4

x_legend = [0.0, .355, .71]
y_legend = [1.026, 0.815, 0.604, 0.393, 0.18]

dtype_definitions = {'expocode': 'str', 'organization': 'str', 'investigators': 'str', 'platform_name': 'str', 'platform_type': 'str', 'qc_flag': 'str', 'socat_version': 'str'}

# [x-axis, y-axis, color-by]
thumbnail_pairs = [
    ['longitude','latitude','WOCE_CO2_water'],
    ['time','sample_number','WOCE_CO2_water'],
    ['time','longitude','WOCE_CO2_water'],
    ['time','latitude','WOCE_CO2_water'],
    ['time','temp','WOCE_CO2_water'],
    ['time','Temperature_equi','WOCE_CO2_water'],
    ['time','fCO2_recommended','WOCE_CO2_water'],
    ['temp','fCO2_recommended','WOCE_CO2_water'],
    ['time','sal','WOCE_CO2_water'],
    ['time','woa_sss','WOCE_CO2_water'],
    ['time','Pressure_atm','WOCE_CO2_water'],
    ['time','Pressure_equi','WOCE_CO2_water'],
    ['time','delta_temp','WOCE_CO2_water'],
    ['time','xCO2_water_equi_temp_dry_ppm','WOCE_CO2_water'],
    ['time','xCO2_water_sst_dry_ppm','WOCE_CO2_water']
]

# for thumbnails
thumbnail_num_plots = len(thumbnail_pairs)
thumbnail_num_rows = int(thumbnail_num_plots/3)
if thumbnail_num_rows == 0:
    thumbnail_num_rows = thumbnail_num_rows + 1
if thumbnail_num_plots > 3 and thumbnail_num_plots%3 > 0:
    thumbnail_num_rows = thumbnail_num_rows + 1
image_height = thumbnail_num_rows*450


socatQC = { 
				'validExpoChars': "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-",

				'fco2two': "Accuracy of calculated fCO2w (at SST) less than 2 uatm",
				'fco2five': "Accuracy of calculated fCO2w (at SST) less than 5 uatm",
				'fco2ten': "Accuracy of calculated fCO2w (at SST) less than 10 uatm",
				'fco2bad': "Accuracy of calculated fCO2w (at SST) more than 10 uatm",

				'soptrue': "Followed standard methods/SOP",
				'sopfalse': "Did not follow standard methods/SOP",

				'metacomplete': "Metadata complete",
				'metalacking': "Metadata not complete",

				'datagood': "Data quality acceptable",
				'databad': "Significant amount of unacceptable-quality data",

				'crossfound': "High-quality cross-over with ",
				'crossnone': "No high-quality cross-overs found at the time of this QC",

				'commentSpacer': ".  "
			}

thumbnail_vars = []
for sub_list in thumbnail_pairs:
    thumbnail_vars.extend(sub_list)

thumbnail_vars = list(set(thumbnail_vars))

ESRI_API_KEY = os.environ.get('ESRI_API_KEY')

zoom = 1
center = {'lon': 0.0, 'lat': 0.0}
map_limits = {"west": -180, "east": 180, "south": -89, "north": 89}

map_height = 600
map_width = 1200

agg_x = 72
agg_y = 36

map_title_base = 'Trajectory from the latest SOCAT Decimated Data Set '
decimated_url = 'https://data.pmel.noaa.gov/socat/erddap/tabledap/socat_v2025_decimated'
full_url = 'https://data.pmel.noaa.gov/socat/erddap/tabledap/socat_v2025_fulldata'

edf = pd.read_sql('SELECT * from cruises', con=postgres_engine)
expos = sorted(list(edf['expocode']))
initial_expo_options = []

for code in expos:
    initial_expo_options.append({'label':code, 'value':code})
initial_expo_value = [expos[0]]

investigaors = list(edf['investigators'].unique())
investigaors_options = []
for investigator in sorted(investigaors):
    investigaors_options.append({'value': investigator, 'label': investigator})

organizations = list(edf['organization'].unique())
organization_options = []
for organization in sorted(organizations):
    organization_options.append({'value': organization, 'label': organization})

# Define Dash application structure
app = Dash(__name__)
server = app.server  # expose server variable for Procfile

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
# DEBUG print('starting info query')
dinfo = Info(decimated_url)
variables, long_names, standard_name, units, v_d_types = dinfo.get_variables()
variable_options = []
for var in variables:   
    if var != 'lat_meters' and var != 'lon_meters':
        variable_options.append({'label':var, 'value': var})
# DEBUG print('finished info meta')
start_date, end_date, start_seconds, end_seconds = dinfo.get_times()
# DEBUG print('finished info times')

columns_for_WOCE_edits = ["WOCE_CO2_water" ,"WOCE_CO2_atm", "fCO2_recommended", "expocode", "time", "longitude", "latitude"]


# all columns: expocode,dataset_name,platform_name,platform_type,organization,geospatial_lon_min,geospatial_lon_max,geospatial_lat_min,geospatial_lat_max,time_coverage_start,time_coverage_end,investigators,socat_version,all_region_ids,socat_doi,qc_flag,sample_number,year,month,day,hour,minute,second,longitude,latitude,depth,sal,Temperature_equi,temp,Temperature_atm,Pressure_equi,Pressure_atm,xCO2_water_equi_temp_dry_ppm,xCO2_water_sst_dry_ppm,xCO2_water_equi_temp_wet_ppm,xCO2_water_sst_wet_ppm,pCO2_water_equi_temp,pCO2_water_sst_100humidity_uatm,fCO2_water_equi_uatm,fCO2_water_sst_100humidity_uatm,xCO2_atm_dry_actual,xCO2_atm_dry_interp,pCO2_atm_wet_actual,pCO2_atm_wet_interp,fCO2_atm_wet_actual,fCO2_atm_wet_interp,delta_xCO2,delta_pCO2,delta_fCO2,relative_humidity,specific_humidity,ship_speed,ship_dir,wind_speed_true,wind_speed_rel,wind_dir_true,wind_dir_rel,WOCE_CO2_water,WOCE_CO2_atm,woa_sss,pressure_ncep_slp,fCO2_insitu_from_xCO2_water_equi_temp_dry_ppm,fCO2_insitu_from_xCO2_water_sst_dry_ppm,fCO2_from_pCO2_water_water_equi_temp,fCO2_from_pCO2_water_sst_100humidity_uatm,fCO2_insitu_from_fCO2_water_equi_uatm,fCO2_insitu_from_fCO2_water_sst_100humidty_uatm,fCO2_from_pCO2_water_water_equi_temp_ncep,fCO2_from_pCO2_water_sst_100humidity_uatm_ncep,fCO2_insitu_from_xCO2_water_equi_temp_dry_ppm_woa,fCO2_insitu_from_xCO2_water_sst_dry_ppm_woa,fCO2_insitu_from_xCO2_water_equi_temp_dry_ppm_ncep,fCO2_insitu_from_xCO2_water_sst_dry_ppm_ncep,fCO2_insitu_from_xCO2_water_equi_temp_dry_ppm_ncep_woa,fCO2_insitu_from_xCO2_water_sst_dry_ppm_ncep_woa,fCO2_recommended,fCO2_source,delta_temp,region_id,calc_speed,etopo2,gvCO2,dist_to_land,day_of_year,time,lon360,tmonth,nobs_full,nobs_deci


app.layout = ddk.App(show_editor=True, theme=theme, children=[
    dcc.Store(id='plot-data-change'),
    dcc.Store(id='map-info'),
    dcc.Store(id='make-cruise-tracks'),
    html.Div(id='kick', style={'visibility':'none'}),
    ddk.Header(children=[
        ddk.Logo(src='https://www.socat.info/wp-content/uploads/2017/06/cropped-socat_cat.png'),
        ddk.Title('Surface Ocean CO\u2082 QC Editor'),
        ddk.Modal(id='debug-woce-flag', target_id='show-woce-edits-card', hide_target=True, children=[
            html.Button(id='show-edits', children='DEBUG: See edited rows.'),
        ]),
        ddk.Modal(id='debug-qc-entries', target_id='show-qc-entries-card', hide_target=True, children=[
            html.Button(id='show-qc-entries', children='DEBUG: See added QC entries.'),
        ])
    ]),
    ddk.Card(id='show-woce-edits-card', children=[
        dag.AgGrid(id='edited-points')
    ]),
    ddk.Card(id='show-qc-entries-card', children=[
        dag.AgGrid(id='qc-entries')
    ]),
    dcc.Tabs(id="top-level-tabs", value='map', style=tabs_styles, children=[
        dcc.Tab(id='map-tab', value='map', label='Cruise Selection', style=tab_style, selected_style=tab_selected_style, children=[
            ddk.Card(width=.25, children=[
                html.Div(style={'height':'82vh', 'overflow':'scroll'}, children=[
                    ddk.ControlCard(children=[
                        ddk.CardHeader("Selection Constraints"),
                        dcc.Loading(children=[
                            html.Button(id='reset', children=["Reset"], disabled=True),                     
                            html.Button(id='search', children=["Find Cruises"], disabled=False)
                        ]),
                    ]),
                    ddk.ControlCard(children=[
                        ddk.CardHeader('Latitude/Longitude Contraint'),
                        ddk.Block(width=1, children=[
                            ddk.Block(width=.3), ddk.Block(width=.3, children=[dcc.Input(id='ur_lat', type='text', value=90, style={'width':'12ch'})]),ddk.Block(width=.3),
                            ddk.Block(width=.3, children=[dcc.Input(id='ll_lon', type='text', value=-180, style={'width':'12ch'})]), ddk.Block(width=.3, ),ddk.Block(width=.3, children=[dcc.Input(id='ur_lon', type='text', value=180, style={'width':'12ch'})]),
                            ddk.Block(width=.3), ddk.Block(width=.3, children=[dcc.Input(id='ll_lat', type='text', value=90, style={'width':'12ch'})]),ddk.Block(width=.3),
                        ])
                    ]),
                    # ddk.ControlCard(children=[
                    #     ddk.CardHeader("Variable on the Map"),
                    #     dcc.Dropdown(id='map-variable', placeholder='Color Variable on Map', options=variable_options, value='fCO2_recommended')
                    # ]),
                    ddk.ControlCard(children=[
                        ddk.CardHeader("Expocode"),
                        dcc.Dropdown(id='expocode', placeholder='Select expocodes', multi=True, clearable=True, options=initial_expo_options)
                    ]),
                    ddk.Block(width=1, children=[
                        ddk.Block(width=.5, children=[
                            ddk.ControlCard(children=[
                                ddk.CardHeader("Region"),
                                dcc.Dropdown(id='region', multi=True,
                                    placeholder='Region',
                                    options=[
                                        {'value': "A", "label":'North Atlantic'},
                                        {'value': "C", "label": "Coastal"},
                                        {'value': "I", "label":'Indian'},
                                        {'value': "N", "label":"North Pacific"},
                                        {'value': "O", "label": "Southern Oceans"},
                                        {'value': "R", "label": "Arctic"},
                                        {'value': "T", "label": "Tropical Pacific"},
                                        {'value': "Z", "label": "Tropical Atlantic"}
                                    ]
                                ),
                            ]),
                        ]),
                        ddk.Block(width=.5, children=[
                            ddk.ControlCard(children=[
                                ddk.CardHeader("WOCE Flag"),
                                dcc.Dropdown(id='woce-co2-water', placeholder='WOCE CO\u2082 Water',
                                    multi=True,
                                    options=[
                                        {'value': "2", "label":'2'},
                                        {'value': "3", "label": "3"},
                                        {'value': "4", "label":'4'},
                                    ]
                                )
                            ]),
                        ])
                    ]),
                    # https://stackoverflow.com/questions/70714819/dash-plotly-datetime-selection
                    ddk.ControlCard(children=[
                        ddk.CardHeader("Date Range"),
                        ddk.ControlItem(label="Start Date", label_position="left", children=[
                            dcc.Input(id='start-date-picker', value=start_date, type="date")
                        ]),
                        ddk.ControlItem(label="End Date", label_position="left", children=[
                            dcc.Input(id='end-date-picker', value=end_date, type="date")
                        ])
                    ]),
                    ddk.ControlCard(children=[
                        ddk.CardHeader("Investigators"),
                        dcc.Dropdown(id='investigator', placeholder='Investigators', clearable=True, multi=True, options=investigaors_options), 
                    ]),
                    ddk.ControlCard(children=[
                        ddk.CardHeader("Organization"),
                        dcc.Dropdown(id='organization', placeholder='Organizations', searchable=True, options=organization_options),
                    ]),
                    ddk.ControlCard(children=[
                        ddk.CardHeader("QC Flag"),
                        dcc.Dropdown(id='qc-flag', placeholder='Select QC Flag', clearable=True, multi=True, 
                            options=[
                                    {'label': 'A', 'value': 'A'},
                                    {'label': 'B', 'value': 'B'},
                                    {'label': 'C', 'value': 'C'},
                                    {'label': 'D', 'value': 'D'},
                                    {'label': 'E', 'value': 'E'},
                                    {'label': 'Q', 'value': 'Q'},
                                    {'label': 'U', 'value': 'U'},
                                    {'label': 'N', 'value': 'N'},
                                ], 
                            # value=["Q", "U", "N"]
                        ),   
                    ]),
                    ddk.ControlCard(children=[
                        ddk.CardHeader("Platform Type"),
                        dcc.Dropdown(id='platform-type', placeholder='Select Platform Type', clearable=True, multi=True, options=[
                            {'label': "Autonomous Surface Vehicle", 'value': "Autonomous Surface Vehicle"},
                            {'label': "Boat", 'value': "Boat"},
                            {'label': "Drifting Buoy", 'value': "Drifting Buoy"},
                            {'label': "Mooring", 'value': "Mooring"},
                            {'label': "Ship", 'value': "Ship"}
                        ]),  
                    ])
                ]),
            ]),
            ddk.Card(width=.75, style={'height': '86vh'}, children=[
                ddk.CardHeader(id='map-graph-header', title='Select search latitude and longitude range'),                 
                    ddk.Graph(id='map-graph', style={'height': '95%', 'width':'95%'}, 
                ),
            ]),
           
        ]),
        dcc.Tab(id='table-tab', value='table', label='Table and Map of Selected Cruises', style=tab_style, selected_style=tab_selected_style, children=[
            dcc.Tabs(id="selected-cruises-tabs", value='table-sub-tab', style=tabs_styles, children=[
                dcc.Tab(id='table-sub-tab', label='Table of Selected Cruises', value='table-sub-tab', style=tab_style, selected_style=tab_selected_style, children=[
                    ddk.Card(children=[
                        # ddk.CardHeader(fullscreen=True),
                        dcc.Loading(children=[
                            dag.AgGrid(id='table-of-cruises', dashGridOptions={'pagination':True, "paginationAutoPageSize": True}, 
                                                columnSize="sizeToFit",
                                                defaultColDef={"resizable": True},
                                                style={'height': '80vh'}),
                        ])
                    ])
                ]),
                dcc.Tab(id='tracks-sub-tab', label='Map of Selected Cruise', value='tracks-sub-tab', style=tab_style, selected_style=tab_selected_style, children=[
                    ddk.Card(style={'height': '86vh'}, children=[
                        dcc.Loading(children=[
                            ddk.CardHeader(id='cruise-tracks-header', title='Select search search criteria on the first tab.'),
                            html.Div(id='track-data-loading', style={'display': 'none'})
                        ]),
                        ddk.Graph(id='cruise-tracks', style={'height': '95%', 'width':'95%'},),
                    ]),
                ])
            ])
        ]),
        dcc.Tab(id='plots', value='plots', label='Plots and QC', style=tab_style, selected_style=tab_selected_style, children=[
                ddk.Card(width=.25, children=[
                    ddk.ControlCard(children=[
                        ddk.CardHeader('Download Data'),
                        ddk.Block(width=1, children=[
                            dcc.Loading(children=[
                                html.A(id='show', children=[html.Button("Show", id='show-button',)], href=full_url, target='_blank'),
                                html.A(id='csv', children=[html.Button('CSV', id='csv-button', )], href=full_url, target='_blank'),
                                html.A(id='netcdf', children=[html.Button('netCDF', id='netcdf-button',)], href=full_url, target='_blank')
                            ])
                        ])
                    ]),
                    ddk.ControlCard(children=[   
                        ddk.CardHeader('Expocodes to Plot'),
                        dcc.Dropdown(id='plot-expocode', multi=False, clearable=False)
                    ]),
                    ddk.Card(id='save-full-message-card', style={'visibility':'hidden'}, children=[
                        ddk.CardHeader(title='These changes have been saved...'),
                        html.Div(id='save-full-message'),
                        html.Button('OK', id='close-save-full-message')
                    ])
                ]),
                ddk.Card(width=.75, children=[
                dcc.Tabs(id='plot-qc-level-tabs', style=tabs_styles, children=[
                    dcc.Tab(id='trajectories', value='trajectories', label='Map of Selected Cruise', style=tab_style, selected_style=tab_selected_style, children=[
                        ddk.Card(style={'height': '85vh'}, children=[
                            dcc.Loading(children=[
                                ddk.CardHeader(id='trace-graph-header', title='Selected Cruise                                       ', children=[
                                    dcc.Dropdown(id='trace-variable', options=variable_options, value='fCO2_recommended', multi=False,)
                                ]),
                            ]),
                            # dcc.Loading(
                                ddk.Graph(id='trace-graph', style={'height': '95%', 'width':'95%'}
                                    # config={'modeBarButtonsToAdd':
                                    #     [
                                    #         'zoom2d',
                                    #         'drawopenpath',
                                    #         'drawclosedpath',
                                    #         'drawcircle',
                                    #         'drawrect',
                                    #         'eraseshape'
                                    #     ]
                                    # }
                                ),
                            # ),
                        ])
                    ]),
                    dcc.Tab(id='prop-prop', value='prop-prop-plot', label='Property-Propery Plot', style=tab_style, selected_style=tab_selected_style, children=[
                        ddk.ControlCard(id='prop-prop-controls', orientation='h', children=[
                            ddk.CardHeader('Property-Property Controls'),
                            ddk.ControlItem(label='Flag Selected Points', children=[
                                ddk.Block(children=[
                                    ddk.Modal(id='edit-flags-modal', target_id='selected-points-card', hide_target=True, children=[
                                        html.Button(id='flag', children="Set Flags")
                                    ])
                                ])
                            ]),
                            ddk.Block(width=1, id='selected-points-card', style={'height': '85vh', 'width': '85vw'}, children=[
                                
                                    ddk.ControlCard(width=1, style={'height': '35vh'}, orientation='h', children=[
                                        ddk.CardHeader(title='Set WOCE Flags'),
                                        ddk.ControlItem(width=.3, label='Set WOCE_CO2_water Checked Rows:', children=[
                                            html.Button('Set', id='set-woce-water', style={'width': '95px'}),
                                            dcc.Dropdown(id='qc-woce-co2-water', placeholder='Pick a Flag Value',
                                                multi=False, style={'width': '200px' },
                                                options=[
                                                    {'value': "2", "label": '2'},
                                                    {'value': "3", "label": "3"},
                                                    {'value': "4", "label": '4'},
                                                ]
                                            )
                                        ]),
                                        # ddk.ControlItem(children=[html.H6("Double click the WOCE Flag cell you want to change. When the menu appears, select the value you want to assign.")]),
                                        ddk.ControlItem(label="Save Flags", children=[html.Button(id='save-woce-flags', children='Save Flags')]),
                                        ddk.ControlItem(label='Comment', children=[dcc.Textarea(id='comment', rows=4, cols=65)]),
                                        
                                    ]),
                                    ddk.Card(style={'position': 'absolute', 'bottom': 0}, children=[
                                        dag.AgGrid(id='selected-points', style={'height': '55vh'}, dashGridOptions={"rowSelection": "multiple", "suppressRowClickSelection": True})
                                    ]) 
                            ]),
                            ddk.ControlItem(label='X-axis', children=[
                                dcc.Dropdown(id='prop-prop-x', value='time', clearable=False)
                            ]),
                            ddk.ControlItem(label='Y-axis', children=[
                                dcc.Dropdown(id='prop-prop-y', value='fCO2_recommended', clearable=False)
                            ]),
                            ddk.ControlItem(label='Color By', children=[
                                dcc.Dropdown(id='prop-prop-colorby', value='expocode', clearable=False)
                            ]),
                        ]), 
                        ddk.Card(children=[
                            dcc.Loading(children=[
                                ddk.CardHeader(id='prop-prop-graph-header', title='Property-proptery plot'),
                                dcc.Graph(id='prop-prop-graph', style={'height':'60vh'}), 
                                html.Div(id='prop-prop-loading') # Hides the card while the data is being pulled from ERDDAP
                            ])
                        ])
                    ]),
                    dcc.Tab(id='thumbnails-tab', value='prop-prop-thumbs', label='Thumbnail Plots', style=tab_style, selected_style=tab_selected_style, children=[
                        ddk.Card(children=[
                            dcc.Loading(color='white', type='dot', children=[
                                ddk.CardHeader(id='thumbnails-header', title='Thumbnail Plots'),
                            ]),
                            dcc.Loading(
                                dcc.Graph(id='thumbnails-graph', style={'height': image_height+40},
                                    config={'modeBarButtonsToAdd':
                                        [
                                            'zoom2d',
                                            'drawopenpath',
                                            'drawclosedpath',
                                            'drawcircle',
                                            'drawrect',
                                            'eraseshape'
                                        ]
                                    }
                                ),
                            ),
                        ]),
                    ]),
                    dcc.Tab(id='cruise-qc-tab', value='cruise-qc', label='Cruise QC', style=tab_style, selected_style=tab_selected_style, children=[
                        ddk.Card(width=1, id='cruise-qc-card', style={'height':'90vh'}, children=[
                            ddk.CardHeader(id='cruise-qc-card-header', title='Cruise QC for ...'),
                            ddk.ControlCard(width=1., id='expo-menu', orientation='h', children=[
                                ddk.ControlItem(label='Add Item', label_position='left', id='cruise-qc-button-item', children=[
                                    ddk.Modal(id='add-qc-dialog', target_id='add-qc-dialog-container', hide_target=True, customModalToggle='cruise-qc-cancel', children=[
                                        html.Button(id='add-cruise-qc', children=["Add QC"])
                                    ])
                                ])
                            ]),
                            dag.AgGrid(
                                id='cruise-qc-grid', 
                                dashGridOptions={'pagination':True},
                                columnSize="sizeToFit",
                                defaultColDef={"resizable": True},
                                style={"height": 650, "width": "100%",}, 
                            ),
                            html.Div(id='add-qc-dialog-container', style={'margin': '50px', 'padding': '25px', 'width':'58vw', 'height':650,}, children=[
                                ddk.ControlCard(id='add-qc-card', style={'height': 600, 'width': '55vw', 'overflow-y': 'scroll'}, children=[
                                    ddk.CardHeader(id='cruise-qc-card-title', title='Cruise QC for ...'),
                                    ddk.ControlItem(label='Regions', children=[
                                        dcc.Checklist(id='qc-region-value',
                                            options=[
                                                {'label': 'Costal', 'value': 'costal'},
                                                {'label': 'Arctic', 'value': 'arctic'},
                                                {'label': 'Global (Override regional QC flags)', 'value': 'global'},
                                            ],
                                            value=['global']
                                        )
                                    ]),
                                    ddk.ControlItem(label='Accuracy of calculated aqueous fCO2 at SST:',children=[
                                        dcc.RadioItems(id='fco2-comment',
                                            options=[
                                                {'label': '< 2 μatm (A, B)', 'value': 'fco2two'},
                                                {'label': '< 5 μatm (C, D)', 'value': 'fco2five'},
                                                {'label': '< 10 μatm (E)', 'value': 'fco2ten'},
                                                {'label': '> 10 μatm (F, S)', 'value': 'fco2bad'},
                                                {'label': '(no comment)', 'value': 'fco2no'},
                                            ], value='fco2no'
                                        )
                                    ]),
                                    ddk.ControlItem(label='Followed approved methods/SOP criteria:', children=[
                                        dcc.RadioItems(id='sop-comment', options=[
                                            {'label': 'true (A, B)', 'value':'soptrue'},
                                            {'label': 'false (C, D, E) - specify not followed in additional comments', 'value': 'sopfalse'},
                                            {'label': '(no comment)', 'value': 'sopno'}
                                        ], value='sopno' )
                                    ]),
                                    ddk.ControlItem(label='Metadata documentation:', children=[
                                        dcc.RadioItems(id='meta-comment', options=[
                                            {'label': 'complete (A, B, C, E)', 'value': 'metacomplete'},
                                            {'label': 'incomplete (D) - specify missing in additional comments', 'value': 'metalacking'},
                                            {'label': '(no comment)', 'value': 'metano'}
                                        ], value='metano' )
                                    ]),
                                    ddk.ControlItem( label='Data quality:', children=[
                                        dcc.RadioItems(id='data-comment', options=[
                                            {'label': 'acceptable (A, B, C, D, E)', 'value': 'datagood'},
                                            {'label': 'significant amount of unacceptable data (F, S)', 'value': 'databad'},
                                            {'label': '(no comment)', 'value': 'datano'}
                                        ], value='datano')
                                    ]),
                                    ddk.ControlItem(label='High-quality cross-over:',  children=[
                                        dcc.RadioItems(id='xover-comment', options=[
                                            {'label': 'found with dataset (A)', 'value': 'crossfound'},
                                            {'label': 'none found (B, C, D, E)', 'value': 'crossnone'},
                                            {'label': '(no comment)', 'value': 'crossno'}
                                        ], value='crossno')
                                    ]),
                                    ddk.ControlItem(label='Quality Control Flag to Assign', children=[
                                        dcc.Dropdown(id='added-qc-flag', options=[
                                            {'label': 'Comment', 'value': 'comment'},
                                            {'label': 'A', 'value': 'A'},
                                            {'label': 'B', 'value': 'B'},
                                            {'label': 'C', 'value': 'C'},
                                            {'label': 'D', 'value': 'D'},
                                            {'label': 'E', 'value': 'E'},
                                            {'label': 'F', 'value': 'F'},
                                            {'label': 'Suspend', 'value': 'suspend'},
                                            {'label': 'Exclude', 'value': 'Exclude'},
                                        ], value='comment', style={'width': '120px'}, multi=False)
                                    ]),
                                    ddk.ControlItem(label='Additional Comment', children=[
                                        dcc.Textarea(id='qc-additional-comment', rows=10, cols=80)
                                    ]),
                                    ddk.Block(width=.90, children=[
                                        html.Button("Save", id='cruise-qc-save'),
                                        html.Button("Cancel!", id='cruise-qc-cancel', style={'background-color': theme['accent_negative']})
                                    ]),        
                                ])
                            ]),
                        ]),
                    ])
                ])
            ])
        ])
    ]),
    
    ddk.Footer(children=[
        html.Hr(),
            ddk.Block(children=[
                ddk.Block(width=.3, children=[
                    html.Div(children=[
                        dcc.Link('National Oceanic and Atmospheric Administration',
                                href='https://www.noaa.gov/', style={'font-size': '.8em'}),
                    ]),
                    html.Div(children=[
                        dcc.Link('Pacific Marine Environmental Laboratory',
                                href='https://www.pmel.noaa.gov/',style={'font-size': '.8em'}),
                    ]),
                    html.Div(children=[
                        dcc.Link('oar.pmel.webmaster@noaa.gov', href='mailto:oar.pmel.webmaster@noaa.gov', style={'font-size': '.8em'})
                    ]),
                    dcc.Link('DOC |', href='https://www.commerce.gov/', style={'font-size': '.8em'}),
                    dcc.Link(' NOAA |', href='https://www.noaa.gov/', style={'font-size': '.8em'}),
                    dcc.Link(' OAR |', href='https://www.research.noaa.gov/', style={'font-size': '.8em'}),
                    dcc.Link(' PMEL |', href='https://www.pmel.noaa.gov/', style={'font-size': '.8em'}),
                    dcc.Link(' Privacy Policy |', href='https://www.noaa.gov/disclaimer', style={'font-size': '.8em'}),
                    dcc.Link(' Disclaimer |', href='https://www.noaa.gov/disclaimer',style={'font-size': '.8em'}),
                    dcc.Link(' Accessibility', href='https://www.pmel.noaa.gov/accessibility',style={'font-size': '.8em'})
                ]),
                ddk.Block(width=.7,children=[html.Img(src=app.get_asset_url('logo-PMEL-lockup-light_noaaPMEL_horizontal_rgb-txt_2024.png'), style={'height': '90px', 'padding':'14px'})])
            ])
        ], 
        style={"color":"white"}
    )

])


@app.callback(
    [
        Output('selected-points', 'rowData', allow_duplicate=True),
        Output('selected-points', 'selectedRows')
    ],
    [
        Input('set-woce-water', 'n_clicks')
    ],
    [
        State('selected-points', 'rowData'),
        State('selected-points', 'selectedRows'),
        State('qc-woce-co2-water', 'value')
    ], prevent_initial_call = True
)
def apply_woce_water(click, row_data, selected_rows, woce_flag):
    if row_data is None or len(row_data) < 1:
        return no_update
    if woce_flag is None or len(woce_flag) < 1:
        return no_update
    if selected_rows is None or len(selected_rows) < 1:
        return no_update
    for srow in selected_rows:
        time = srow['time']
        for row in row_data:
            if row['time'] == time:
                row['WOCE_CO2_water'] = int(woce_flag)
    return row_data, []


@app.callback(
    [
        Output('edited-points', 'rowData'),
        Output('edited-points', 'columnDefs'),
    ],
    [
        Input('show-edits', 'n_clicks')
    ], prevent_initial_call=True
)
def show_edits(clicked):
    edited_rows = db.show_saves()
    columnDefs=[{"field": i, "headerName": i} for i in sorted(edited_rows.columns, key=str.casefold)]
    return [edited_rows.to_dict("records"), columnDefs]


@app.callback(
    [
        Output('qc-entries', 'rowData'),
        Output('qc-entries', 'columnDefs')
    ],
    [
        Input('show-qc-entries', 'n_clicks')
    ]
)
def show_qc(clicked):
    new_qc = db.show_qc()
    columnDefs=[{"field": i, "headerName": i} for i in sorted(new_qc.columns, key=str.casefold)]
    return [new_qc.to_dict("records"), columnDefs]

@app.callback(
    [
        Output('add-qc-dialog', 'expanded', allow_duplicate=True),
    ],
    [
        Input('cruise-qc-cancel', 'n_clicks'),
    ], prevent_initial_call=True
)
def close_add_quc(click):
    return [False]


#### CURRENT ATTEMPT AT SAVE IMPLEMENTATION

@app.callback(
    [
        Output('comment', 'value'),
        Output('save-full-message-card', 'style', allow_duplicate=True),
        Output('save-full-message', 'children', allow_duplicate=True),
        Output('edit-flags-modal', 'expanded')
    ],
    [
        Input('save-woce-flags', 'n_clicks')
    ],
    [
        State('selected-points', 'rowData'),
        State('comment','value'),
    ], prevent_initial_call=True
)
def save_flags(click, edited_points, in_comment):
    reminder = 'You must supply a comment.'
    ex_reminder = 'No, really. You must supply a comment telling what you did and why.'
    if in_comment is None or len(in_comment) == 0 or in_comment == reminder or in_comment == ex_reminder:
        if in_comment == reminder:
            return ex_reminder, no_update, no_update, True
        else:
            return reminder, no_update, no_update, True
    selected_data_string = redis_instance.hget("cache","edit-table-data").decode('utf-8')
    selected_data_json = json.loads(selected_data_string)
    selected_data = pd.read_json(StringIO(selected_data_json))
    as_edited = pd.DataFrame(edited_points)
    edits = pd.concat([selected_data, as_edited]).drop_duplicates(keep=False)
    start = int(edits.shape[0]/2)
    d = datetime.now(timezone.utc)
    d = str(d)
    d = d.replace(d[-7:], 'Z')
    edits.loc[:, 'edit_timestamp'] = d
    edits.loc[:, 'comment'] = in_comment
    save_edits = edits.iloc[start:]
    save_edits.to_sql(constants.edits_table, postgres_engine, if_exists='append', index=False)
    # save_edits.to_sql(edits_table, db.mysql_engine, if_exists='append', index=False)
    return '', {'visibility':'visible'}, f'The WOCE flag changes have been saved.  {save_edits.shape[0]} rows changed.', False


@app.callback(
    [
        Output('cruise-qc-grid', 'columnDefs'),
        Output('cruise-qc-grid', 'rowData'),
        Output('cruise-qc-card-header', 'title'),
    ],
    [
        Input('plot-qc-level-tabs', 'value'),
        Input('plot-expocode', 'value')
    ], prevent_initial_call=True
)
def show_cruise_qc(click, expocode_to_show):
    if click != 'cruise-qc':
        return [no_update, no_update, no_update]        
    if expocode_to_show is not None and len(expocode_to_show) > 0:
        records = db.get_cruise_qc(expocode_to_show)
        columnDefs = []   
        for i in sorted(records.columns, key=str.casefold):
            columnDefs.append({"field": i, "headerName": i, 'wrapText': True, 'autoHeight': True, 'cellStyle': {"lineHeight": "unset"}})
        return [columnDefs, records.to_dict("records"), f'Cruise QC for {expocode_to_show}']
    else:
        return [[], pd.DataFrame(),'No expocode selected.']


@app.callback(
    [
        Output('top-level-tabs', 'value', allow_duplicate=True)
    ],
    [
        Input('search', 'n_clicks')
    ], prevent_initial_call = True
)
def set_tab(click):
    return['table']


# @app.callback(
#     [
#         Output('jan-sw', 'disabled'),
#         Output('feb-sw', 'disabled'),
#         Output('mar-sw', 'disabled'),
#         Output('apr-sw', 'disabled'),
#         Output('may-sw', 'disabled'),
#         Output('jun-sw', 'disabled'),
#         Output('jul-sw', 'disabled'),
#         Output('aug-sw', 'disabled'),
#         Output('sep-sw', 'disabled'),
#         Output('oct-sw', 'disabled'),
#         Output('nov-sw', 'disabled'),
#         Output('dec-sw', 'disabled'),
#     ],
#     [
#         Input('jan-sw', 'checked'),
#         Input('feb-sw', 'checked'),
#         Input('mar-sw', 'checked'),
#         Input('apr-sw', 'checked'),
#         Input('may-sw', 'checked'),
#         Input('jun-sw', 'checked'),
#         Input('jul-sw', 'checked'),
#         Input('aug-sw', 'checked'),
#         Input('sep-sw', 'checked'),
#         Input('oct-sw', 'checked'),
#         Input('nov-sw', 'checked'),
#         Input('dec-sw', 'checked'),        
#     ], prevent_initial_call=True

# )
# def set_season(jan_sw, feb_sw, mar_sw, apr_sw, may_sw, jun_sw, jul_sw, aug_sw, sep_sw, oct_sw, nov_sw, dec_sw):
#     print('season change fired ----------------------')
#     checked = [jan_sw, feb_sw, mar_sw, apr_sw, may_sw, jun_sw, jul_sw, aug_sw, sep_sw, oct_sw, nov_sw, dec_sw]
#     print('checked ----->', checked)
#     disabled = [True, True, True, True, True, True, True, True, True, True, True, True]
#     for i, check in enumerate(checked):
#         print('checking ', i, ' at = ', checked[i])
#         past =  (i-1)
#         future = ((i%12 + 1)%12)
#         if check:
#             print('checked set to false', i)
#             disabled[i] = False            
#         if checked[past] and not checked[future]:
#             print('past and not future', i)
#             disabled[i] = False
#         if not checked[past] and checked[future]:
#             print('not past and future', i)
#             disabled[i] = False
#         if check and checked[past] and checked[future]:
#             print('past and future', i)
#             disabled[i] = True
#     if disabled.count(True) == 12:
#         disabled = [False, False, False, False, False, False, False, False, False, False, False, False]
#     print('disabled ----------|', disabled)
#     print('=-=-=- done --==-=-=-=-')
#     return disabled
        

# @app.callback(
#     Output("modal-review-table", "opened"),
#     Output('edited-points', 'rowData'),
#     Output('edited-points', 'columnDefs'),
#     Input('edited-rows-button', 'n_clicks'),
#     Input('edit-delete', 'n_clicks'),
#     State("modal-edit-table", "opened"),
#     prevent_initial_call=True,
# )
# def modal_open_debug(show_button, delete_button, opened):
#     triggered_id = callback_context.triggered_id
#     if triggered_id == 'edited-rows-button':
#         edited_rows = db.show_saves()
#         columnDefs=[{"field": i, "headerName": i} for i in sorted(edited_rows.columns, key=str.casefold)]      
#     elif triggered_id == 'edit-delete':
#         db.delete_all_rows()
#         return [not opened, None, None]
#     return [not opened, edited_rows.to_dict("records"), columnDefs]


# @app.callback(
#     Output("modal-cruise-flags", "opened", allow_duplicate=True),
#     Input('add-cruise-qc', 'n_clicks'),
#     State("modal-edit-table", "opened"),
#     # Verify, but I think the expocode has been set in the menu State()
#     prevent_initial_call=True,
# )
# def modal_open_debug(show_button, opened):
#     return [not opened]

@app.callback(
    [
        Output('save-full-message-card', 'style', allow_duplicate=True)
    ],
    [
        Input('close-save-full-message', 'n_clicks')
    ], prevent_initial_call=True
)
def close_save_qc_entry(click):
    return [{'visibility':'hidden'}]


@app.callback(
    Output('save-full-message', 'children'),
    Output('save-full-message-card', 'style', allow_duplicate=True),
    Output('add-qc-dialog', 'expanded', allow_duplicate=True),
    Input('cruise-qc-save', 'n_clicks'),
    State('qc-region-value', 'value'),
    State('added-qc-flag', 'value'),
    State('fco2-comment', 'value'),
    State('sop-comment', 'value'),
    State('meta-comment', 'value'),
    State('data-comment', 'value'),
    State('xover-comment', 'value'),
    State('qc-additional-comment', 'value'),
    State('plot-expocode', 'value'),
    prevent_initial_call=True
) 
def show_and_save_comments(click, qc_region_value, flag_value, fco2, sop, meta, data, xover, additional_comment, expocode):
    print('qc flag saved')
    full_comment = ''
    if fco2 is not None and fco2 != 'fco2no':
        full_comment = full_comment + socatQC[fco2]
        print('added a comment ', full_comment)
    if sop is not None and sop != 'sopno':
        if len(full_comment) > 0:
            full_comment = full_comment + socatQC['commentSpacer']
        full_comment = full_comment + socatQC[sop]
        print('add sop comment ', full_comment)
    if meta is not None and meta != 'metano':
        if len(full_comment) > 0:
            full_comment = full_comment + socatQC['commentSpacer']
        full_comment = full_comment + socatQC[meta]
    if data is not None and data != 'datano':
        if len(full_comment) > 0:
            full_comment = full_comment + socatQC['commentSpacer']
        full_comment = full_comment + socatQC[data]
    if xover is not None and xover != 'crossno':
        if len(full_comment) > 0:
            full_comment = full_comment + socatQC['commentSpacer']
        full_comment = full_comment + socatQC[xover]
    if len(full_comment) > 0:
        full_comment = full_comment + '.'
    if len(additional_comment) > 0:
        full_comment = full_comment + ' ' + additional_comment
    print ('final comment ', full_comment)
    df_json_string = redis_instance.hget('cache', 'table-of-cruises').decode('utf-8')
    df = pd.read_json(StringIO(json.loads(df_json_string)))
    row = df.loc[df['expocode']==expocode]
    socat_version = str(row['socat_version'])
    # TODO we need to know who is logged in
    # Actually save the stuff instead of returning a query string
    frame_input = {'qc_flag': flag_value, 'qc_time': datetime.now(timezone.utc).isoformat(), 'expocode': expocode, 'socat_version': socat_version, 'region_id': qc_region_value, 'reviewer_id': 'FAKE REVIEWER', 'qc_comment': full_comment}
    df = pd.DataFrame(frame_input)
    df.to_sql(constants.qc_entries_table, postgres_engine, if_exists='append', index=False)
    return [full_comment, {'visibility':'visible'}, False]


# @app.callback(
#     Output("modal-edit-table", "opened"),
#     Output('comment', 'value'),
#     Input('prop-prop-graph', 'selectedData'),
#     Input('edit-save', 'n_clicks'),
#     Input('edit-cancel', 'n_clicks'),
#     State('selected-points', 'rowData'),
#     State("modal-edit-table", "opened"),
#     State('comment','value'),
#     prevent_initial_call=True,
# )
# def modal_open_edit(in_selected_data, save_button, cancel_button, rowData, opened, in_comment):
#     reminder = 'You must supply a comment.'
#     ex_reminder = 'No, really. You must supply a comment telling what you did and why.'
#     if in_selected_data is None: 
#         raise exceptions.PreventUpdate
#     if len(in_selected_data['points']) == 0:
#         raise exceptions.PreventUpdate
#     triggered_id = callback_context.triggered_id
#     if triggered_id == 'edit-save':
#         if in_comment is None or len(in_comment) == 0 or in_comment == reminder or in_comment == ex_reminder:
#             if in_comment == reminder:
#                 return no_update, ex_reminder
#             else:
#                 return no_update, reminder
#         selected_data_string = redis_instance.hget("cache","edit-table-data").decode('utf-8')
#         selected_data_json = json.loads(selected_data_string)
#         selected_data = pd.read_json(selected_data_json)
#         as_edited = pd.DataFrame(rowData)
#         edits = pd.concat([selected_data, as_edited]).drop_duplicates(keep=False)
#         start = int(edits.shape[0]/2)
#         d = datetime.utcnow()
#         d = str(d)
#         d = d.replace(d[-7:], 'Z')
#         edits.loc[:, 'edit_timestamp'] = d
#         edits.loc[:, 'comment'] = in_comment
#         save_edits = edits.iloc[start:]
#         save_edits.to_sql(edits_table, postgres_engine, if_exists='append', index=False)
#         save_edits.to_sql(edits_table, db.mysql_engine, if_exists='append', index=False)
#     return not opened, ''


@app.callback(
    [
        Output('prop-prop-x', 'options'),
        Output('prop-prop-y','options'),
        Output('prop-prop-colorby', 'options'),
        Output('start-date-picker', 'min'),
        Output('start-date-picker', 'max'),
        Output('start-date-picker', 'value'),
        Output('end-date-picker', 'min'),
        Output('end-date-picker', 'max'),
        Output('end-date-picker', 'value'),
        Output('investigator', 'options'),
        Output('organization', 'options'),
    ],
    [
        Input('kick', 'n_clicks')
    ]
)
def set_up(click_in):
    # DEBUG print('running setup')
    inv_url = decimated_url + '.csv?investigators&distinct()'
    inv_df = pd.read_csv(inv_url, skiprows=[1])
    investigator_options = []
    for investigator in sorted(inv_df['investigators']):
        investigator_options.append({'label': investigator, 'value': investigator})
    org_url = decimated_url + '.csv?organization&distinct()'
    org_df = pd.read_csv(org_url, skiprows=[1])
    org_options = []
    for org in sorted(org_df['organization']):
        org_options.append({'label': org, 'value': org})
    return [variable_options, variable_options, variable_options, start_date, end_date, start_date, start_date, end_date, end_date, investigator_options, org_options]


@app.callback(
    [
        Output('plot-data-change', 'data'),
        Output('show','href'),
        Output('csv','href'),
        Output('netcdf','href'),
        Output('prop-prop-loading', 'children')
    ],
    [
        Input('plot-expocode', 'value')
    ]
)
def cache_plot_data(in_plot_expocode):
    to_get = ','.join(variables)
    expo_con = util.make_con('expocode', in_plot_expocode)
    all_csv_url = f'{full_url}.csv?{to_get}{expo_con}'
    all_nc_url = all_csv_url.replace('csv','ncCF')
    all_html_url = all_csv_url.replace('csv','htmlTable')
    if in_plot_expocode is not None and len(in_plot_expocode) > 0:
        if not redis_instance.hexists("cache", str(in_plot_expocode)):
            url = f'{full_url}.csv?{to_get}&expocode="{in_plot_expocode}"'
            print(f'caching data from:\n{url}')
            df = pd.read_csv(url, skiprows=[1])
            redis_instance.hset('cache', in_plot_expocode, json.dumps(df.to_json()))
            # DEBUG this is not working
            # redis version??????
            # redis_instance.hexpire('cache', 3600, expo)
        return ['new_data', all_html_url, all_csv_url, all_nc_url, '']
    else:
        return ['no data', full_url, full_url, full_url,'']


@app.callback(
    [
        Output('trace-graph', 'figure'),
        Output('trace-graph-header', 'title'),
    ],
    [
        Input('plot-data-change','data'),
        Input('trace-variable', 'value')
    ],
    [
        State('plot-expocode', 'value'), 
    ], prevent_initial_call=True
)
def update_trace(in_change, trace_in_variable, trace_in_expocode, ):
    if trace_in_variable is None or len(trace_in_variable) < 1:
        trace_in_variable = 'fCO2_recommended'
    to_get = ','.join(variables)
    
    if trace_in_expocode is not None and len(trace_in_expocode) > 0:
        if redis_instance.hexists('cache', str(trace_in_expocode)):
            df_json_string = redis_instance.hget('cache', trace_in_expocode).decode('utf-8')
            df = pd.read_json(StringIO(json.loads(df_json_string)), dtype=dtype_definitions)
        else:
            url = f'{full_url}.csv?{to_get}&expocode="{trace_in_expocode}"'            
            df = pd.read_csv(url, skiprows=[1], dtype=dtype_definitions)
            redis_instance.hset('cache', str(code), json.dumps(df.to_json()))
            redis_instance.hexpire('cache', 3600, code)
    else:
        return [get_blank('Plot is being created or you need to select an expocode.'), no_update]  

    df = df.loc[df[trace_in_variable].notna()]
    rmin = df[trace_in_variable].min()
    rmax = df[trace_in_variable].max()
    figure = px.scatter_geo(df,
                            lat='latitude',
                            lon='longitude',
                            color=trace_in_variable,
                            color_continuous_scale='Viridis',
                            hover_data=['expocode','time','latitude','longitude',trace_in_variable],
                            range_color=[rmin,rmax], custom_data=['expocode'],)
    figure.update_traces(marker={'size':6})
    figure.update_coloraxes(colorbar={'orientation':'v', 'title_side':'right'})
    figure.update_geos(fitbounds='locations', lonaxis_range=[-180,180], lataxis_range=[-90,90])
    figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan', resolution=50)
    # figure.update_coloraxes(colorbar={'orientation':'h', 'thickness':20, 'y': -.175, 'title': None})
    title = f'All {trace_in_variable} data from {str(trace_in_expocode)}'
    #DEBUG
    print(f'returning value from trace of {trace_in_expocode}')
    return [figure, title]
    


@app.callback(
    [
        Output('map-graph', 'figure'),
        Output('map-graph-header', 'title'),
        Output('reset', 'disabled')
    ],
    [
        Input('kick', 'children'),
        Input('top-level-tabs', 'value'),
        Input('map-info', 'data')
        # Input('map-variable', 'value'),
        # Input('region', 'value'),
        # Input('woce-co2-water', 'value'),
        # Input('start-date-picker', 'value'),
        # Input('end-date-picker', 'value'),
        # Input('investigator', 'value'),            # These could be done with pattern matching, but...
        # Input('organization', 'value'),
        # Input('qc-flag', 'value'),
        # Input('platform-type', 'value'),
        # Input('expocode', 'value'),

# Above + map_info affects selected cruises

        # Input('map-info', 'data'),
        # Input('jan-sw', 'checked'),
        # Input('feb-sw', 'checked'),
        # Input('mar-sw', 'checked'),
        # Input('apr-sw', 'checked'),
        # Input('may-sw', 'checked'),
        # Input('jun-sw', 'checked'),
        # Input('jul-sw', 'checked'),
        # Input('aug-sw', 'checked'),
        # Input('sep-sw', 'checked'),
        # Input('oct-sw', 'checked'),
        # Input('nov-sw', 'checked'),
        # Input('dec-sw', 'checked'),  
    ],
    [
        State('region', 'value')
    ]
)
def update_map(kick, top_tab_value,
# map_in_variable, in_regions, in_woce_water, in_start_date, in_end_date, in_investigator, in_org, in_qc_flag, in_platform_type, 
map_info,
            #    jan_sw, feb_sw, mar_sw, apr_sw, may_sw, jun_sw, jul_sw, aug_sw, sep_sw, oct_sw, nov_sw, dec_sw,
            #    map_in_expocode
region_id
):
    map_type = 'geo'
    # DEBUG print('\n\nfiring update map')
    if ctx.triggered_id == 'top_tab_value' and top_tab_value != 'map':
        # DEBUG print(f'not updating map {ctx.triggered_id} and {top_tab_value}')
        return no_update

    try:
        df = pd.read_sql('SELECT * from map_counts', con=postgres_engine)
    except Exception as e:
        print(e)
        figure = go.Figure(go.Scattergeo())
        figure.update_layout(margin={'t':25, 'b':25, 'l':0, 'r':0})
        figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan', resolution=50)
        figure.update_layout(title='Query returned no results.')
        return [figure, 'No matching data found.', False]  

    title = f'Number of observations in each 1\u00B0 by 1\u00B0 box (Log10 color scale).'
    map_type='geo'
    mask_trace = None
    filtered_df = pd.DataFrame()
    if region_id is not None and len(region_id) > 0:
        # DEBUG print(f'using region_id of {region_id}')
        filtered_df = df.loc[df['region_id'].isin(region_id)]
    else:
        if map_info is not None and len(map_info) > 0:
            bounds = json.loads(map_info)
            if bounds['ll']['latitude'] != -90 and bounds['ll']['latitude'] != 90 and bounds['ll']['longitude'] != -180 and bounds['ur']['longitude'] != 180:
                mask = (df['latitude'] >= bounds['ll']['latitude']) & (df['latitude'] <= bounds['ur']['latitude'])& (df['longitude'] >= bounds['ll']['longitude']) & (df['longitude'] <= bounds['ur']['longitude'])
                filtered_df = df[mask]
    if not filtered_df.empty:
        mask_trace = px.scatter_geo(filtered_df, lat='latitude', lon='longitude')
        mask_trace.update_traces(marker=dict(size=7, color='green'), uirevision='x888')
    if map_type == 'geo':

        figure = px.scatter_geo(df,
                                lat='latitude',
                                lon='longitude',
                                color='log',
                                # hover_data=['expocode','time','latitude','longitude',map_in_variable],
                                # custom_data=['expocode'], 
                                projection='equirectangular',
                                color_continuous_scale='Inferno',
                                hover_data={'latitude':True, 'longitude':True, 'count':True,'log':False},
                                )
        
        figure.update_coloraxes(colorbar={'title': 'count', 'orientation':'v', 'title_side':'right','lenmode':'fraction', 'len':.65, 'y':.5, 
                                'tickvals': [0, math.log10(10), math.log10(100), math.log10(1000), math.log10(10000), math.log10(100000)],
                                'ticktext': ['0', '10', '100', '1,000', '10,000', '100,000'], 
                                'tickmode':'array'
                                },
                                )
        figure.update_geos(fitbounds='locations', lonaxis_range=[-180,180], lataxis_range=[-90,90], showland=True, landcolor='lightgrey', showocean=True, oceancolor="#9bedff", showlakes=True, lakecolor="#9bedff", coastlinecolor='black', coastlinewidth=1, resolution=50)
        figure.update_traces(marker=dict(size=7), selected_marker_color='green' )
        figure.update_layout(title=title, uirevision='x9999')
        if mask_trace is not None:
            # DEBUG print('adding mask')
            figure.add_traces(list(mask_trace.select_traces()))
    else:
        figure = px.scatter_map(df, lat='latitude', lon='longitude', color='fCO2_recommended', 
                                hover_data=['latitude', 'longitude'], 
                                hover_name='expocode', color_continuous_scale='Viridis')
        figure.update_traces(marker={'size': marker_size})
        figure.update_coloraxes(colorbar={'orientation':'v', 'title_side':'right'})

        figure.update_layout(
            showlegend=False,
            map_style="carto-voyager-nolabels",
            # map_layers=[
            #     {
            #         "below": 'traces',
            #         "sourcetype": "raster",
            #         "sourceattribution": "&nbsp;GEBCO &amp; NCEI&nbsp;",
            #         "source": [
            #             'https://tiles.arcgis.com/tiles/C8EMgrsFcRFL6LrL/arcgis/rest/services/GEBCO_basemap_NCEI/MapServer/tile/{z}/{y}/{x}'
            #         ]
            #     }
            # ],
            map_zoom=zoom,
            map_center=center,
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            legend=dict(
                orientation="v",
                x=-.01,
            ),
            modebar_orientation='v',
        )

    # expos_mapped = df['expocode'].unique()
    # map_summary = f' {len(expos_mapped)} cruises shown on the map.'
    # map_expo_options = []

    # expos_mapped = sorted(expos_mapped)
    # for code in expos_mapped:
    #     map_expo_options.append({'value': code, 'label': code})

    # print(f'dumping {len(map_expo_options)} to redis cache')
    # redis_instance.hset("cache", "expocodes", json.dumps(expos_mapped))
    # redis_instance.hset("cache", "expocode_options",json.dumps(map_expo_options))
    # return [figure, title + map_summary + plot_summary + subselection_summary, False]
    return [figure, '', False]

def get_map_ranges(df):
    lon_neg = df[df['longitude']<0].count()
    lon_pos = df[df['longitude']>0].count()
    pos = lon_pos['longitude']
    neg = lon_neg['longitude']
    lon_pos180 = df[df['longitude']>175].count() 
    lon_posM180 = df[df['longitude']<-175].count()
    near180 = lon_pos180['longitude'] + lon_posM180['longitude']
    if pos > 0 and neg > 0 and near180 > 0:
        fitbounds = False
        all_pos = df[df['longitude'] > 0]
        all_neg = df[df['longitude'] < 0]
        lon_min = all_pos['longitude'].min()
        lon_max = all_neg['longitude'].max()
    else:
        fitbounds = True
        lon_min = df['longitude'].min()
        lon_max = df['longitude'].max()
    lat_min = df['latitude'].min()
    lat_max = df['latitude'].max()
    return lat_min, lat_max, lon_min, lon_max, fitbounds


@app.callback(
    [
        Output('selected-points', 'rowData'),
        Output('selected-points', 'columnDefs')
    ],
    [
        Input('flag', 'n_clicks')
    ],
    [
        State('prop-prop-graph', 'selectedData')
    ]
)
def show_selected_points(click, in_points):
    if in_points is not None:
        all_data_string = redis_instance.hget("cache","plot-data").decode('utf-8')
        all_data_json = json.loads(all_data_string)
        all_data = pd.read_json(StringIO(all_data_json))
        # TODO These are the columns from the plot, maybe we should use the columns defined as necessary for setting the flags
        column_names = sorted(all_data.columns, key=str.casefold)
        column_names.remove('WOCE_CO2_water')
        column_names.insert(0, 'WOCE_CO2_water')
        column_names.remove('WOCE_CO2_atm')
        column_names.insert(1, 'WOCE_CO2_atm')
        columnDefs = []
        for idx, i in enumerate(column_names):
            if 'WOCE' in i:
                if idx == 0:
                    columnDefs.append(
                        {
                            "field": i, "headerName": i, 'editable': True, 'sortable': True, 'cellEditor': 'agSelectCellEditor', "tooltipComponent": "CustomTooltip",
                            'cellEditorParams': {'values': [2, 3, 4]},"tooltipField": i, 'tooltipShowDelay': 0, "checkboxSelection": True, "headerCheckboxSelection": True,
                        }
                    )
                else:
                    columnDefs.append(
                        {
                            "field": i, "headerName": i, 'editable': True, 'sortable': True, 'cellEditor': 'agSelectCellEditor', "tooltipComponent": "CustomTooltip",
                            'cellEditorParams': {'values': [2, 3, 4]},"tooltipField": i, 'tooltipShowDelay': 0, 
                        }
                    )
            elif 'time' in i:
                columnDefs.append({"field": i, "headerName": i, 'sortable': True})
            else:
                columnDefs.append({"field": i, "headerName": i})
        selected_points = in_points['points']
        times = []
        for point in selected_points:
            customs = point['customdata']
            times.append(customs[0])
        to_show = all_data.loc[all_data['time'].isin(times)]
        redis_instance.hset("cache", 'edit-table-data', json.dumps(to_show.to_json()))
        return [to_show.to_dict("records"), columnDefs]
    else:
        raise exceptions.PreventUpdate



@app.callback(
    [
        # Output('ll_lat', 'value'),
        # Output('ll_lon', 'value'),
        # Output('ur_lat', 'value'),
        # Output('ur_lon', 'value'),
        Output('map-info', 'data'),
        Output('region', 'value')
    ],
    [
        Input('map-graph','selectedData')
    ], prevent_initial_call=True
)
def selectData(selectData):
    # DEBUG 
    # print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    # print(selectData)
    # print('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    map_info = None
    
    if selectData is not None and 'range' in selectData:
        geo_range = selectData['range']['geo']
       
        map_info = {
            'll': {
                'longitude': float(f'{geo_range[0][0]:.2f}'),
                'latitude': float(f'{geo_range[1][1]:.2f}')
            },
            'ur': {
                'longitude': float(f'{geo_range[1][0]:.2f}'),
                'latitude': float(f'{geo_range[0][1]:.2f}')
            }
        }
        redis_instance.hset('cache', 'current_selection', json.dumps(selectData))
    # This is either changing to the whole globe or a selected region. Reset the region menu to nothing in either case.
    if map_info is None:
        raise exceptions.PreventUpdate
    else:
        # return [map_info['ll']['latitude'], map_info['ll']['longitude'], map_info['ur']['latitude'], map_info['ur']['longitude'], json.dumps(map_info), '']
        return[json.dumps(map_info), '']



@app.callback(
    [
        Output('plot-expocode', 'value', allow_duplicate=True),
        Output('top-level-tabs', 'value', allow_duplicate=True),
        Output('plot-qc-level-tabs', 'value', allow_duplicate=True)
    ],
    [
        Input('cruise-tracks', 'clickData')
    ], prevent_initial_call=True
)
def set_platform_code_from_map(in_click):
    out_expocode = None
    # DEBUG 
    print('=-=-=-=-=- starting set_platform_code_from_map =-=-=-=-=-=')
    print('printing click')
    print(str(in_click))
    if in_click is not None:
        # DEBUG print('getting first point')
        fst_point = in_click['points'][0]
        # DEBUG print(fst_point)
        if 'customdata' in fst_point:
            out_value = fst_point['customdata'][0]
            # DEBUG print('expo to add because of click ' + out_value)
            # DEBUG print('existing expo ' + str(state_in_expovalue))
        else:
            # DEBUG print('no custom data in click')
            raise exceptions.PreventUpdate
        return [out_value, 'plots', 'prop-prop-plot']



@app.callback(
    [
        Output('prop-prop-graph', 'figure'),
        Output('prop-prop-graph-header', 'title')
    ],
    [
        Input('plot-data-change', 'data'),
        Input('prop-prop-x', 'value'),
        Input('prop-prop-y', 'value'),
        Input('prop-prop-colorby', 'value'),
    ],
    [
        State('plot-expocode', 'value'),
 
    ], prevent_initial_call=True
)
def make_property_property(plot_data_store, in_prop_prop_x, in_prop_prop_y, in_prop_prop_colorby, plot_in_expocode,):
    in_map_variable = 'fCO2_recommended'
    to_get = '.'.join(variables)
    x_label = None
    y_label = None
    legend_title = None
    # DEBUG print('updating the property-propery plot ' + str(plot_in_expocode))

    if plot_in_expocode is None or len(plot_in_expocode) == 0:
        # DEBUG print('data-plot: no expo')
        return [get_blank('Choose an expocode from the menu at right.'), 'No expocode selected.']
    if in_map_variable is None or len(in_map_variable) == 0:
        # DEBUG print('data-plot: no variable')
        raise exceptions.PreventUpdate
    if plot_data_store == 'no':
        # DEBUG print('no new data')
        raise exceptions.PreventUpdate
    
    if plot_in_expocode is not None and len(plot_in_expocode) > 0:
        if redis_instance.hexists('cache', str(plot_in_expocode)):
            df_json_string = redis_instance.hget('cache', plot_in_expocode).decode('utf-8')
            df = pd.read_json(StringIO(json.loads(df_json_string)), dtype=dtype_definitions)
        else:
            url = f'{full_url}.csv?{to_get}&expocode="{plot_in_expocode}"'            
            df = pd.read_csv(url, skiprows=[1], dtype=dtype_definitions)
            redis_instance.hset('cache', str(code), json.dumps(df.to_json()),)
            redis_instance.hexpire('cache', 3600, code)
    else:
        return [get_blank('Select an expocode form the menu.'), no_update] 

    if df.shape[0] < 1:
        raise exceptions.PreventUpdate

    df['expocode'] = df['expocode'].astype(str)
    df['WOCE_CO2_water'] = df['WOCE_CO2_water'].astype(str)

    if in_prop_prop_colorby == 'expocode':
        cmap = px.colors.qualitative.Light24
    else:
        cmap = px.colors.qualitative.Dark24

    print('making property-property-plot')
    card_title = f'{in_prop_prop_y} vs {in_prop_prop_x} colored by {in_prop_prop_colorby} from {plot_in_expocode}'
    figure = px.scatter(df,
                        x=in_prop_prop_x,
                        y=in_prop_prop_y,
                        color=in_prop_prop_colorby,
                        hover_name='expocode',
                        hover_data=['time',in_prop_prop_x,in_prop_prop_y,in_prop_prop_colorby],
                        custom_data=['time'],
                        color_discrete_sequence=cmap,
                        category_orders={"WOCE_CO2_water": ["2", "3", "4", "5", "1"]},
                        color_continuous_scale=px.colors.sequential.Viridis,
        )



    plot_data = df[columns_for_WOCE_edits]
    redis_instance.hset("cache", "plot-data", json.dumps(plot_data.to_json()))
    return[figure, card_title]



@app.callback(
    [
        Output('thumbnails-graph', 'figure'),
        Output('thumbnails-header', 'title')
    ],
    [
        Input('plot-data-change', 'data'),
    ],
    [
        State('plot-expocode', 'value'),
 
    ], prevent_initial_call=True
)
def make_thumbnails(plot_data_store, plot_in_expocode,):
    #DEBUG print(f'plot type from tab {in_plot_type}')
    to_get = '.'.join(variables)

    # DEBUG print('updating the thumbnail plots ' + str(plot_in_expocode))

    if plot_in_expocode is None or len(plot_in_expocode) == 0:
        # DEBUG print('data-plot: no expo')
        return [get_blank('Choose an expocode from the menu.'), 'No expocode seledcted.']

    if plot_data_store == 'no':
        # DEBUG print('no new data')
        raise exceptions.PreventUpdate
    
    if plot_in_expocode is not None and len(plot_in_expocode) > 0:
        if redis_instance.hexists('cache', str(plot_in_expocode)):
            df_json_string = redis_instance.hget('cache', plot_in_expocode).decode('utf-8')
            df = pd.read_json(StringIO(json.loads(df_json_string)))
        else:
            url = f'{full_url}.csv?{to_get}&expocode="{plot_in_expocode}"'            
            df = pd.read_csv(url, skiprows=[1])
            redis_instance.hset('cache', str(code), json.dumps(df.to_json()))
            redis_instance.hexpire('cache', 3600, code)
    else:
        return [get_blank('Select an expocode form the menu.'), no_update] 

    if df.shape[0] < 1:
        print('No data to plot')
        raise exceptions.PreventUpdate

    df['expocode'] = df['expocode'].astype(str)
    df['WOCE_CO2_water'] = df['WOCE_CO2_water'].astype(str)

    cmap = px.colors.qualitative.Light24

    card_title = f'Property Property Thumbnails for {plot_in_expocode}'
    plots = []
    subplot_titles = []


    legend = 1
    for pair in thumbnail_pairs:
        if legend == 1 :
            leg = 'legend'
        else:
            leg = 'legend' + str(legend)
        subplot_title = pair[1] + ' vs ' + pair[0] + ' colored by ' + pair[2]
        x = pair[0]
        y = pair[1]
        color_by = pair[2]
        subplot_titles.append(subplot_title)
        if color_by == 'expocode':
            cmap = px.colors.qualitative.Light24
        else:
            cmap = px.colors.qualitative.Dark24
        subplot = px.scatter(df,
                    x=x,
                    y=y,
                    color=color_by,
                    hover_name='expocode',
                    # labels = {'color': color_by},
                    hover_data=['time','latitude','longitude','expocode', x, y, color_by],
                    custom_data=['time'],
                    color_discrete_sequence=cmap,
                    category_orders={"WOCE_CO2_water": ["2", "3", "4", "5","1"]},
                    # color_continuous_scale=px.colors.sequential.Viridis,
                )
        subplot.update_traces(legend=leg)
        plots.append(subplot)
        legend = legend + 1
    figure = make_subplots(cols=3, rows=thumbnail_num_rows, shared_xaxes=False, shared_yaxes=False)
    i = 1
    j = 1
    for d, plot in enumerate(plots):
        traces = list(plot.select_traces())
        figure.add_traces(traces, i, j)
        figure.update_xaxes(title_text=thumbnail_pairs[d][0], showticklabels=True, row=i, col=j)
        figure.update_yaxes(title_text=thumbnail_pairs[d][1], showticklabels=True, row=i, col=j)
        if d == 0:
            legend_name = 'legend'
        else:
            legend_name = 'legend' + str(d+1)
        figure.layout[legend_name] = {
            "title": color_by,
            "xref": "paper",
            "yref": "paper",
            "y": y_legend[i-1],
            "x": x_legend[j-1],
            "orientation": 'h',
        }
        j = (j + 1)%4
        if j == 0:
            j = j + 1
        if j == 1 and d > 0:
            i = i + 1

    figure.update_layout(height=image_height, margin=dict( l=80, r=80, b=80, t=80, ))
    # DEBUG print('returning figure and title')
    # DEBUG print(figure)
    return[figure, card_title]


@app.callback(
    [
        Output('plot-expocode', 'value', allow_duplicate=True),
        Output('top-level-tabs', 'value', allow_duplicate=True),
        Output('plot-qc-level-tabs', 'value', allow_duplicate=True)
    ],
    [
        Input('table-of-cruises', 'cellClicked')
    ], prevent_initial_call=True
)
def set_expo_from_table_click(cell):
    # DEBUG 
    print(f"clicked on cell value:  {cell['value']}, column:   {cell['colId']}, row index:   {cell['rowIndex']}")
    if cell['colId'] == 'prop':
        return [cell['value'], 'plots', 'prop-prop-plot',]
    elif cell['colId'] == 'thumbnails':
        return [cell['value'], 'plots', 'prop-prop-thumbs',]
    elif cell['colId'] == 'CruiseQC':
        # TODO this isn't correct
        return [cell['value'], 'plots', 'cruise-qc',]
    else:
        raise exceptions.PreventUpdate


@app.callback(
    [
        Output('table-of-cruises', 'rowData'),
        Output('table-of-cruises', 'columnDefs'),
        Output('plot-expocode', 'options'),
        Output('plot-expocode', 'value'),
        Output('make-cruise-tracks', 'data'),
        Output('track-data-loading', 'children')
    ],
    [
        Input('top-level-tabs', 'value')
    ],
    [
        State('expocode', 'value'),
        State('start-date-picker', 'value'),
        State('end-date-picker', 'value'),
        State('woce-co2-water', 'value'),
        State('region', 'value'),
        State('investigator', 'value'),
        State('organization', 'value'),
        State('qc-flag', 'value'),
        State('platform-type', 'value'),
        State('map-info', 'data')
    ], prevent_initial_call=True
)
def make_table_of_crusies(da_click, mt_in_expocodes, mt_in_start_date, mt_in_end_date, mt_in_woce_water, mt_in_regions, mt_in_investigator, mt_in_org, mt_in_qc_flag, mt_in_platform_type, mt_in_map_info):
    if da_click != "table":
        return [no_update, no_update, no_update, no_update, no_update, no_update]
    vars_to_get = ['expocode', 'platform_name',	'platform_type', 'investigators', 'qc_flag', 'socat_version']
    expo_con = util.make_con('expocode', mt_in_expocodes)
    time_con = '&time>='+mt_in_start_date+'&time<='+mt_in_end_date
    investigator_con = util.make_con('investigators', mt_in_investigator)
    org_con = util.make_con('organization', mt_in_org)
    if org_con:
        vars_to_get.append('organization')
    qc_flag_con = util.make_con('qc_flag', mt_in_qc_flag)
    platform_type_con = util.make_con('platform_type', mt_in_platform_type)
    woce_water_con = util.make_con('WOCE_CO2_water', mt_in_woce_water)
    region_con = util.make_con('region_id', mt_in_regions)
    if region_con:
        vars_to_get.append('region_id')
    url = decimated_url + '.csv?' + ','.join(vars_to_get) + expo_con + region_con + time_con + woce_water_con + investigator_con + org_con + qc_flag_con + platform_type_con + region_con
    if not region_con and mt_in_map_info is not None and len(mt_in_map_info) > 3:
        bounds = json.loads(mt_in_map_info)
        cons = maputil.get_socat_subset(bounds['ll']['longitude'], bounds['ur']['longitude'],bounds['ll']['latitude'],bounds['ur']['latitude'])
        url = url + cons['lat'] + cons['lon']
    url = url + '&distinct()'
    expo_options = []
    print('table URL: ' + url)
    try:
        # dtype to make sure the expocode is a string and does not drop the leading 0
        df = pd.read_csv(url, skiprows=[1], dtype=dtype_definitions)
    except:
        df = pd.DataFrame()
    if not df.empty:
        df['expocode'] = df['expocode'].astype(str)
        df['thumbnails'] = df.loc[:, 'expocode']
        df['documentation'] = 'https://data.pmel.noaa.gov/socat/las/MetadataDocsV2023/' + df.expocode.str.slice(start=0, stop=4) + '/' +  df.expocode + '/'
        df['CruiseQC'] = df.loc[:, 'expocode']
        df['prop'] = df.loc[:, 'expocode']
        df['links'] = df.loc[:, 'expocode']

    table_of_cruises_columnDefs = [
        {'field': 'expocode', 'headerName': 'Expocode'},
        {"field": 'links', "headerName": 'Actions',
            "children": [
                    {"field": "documentation", 'headerName': 'Documentation', 'cellRenderer': "DocLink", 'cellStyle': {'color': 'blue', 'text-decoration': 'underline'}},
                    {"field": "prop", 'headerName': 'Prop-prop plot', 'cellRenderer': 'myButtonCellRenderer', "autoHeight": True},
                    {"field": "thumbnails", 'headerName': 'Thumbnails', 'cellRenderer': 'myButtonCellRenderer', "autoHeight": True},
                    {"field": "CruiseQC", 'headerName': 'CruiseQC', 'cellRenderer': 'myButtonCellRenderer', "autoHeight": True},
                ]
        },
        {'field':'investigators', 'headerName': 'Investigators'},
        {'field': 'platform_name', 'headerName': 'Platform Name'},
        {'field': 'qc_flag', 'headerName': 'QC Flag'},
        {'field': 'socat_version', 'headerName': 'SOCAT Version'}
    ]  

    expos = sorted(list(df['expocode']))
    for code in expos:
        expo_options.append({'label': code, 'value': code})
    expo_value = expos[0]
    # Cache the table so we can extract the version number when/if we go to save a QC entry
    # and so we can make a map from the locations database
    redis_instance.hset('cache', 'table-of-cruises', json.dumps(df.to_json()))
    return [df.to_dict("records"), table_of_cruises_columnDefs, expo_options, expo_value, 'go', '']


@app.callback(
    [
        Output('cruise-tracks', 'figure'),
        Output('cruise-tracks-header', 'title')
    ],
    [
        Input('make-cruise-tracks', 'data')
    ], prevent_initial_call = True
)
def make_cruise_tracks(trigger):
    track_list_string = redis_instance.hget('cache', 'table-of-cruises').decode('utf-8')
    crusies_to_track = pd.read_json(StringIO(json.loads(track_list_string)), dtype=dtype_definitions)
    expos = list(crusies_to_track['expocode'].unique())
    in_set = "','".join(expos)
    in_set = "'" + in_set + "'"
    track_data = pd.read_sql(f"SELECT * FROM tracks WHERE expocode IN ({in_set}) AND platform_type != 'Mooring'", con=postgres_engine)
    stations = pd.read_sql(f"SELECT * FROM tracks WHERE expocode IN ({in_set}) AND platform_type = 'Mooring'", con=postgres_engine)
    if track_data.shape[0] > 100_000:
        track_data = track_data.sample(n=100_000)
    track_data = track_data.sort_values(['expocode', 'time'])
    figure = px.line_geo(track_data, lat='latitude', lon='longitude', color='expocode', 
                        hover_data=['expocode', 'time', 'latitude', 'longitude'],
                        )
    figure.update_traces(line={'width': 5})
    stat_fig = px.scatter_geo(stations, lat='latitude', lon='longitude', color='expocode', 
                              hover_data=['expocode', 'time', 'latitude', 'longitude'])
    figure.add_traces(list(stat_fig.select_traces()))
    figure.update_layout(legend={'orientation' : "v", 'x': 1, 'y': 1, 'xanchor': 'right', 'yanchor': 'top'})
    figure.update_geos(fitbounds='locations', lonaxis_range=[-180,180], lataxis_range=[-90,90])
    figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan', resolution=50)
    title=f'Approximate Cruise Tracks for the Selected Cruises. {str(len(expos))} total tracks.'
    return [figure, title]


@app.callback(
    [
        Output('ll_lat', 'value', allow_duplicate=True),
        Output('ll_lon', 'value', allow_duplicate=True),
        Output('ur_lat', 'value', allow_duplicate=True),
        Output('ur_lon', 'value', allow_duplicate=True),
    ],
    [
        Input('map-info', 'data')
    ], prevent_initial_call=True
)
def set_text_lat_lon(in_map_info):
    if in_map_info is not None and len(in_map_info) > 3:
        map_info = json.loads(in_map_info)
        return [map_info['ll']['latitude'], map_info['ll']['longitude'], map_info['ur']['latitude'], map_info['ur']['longitude']]
    else:
        return [no_update, no_update, no_update, no_update]


@app.callback(
    [
        # Output('ll_lat', 'value', allow_duplicate=True),
        # Output('ll_lon', 'value', allow_duplicate=True),
        # Output('ur_lat', 'value', allow_duplicate=True),
        # Output('ur_lon', 'value', allow_duplicate=True),
        Output('map-info', 'data', allow_duplicate=True),
    ],
    [
        Input('region', 'value')
    ], prevent_initial_call=True
)
def set_bounds_from_region(region_id):
    map_info = None
    if region_id is not None and len(region_id) > 0:
        if isinstance(region_id, list):
            region_id = region_id[0]
        map_info = constants.regions[region_id]
        # DEBUG print(map_info)
    if map_info is None:
        raise exceptions.PreventUpdate
    else:
        # DEBUG print(f'fired the single region {region_id} and {map_info}')
        return [json.dumps(map_info)]
        # return [map_info['ll']['latitude'], map_info['ll']['longitude'], map_info['ur']['latitude'], map_info['ur']['longitude'], json.dumps(map_info),]









# @app.callback(
#     [
#         Output('prop-prop-x-item', 'style'),
#         Output('prop-prop-y-item', 'style'),
#         Output('prop-prop-colorby-item', 'style')
#     ],
#     [
#         Input('plot-type', 'value')
#     ]
# )
# def set_prop_prop_display(in_plot_type):
#     if in_plot_type is not None and in_plot_type == "prop-prop":
#         return [{'display': 'block'}, {'display': 'block'}, {'display':'block'}]
#     return [{'display':'none'}, {'display':'none'}, {'display':'none'}]


@app.callback(
    [
        Output('map-info', 'data', allow_duplicate=True),
        Output('region', 'value', allow_duplicate=True),
        Output('woce-co2-water', 'value', allow_duplicate=True),
        Output('start-date-picker', 'value', allow_duplicate=True),
        Output('end-date-picker', 'value', allow_duplicate=True),
        Output('investigator', 'value', allow_duplicate=True),
        Output('organization', 'value', allow_duplicate=True),
        Output('qc-flag', 'value', allow_duplicate=True),
        Output('platform-type', 'value', allow_duplicate=True),
        Output('expocode', 'value'),
        Output('ll_lat', 'value', allow_duplicate=True),
        Output('ll_lon', 'value', allow_duplicate=True),
        Output('ur_lat', 'value', allow_duplicate=True),
        Output('ur_lon', 'value', allow_duplicate=True),
    ],
    [
        Input('reset', 'n_clicks'),
        State('start-date-picker', 'min'),
        State('end-date-picker', 'max')
    ], prevent_initial_call=True
)
def reset_map(click, min_date, max_date):
    return ['', [], [], min_date, max_date, '', '', [], [], [], -90, -180, 90, 180]


def cc_color_set(index, palette):
    rgb = px.colors.convert_to_RGB_255(palette[index])
    hexi = '#%02x%02x%02x' % rgb
    return hexi

# gunicorn entry point
def get_server():
    # init_client()
    return app.server

def log(method, message, object):
    print(method + ' --> ' + message)
    if object is not None:
        pp.pprint(object)


# Experiment to see if you can zoom smoothly with this callback
@app.callback(
    Output('map-graph', 'figure', allow_duplicate=True),
    Input('map-graph', 'relayoutData'),
    State('map-graph', 'figure'),
    prevent_initial_call=True
)
def limit_map_zoom(relayout_data, figure):
    if relayout_data and 'geo.projection.scale' in relayout_data:
        #you can set these however you like
        min_zoom_scale = 0.5
        max_zoom_scale = 10.0 

        current_zoom_scale = relayout_data['geo.projection.scale']

        print(f'limiting if {current_zoom_scale} is outside min {min_zoom_scale} and max {max_zoom_scale}')

        current_lon = relayout_data.get('geo.center.lon')
        current_lat = relayout_data.get('geo.center.lat')


        fig_copy = go.Figure(figure)

        if current_zoom_scale < min_zoom_scale:
            fig_copy.update_layout(
                geo_projection_scale=min_zoom_scale,
                geo_center_lon=current_lon if current_lon is not None else figure['layout']['geo'].get('center', {}).get('lon'),
                geo_center_lat=current_lat if current_lat is not None else figure['layout']['geo'].get('center', {}).get('lat')
            )
            return fig_copy

        elif current_zoom_scale > max_zoom_scale:
            fig_copy.update_layout(
                geo_projection_scale=max_zoom_scale,
                geo_center_lon=current_lon if current_lon is not None else figure['layout']['geo'].get('center', {}).get('lon'),
                geo_center_lat=current_lat if current_lat is not None else figure['layout']['geo'].get('center', {}).get('lat')
            )
            return fig_copy

    return no_update


if __name__ == '__main__':
    app.run(debug=True, dev_tools_props_check=False)
