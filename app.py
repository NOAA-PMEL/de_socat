from datetime import date, datetime, timezone
from dash_enterprise_libraries import EnterpriseDash
import hashlib
import io
from sys import exception
from itertools import compress
import json
import os
import pprint
import urllib
import math
from io import StringIO
import diskcache
from celery import Celery
from crossover import crossover
import xarray as xr
import cf_xarray
import re
import inspect

import colorcet as cc
from dash import (
    ALL,
    Dash,
    Input,
    Output,
    State,
    clientside_callback,
    ctx,
    dcc,
    exceptions,
    html,
    no_update,
    CeleryManager,
    DiskcacheManager,
    callback_context,
    clientside_callback
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


from sqlalchemy.sql.selectable import NoInit
import util
from datetime import datetime
from dateutil import parser
import callbacks
import layout


from constants import (
    TIME_TO_LIVE,
    FULL_CRUISE_DATA_FIELD_NAME, 
    COLUMNS_FOR_WOCE_EDIT_TABLE_FIELD_NAME, 
    CROSSOVER_DATA_FIELD_NAME, 
    TABLE_OF_CRUISES_URL_FIELD_NAME, 
    CURRENT_GRID_DATA,
    dtype_definitions, 
    short_format, 
    decimated_url, 
    full_url, 
    grid_url, 
    socat_mode, 
    redis_instance, 
    postgres_engine, 
    regions, 
    region_names,
    zoom, 
    center, 
    map_limits, 
    map_height, 
    map_width
)
from blank import get_blank

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# When there will be more than 50,000 (???) points on the property property panel
# either use the decimated data set or
# segement by time to display to show the first 50000 with a time selector menu to see the remaning segments
#



pp = pprint.PrettyPrinter(indent=4)

marker_size = 6

# Sample every N hours
hours_interval = 168

visible = {'visibility': 'visible'}
hidden = {'visibility': 'hidden'}

no_display = {'display': 'none'}
display_block = {'display': ''}

center = {'lon': 0.0, 'lat': 0.0}
zoom = 1.4

x_legend = [0.0, .355, .71]
y_legend = [1.026, 0.815, 0.604, 0.393, 0.18]


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

agg_x = 72
agg_y = 36



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

socat_versions = list(edf['socat_version'].unique())
socat_version_options = []
                                            # shortest, then alphabetically
for socat_version in sorted(socat_versions, key=lambda s: (len(s), s)):
    socat_version_options.append({'value': socat_version, 'label': socat_version})

platform_names = list(edf['platform_name'].unique())
platform_name_options = []
for platform_name in sorted(platform_names):
    platform_name_options.append({'value': platform_name, 'label': platform_name})

celery_app = Celery(broker=os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"), backend=os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))
if os.environ.get("DASH_ENTERPRISE_ENV") == "WORKSPACE":
    # For testing...
    # import diskcache
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)
else:
    # For production...
    background_callback_manager = CeleryManager(celery_app)


# Define Dash application structure
app = EnterpriseDash(__name__, background_callback_manager=background_callback_manager)
server = app.server  # expose server variable for Procfile



months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
logger.debug('__app startup__ starting info query')
dinfo = Info(decimated_url)
variables, long_names, standard_name, units, v_d_types = dinfo.get_variables()
to_get = ','.join(variables)
variable_options = []
for var in variables:   
    if var != 'lat_meters' and var != 'lon_meters':
        variable_options.append({'label':var, 'value': var, 'title': var})
logger.debug('__app startup__  finished info meta')
start_date, end_date, start_seconds, end_seconds = dinfo.get_times()
logger.debug('__app startup__ finished info times')

columns_for_WOCE_edits = ["WOCE_CO2_water" ,"WOCE_CO2_atm", "fCO2_recommended", "expocode", "time", "longitude", "latitude"]


socat_release_options = [
    {'label': 'SOCAT v2025', 'value':'v2025'},
    {'label': 'SOCAT v2024', 'value':'v2024', 'disabled': True},
    {'label': 'SOCAT v2023', 'value':'v2023', 'disabled': True},
    {'label': 'SOCAT v2022', 'value':'v2022', 'disabled': True},
    {'label': 'SOCAT v2021', 'value':'v2021', 'disabled': True},
    {'label': 'SOCAT v2020', 'value':'v2020', 'disabled': True},
    {'label': 'SOCAT v2019', 'value':'v2019', 'disabled': True},
    {'label': 'SOCAT v6', 'value':'v6', 'disabled': True},
    {'label': 'SOCAT v5', 'value':'v5', 'disabled': True},
    {'label': 'SOCAT v4', 'value':'v4', 'disabled': True},
    {'label': 'SOCAT v3', 'value':'v3', 'disabled': True},
    {'label': 'SOCAT v2', 'value':'v2', 'disabled': True},
    {'label': 'SOCAT v1.5', 'value':'v1.5', 'disabled': True},

]
grid_dataset_options=[]
grid_dataset_titles = {
    'v2025': {},
    'v2024': {},
    'v2023': {},
    'v2022': {},
    'v2021': {},
    'v2020': {},
    'v2019': {},
    'v6': {},
    'v5': {},
    'v4': {},
    'v3': {},
    'v2': {},
    'v1.5': {},
}

# Read from a file until the data sets are in an accesible ERDDAP
grids = pd.read_csv("http://smokey.pmel.noaa.gov:8140/erddap/tabledap/allDatasets.csv", skiprows=[1])

grids = grids.loc[grids['title'].str.contains('SOCAT', na=False)]
for row_num, row in grids.iterrows():
    for release in grid_dataset_titles.keys():
        match = re.search(rf'\b{re.escape(release.lower())}\b', row['title'].lower())
        if match is not None:
            grid_dataset_titles[release].update({row['datasetID']: row['title']})
    grid_dataset_options.append({'label': row['title'], 'value': row['datasetID']})

# all columns: expocode,dataset_name,platform_name,platform_type,organization,geospatial_lon_min,geospatial_lon_max,geospatial_lat_min,geospatial_lat_max,time_coverage_start,time_coverage_end,investigators,socat_version,all_region_ids,socat_doi,qc_flag,sample_number,year,month,day,hour,minute,second,longitude,latitude,depth,sal,Temperature_equi,temp,Temperature_atm,Pressure_equi,Pressure_atm,xCO2_water_equi_temp_dry_ppm,xCO2_water_sst_dry_ppm,xCO2_water_equi_temp_wet_ppm,xCO2_water_sst_wet_ppm,pCO2_water_equi_temp,pCO2_water_sst_100humidity_uatm,fCO2_water_equi_uatm,fCO2_water_sst_100humidity_uatm,xCO2_atm_dry_actual,xCO2_atm_dry_interp,pCO2_atm_wet_actual,pCO2_atm_wet_interp,fCO2_atm_wet_actual,fCO2_atm_wet_interp,delta_xCO2,delta_pCO2,delta_fCO2,relative_humidity,specific_humidity,ship_speed,ship_dir,wind_speed_true,wind_speed_rel,wind_dir_true,wind_dir_rel,WOCE_CO2_water,WOCE_CO2_atm,woa_sss,pressure_ncep_slp,fCO2_insitu_from_xCO2_water_equi_temp_dry_ppm,fCO2_insitu_from_xCO2_water_sst_dry_ppm,fCO2_from_pCO2_water_water_equi_temp,fCO2_from_pCO2_water_sst_100humidity_uatm,fCO2_insitu_from_fCO2_water_equi_uatm,fCO2_insitu_from_fCO2_water_sst_100humidty_uatm,fCO2_from_pCO2_water_water_equi_temp_ncep,fCO2_from_pCO2_water_sst_100humidity_uatm_ncep,fCO2_insitu_from_xCO2_water_equi_temp_dry_ppm_woa,fCO2_insitu_from_xCO2_water_sst_dry_ppm_woa,fCO2_insitu_from_xCO2_water_equi_temp_dry_ppm_ncep,fCO2_insitu_from_xCO2_water_sst_dry_ppm_ncep,fCO2_insitu_from_xCO2_water_equi_temp_dry_ppm_ncep_woa,fCO2_insitu_from_xCO2_water_sst_dry_ppm_ncep_woa,fCO2_recommended,fCO2_source,delta_temp,region_id,calc_speed,etopo2,gvCO2,dist_to_land,day_of_year,time,lon360,tmonth,nobs_full,nobs_deci
# 
# 


# These callbacks apply to items in the layout that are only present when the QC Editor mode is active
if socat_mode == "QC_EDITOR":
    callbacks.register_editor_callbacks(app)



footer_image = app.get_asset_url(
    "logo-PMEL-lockup-light_noaaPMEL_horizontal_rgb-txt_2024.png"
)
app.layout = layout.get_layout(
    initial_expo_options, 
    start_date, 
    end_date, 
    investigaors_options, 
    variable_options, 
    organization_options, 
    socat_version_options, 
    platform_name_options, 
    full_url, 
    image_height, 
    footer_image,
    grid_dataset_options,
    socat_release_options
)
app.setup_shortcuts(size='slim', title='')

@app.callback(
    [
        Output('aggregation-controls', 'style'),
        Output('aggregation-type', 'style'),
        Output('grid-download', 'style')
    ],
    [
        Input('time-aggregations-switch', 'value')
    ]
)
def toggle_time_aggregation(in_switch):
    fname = inspect.currentframe().f_code.co_name
    logger.debug(f"__{fname}__: startd with value {in_switch}")
    if in_switch:
        return [{'display': ''}, {'display': ''}, {'display': 'none'}]
    else:
        return [{'display': 'none'}, {'display': 'none'}, {'display': ''}]


@app.callback(
    [
        Output('cruise-view', 'style'),
        Output('grid-view', 'style')
    ],
    [
        Input('viewer', 'value')
    ], prevent_initial_call=True
)
def switch_viewer(in_view):
    if in_view == "cruises":
        return [{'display': ''}, {'display': 'none'}]
    else:
        return [{'display': 'none'}, {'display': ''}]


@app.callback(
    [
        Output('grid-dataset', 'options'),
        Output('grid-dataset', 'value')
    ],
    [
        Input('grid-socat-release', 'value')
    ]
)
def set_grid_datasets(in_socat_release):
    socat_dataset_options = []
    first_dataset = ''
    if in_socat_release is not None and len(in_socat_release) > 0:
        datasets = grid_dataset_titles[in_socat_release]
        for idx, dataset in enumerate(datasets):
            if idx == 0:
                first_dataset = dataset
            only_one = not ('v2025' in dataset) # DEBUG
            socat_dataset_options.append({'label': datasets[dataset], 'value': dataset, 'disabled': only_one}) # disabled is for DEBUG
        return [socat_dataset_options, first_dataset]
    else:
        raise exceptions.PreventUpdate


@app.callback(
    [
        Output('grid-variable', 'options'),
        Output('grid-year', 'options'),
        Output('grid-year-end', 'options'),
        Output('grid-month', 'style'),
        Output('grid-month-end', 'style')
    ],
    [
        Input('grid-dataset', 'value')
    ], prevent_initial_call=True
)
def get_grid_variables(in_grid_dataset):
    grid_variable_options = []
    year_options = []
    if in_grid_dataset is not None and len(in_grid_dataset) > 0:
        gdInfo = Info(grid_url + '/' + in_grid_dataset)
        variables, long_names, standard_name, units, v_d_types = gdInfo.get_variables()
        start_date, end_date, start_date_timestamp, end_date_timestamp = gdInfo.get_times()
        for var in variables:
            grid_variable_options.append({'label': long_names[var], 'value': var})
        start_date_obj = datetime.strptime(start_date, short_format)
        end_date_object = datetime.strptime(end_date, short_format)
        syear = start_date_obj.year
        eyear = end_date_object.year
        if 'decadal' in in_grid_dataset:
            stride = 10
            show_month = {'display': 'none'}
        elif 'year' in in_grid_dataset:
            show_month = {'display': 'none'}
            stride = 1
        else:
            show_month = {'display': ''}
            stride = 1
        for y in range(syear, eyear+1, stride):
            year_options.append({'label': y, 'value': y})
    return [grid_variable_options, year_options, year_options, show_month, show_month]

#### CURRENT ATTEMPT AT SAVE IMPLEMENTATION
@app.callback(
    [
        Output('grid-title', 'title'),
        Output('grid-map', 'figure'),
        Output('grid-netcdf', 'href'),
        Output('grid-csv', 'href'),
        Output('grid-data-key', 'data')
    ],
    [
        Input('grid-dataset', 'value'),
        Input('grid-variable', 'value'),
        Input('grid-year', 'value'),
        Input('grid-month', 'value'),
        Input('time-aggregations-switch', 'value'),
        Input('grid-year-end', 'value'),
        Input('grid-month-end', 'value'),
        Input('aggregation-type', 'value')
    ],
    [
        State('grid-dataset', 'options')
    ]
)
def grid_map(in_dataset, in_variable, in_year, in_month, in_aggregate_on, in_year_end, in_month_end, in_agg_type, in_dataset_choices,):
    fname = inspect.currentframe().f_code.co_name
    if in_dataset is None or in_variable is None or in_year is None:
        logger.debug(f'__{fname}__ : no update because None input required input')
        return no_update
    elif 'month' in in_dataset and in_month is None:
        logger.debug(f'__{fname}__ : no update because no month for monthly dataset')
        return no_update
    else:
        # e.g. http://smokey.pmel.noaa.gov:8140/erddap/griddap/v2025_c716_f8c7_183a.csv?fco2_ave_unwtd[(2007-06-16):1:(2007-06-16)][(-89.5):1:(89.5)][(-179.5):1:(179.5)]
        # encoded: http://smokey.pmel.noaa.gov:8140/erddap/griddap/v2025_c716_f8c7_183a.csv?sst_ave_unwtd%5B(2011-04-16):1:(2011-04-16)%5D%5B(-89.5):1:(89.5)%5D%5B(-179.5):1:(179.5)%5D
        url = f"{grid_url}/{in_dataset}"
        ds = xr.open_dataset(url)
        if in_month is None:
            in_month = '06'
        selected_time = f"{in_year}-{in_month}-16"
        if not in_aggregate_on:
            csv_url = f"{url}.csv?{in_variable}" + urllib.parse.quote(f"[({selected_time}):1:({selected_time})][(-89.5):1:(89.5)][(-179.5):1:(179.5)]")
            encoded_url = csv_url.encode('utf-8')
            hash_object = hashlib.sha256(encoded_url)
            grid_data_key = hash_object.hexdigest()
            csv_url = 'http://smokey.pmel.noaa.gov:8140/erddap/griddap/v2025_c716_f8c7_183a.csv?sst_ave_unwtd%5B(2011-04-16):1:(2011-04-16)%5D%5B(-89.5):1:(89.5)%5D%5B(-179.5):1:(179.5)%5D'
            netcdf_url = csv_url.replace(".csv", ".nc")
            ds = ds.sel(time=selected_time, method='nearest')
            df = ds.cf.to_dataframe()
            df = df.reset_index(level=None, drop=False, inplace=False)
            df = df.dropna()
            if in_variable not in ['time', 'latitude', 'longitude']:
                sdf = df[['time', 'latitude', 'longitude', in_variable]]
            else:
                sdf = df[['time', 'latitude', 'longitude']]
            redis_instance.hset(grid_data_key, CURRENT_GRID_DATA, json.dumps(sdf.to_json()))
            dataset_name = [x['label'] for x in in_dataset_choices if x['value'] == in_dataset]
            dataset_name = dataset_name[0]
            title = f'{in_variable} for {in_year}-{in_month} from {dataset_name}'
        else:
            if in_year_end is None or in_agg_type is None:
                logger.debug(f'__{fname}__ : no update on aggregate in time because no year or agg_type')
                raise exceptions.PreventUpdate
            elif 'month' in in_dataset and in_month_end is None:
                logger.debug(f'__{fname}__ : no update on aggregate in time because no month for monthly dataset')
                return no_update
            else:
                if in_month_end is None:
                    in_month_end = '06'
                netcdf_url = ''
                csv_url = ''
                grid_data_key = 'NOT SET'
                end_time = f"{in_year_end}-{in_month_end}-16"
                ds = ds.sel(time=slice(selected_time, end_time))
                if in_agg_type == 'mean':
                    ds = ds.mean(dim='time')
                elif in_agg_type == 'min':
                    ds = ds.min(dim='time')
                elif in_agg_type == 'max':
                    ds = ds.max(dim='time')
                elif in_agg_type == 'sum':
                    ds = ds.sum(dim='time')
                df = ds.cf.to_dataframe()
                df = df.reset_index(level=None, drop=False, inplace=False)
                df = df.dropna()
                if in_agg_type == 'sum':
                    # drop zeros from sum
                    df = df[df[in_variable] != 0]

                title = f'{in_agg_type.title()} of {in_variable} from {selected_time} to {end_time}'

        # Calculate the .1 and .9 quantile and use them for the color range rounding down and up to the nearest 5 for numeric data
        if is_numeric_dtype(df[in_variable]):
            ranges = df[in_variable].quantile(q=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
            cmax = 5 * round(ranges[0.9] / 5.0)
            cmin = (ranges[0.1] // 5.0) * 5.0
            logger.debug(f'__{fname}__ color min and max for {in_variable} are {cmin}, {cmax}')
            figure = px.scatter_geo(df, lat='latitude', lon='longitude', color=in_variable, color_continuous_scale='Inferno', range_color=[cmin, cmax])
        else:
            figure = px.scatter_geo(df, lat='latitude', lon='longitude', color=in_variable)
        figure.update_layout(legend={'xanchor':'left', 'x': 0,}, margin={'t':35, 'r':0, 'l':0, 'b':0}, title={'text':title, 'y':.95, 'yanchor':'top'})
        figure.update_coloraxes(colorbar={'title': in_variable, 'lenmode':'fraction', 'len':.65, 'y':.5, 'orientation':'v', 'title_side':'right'})
        figure.update_geos(fitbounds='locations', lonaxis_range=[-180,180], lataxis_range=[-90,90])
        figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan', resolution=50)
        # Title is just for the loading indicator
        return ['', figure, netcdf_url, csv_url, grid_data_key]





@app.callback(
    [
        Output('cruise-qc-grid', 'columnDefs'),
        Output('cruise-qc-grid', 'rowData'),
        Output('cruise-qc-card-header', 'title'),
    ],
    [
        Input('plot-qc-level-tabs', 'value'),
        Input('plot-expocode', 'value'),
        Input('crossover-expocode', 'value')
    ], prevent_initial_call=True
)
def show_cruise_qc(click, expocode_to_show, crossover_expocode):
    if click != 'cruise-qc':
        return [no_update, no_update, no_update]        
    if expocode_to_show is not None and len(expocode_to_show) > 0:
        records = db.get_cruise_qc(expocode_to_show)
        columnDefs = []   
        for i in sorted(records.columns, key=str.casefold):
            columnDefs.append({"field": i, "headerName": i, 'wrapText': True, 'autoHeight': True, 'cellStyle': {"lineHeight": "unset"}})
        title = f'Cruise QC for {expocode_to_show}'
        if crossover_expocode is not None and len(crossover_expocode) > 0:
            title = title  + '  (Crossover information not shown.)'
        return [columnDefs, records.to_dict("records"), title]
    else:
        df = pd.DataFrame()
        return [[], df.to_dict("records"),'No expocode selected.']


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
#     logger.debug('season change fired ----------------------')
#     checked = [jan_sw, feb_sw, mar_sw, apr_sw, may_sw, jun_sw, jul_sw, aug_sw, sep_sw, oct_sw, nov_sw, dec_sw]
#     logger.debug('checked ----->', checked)
#     disabled = [True, True, True, True, True, True, True, True, True, True, True, True]
#     for i, check in enumerate(checked):
#         logger.debug('checking ', i, ' at = ', checked[i])
#         past =  (i-1)
#         future = ((i%12 + 1)%12)
#         if check:
#             logger.debug('checked set to false', i)
#             disabled[i] = False            
#         if checked[past] and not checked[future]:
#             logger.debug('past and not future', i)
#             disabled[i] = False
#         if not checked[past] and checked[future]:
#             logger.debug('not past and future', i)
#             disabled[i] = False
#         if check and checked[past] and checked[future]:
#             logger.debug('past and future', i)
#             disabled[i] = True
#     if disabled.count(True) == 12:
#         disabled = [False, False, False, False, False, False, False, False, False, False, False, False]
#     logger.debug('disabled ----------|', disabled)
#     logger.debug('=-=-=- done --==-=-=-=-')
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
    fname = inspect.currentframe().f_code.co_name
    logger.debug(f'__{fname}__ running setup')
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
        Output('plot-data-change', 'data', allow_duplicate=True),
        # Output('show','href'),
        Output('csv','href'),
        Output('netcdf','href'),
        Output('prop-prop-loading', 'children'),
        Output('crossover-expocode', 'options', allow_duplicate=True),
        Output('crossover-expocode', 'value', allow_duplicate=True),
        Output('crossover-message', 'children', allow_duplicate=True),
    ],
    [
        Input('plot-expocode', 'value'),
        Input('crossover-expocode', 'value'),
    ], prevent_initial_call=True
)
def cache_plot_data(in_plot_expocode, in_crossover_expocode,):
    fname = inspect.currentframe().f_code.co_name
    logger.debug(f"__{fname}__ ========== checking data cache expocode={in_plot_expocode}")

    ctx = callback_context
    if ctx.triggered_id == "plot-expocode":
        message = "Use button to check for crossovers"
        options = []
        value = ''
    else:
        message = no_update
        options = no_update
        value = no_update

    expo_con = util.make_con('expocode', in_plot_expocode)
    all_csv_url = f'{full_url}.csv?{to_get}{expo_con}'
    all_nc_url = all_csv_url.replace('csv','ncCF')
    new_data = False
    new_crossover = False
    if in_plot_expocode is not None and len(in_plot_expocode) > 0:
        key = str(in_plot_expocode)
        cache_data_for_key(key)
        new_data = True
    if in_crossover_expocode is not None and len(in_crossover_expocode) > 0:
        key = str(in_crossover_expocode)
        cache_data_for_key(key)
        new_crossover = True
    if new_data or new_crossover:
        return ['new_data', all_csv_url, all_nc_url, '', options, value, message]
    else:
        # return ['no data', full_url, full_url, full_url,'']
        return ['no', full_url, full_url,'', [], '', 'Use button to check for crossovers.']


def cache_data_for_key(key):
    fname = inspect.currentframe().f_code.co_name
    if not redis_instance.hexists(key, FULL_CRUISE_DATA_FIELD_NAME):
        url = f'{full_url}.csv?{to_get}&expocode="{key}"'
        logger.debug(f'__{fname}__ caching data from: {key}')
        df = pd.read_csv(url, skiprows=[1], dtype=dtype_definitions)
        redis_instance.hset(key, FULL_CRUISE_DATA_FIELD_NAME, json.dumps(df.to_json()))
        redis_instance.expire(key, TIME_TO_LIVE)
        logger.debug(f'__{fname}__ Data for {key} successfull cached.')


def read_cache_for_key(key):
    df_json_string = redis_instance.hget(key, FULL_CRUISE_DATA_FIELD_NAME).decode('utf-8')
    df = pd.read_json(StringIO(json.loads(df_json_string)), dtype=dtype_definitions)
    df = df.sort_values(['time'])
    return df
        

# @app.callback(
#     [
#         Output('plot-data-change', 'data', allow_duplicate=True),
#     ],
#     [
#         Input('crossover-expocode', 'value')
#     ], prevent_initial_call=True
# )
# def cache_crossover_data(in_crossover_expocode):
#     to_get = ','.join(variables)
#     if in_crossover_expocode is not None and len(in_crossover_expocode):
#         if not redis_instance.hexists(str(in_crossover_expocode), FULL_CRUISE_DATA_FIELD_NAME):
#             cross_url = f'{full_url}.csv?{to_get}&expocode="{in_crossover_expocode}"'
#             cdf =  df = pd.read_csv(cross_url, skiprows=[1])
#             redis_instance.hset(str(in_crossover_expocode), FULL_CRUISE_DATA_FIELD_NAME, json.dumps(cdf.to_json()))
#             redis_instance.expire(str(in_crossover_expocode), TIME_TO_LIVE)

#         return ['new_data']
#     else:
#         # return ['no data', full_url, full_url, full_url,'']
#         return ['no data']


@app.callback(
    [
        Output('trace-graph', 'figure', allow_duplicate=True),
        Output('trace-graph-header', 'title', allow_duplicate=True),
    ],
    [
        Input('top-level-tabs', 'value')
    ], prevent_initial_call=True
)
def reset_trace(tab):
    if tab == 'map':
        return [get_blank("Don't see the cruises you expect?<br>Go back and click the Find Cruises button."), no_update]
    elif tab =='table':
        return [get_blank("Plotting the selected cruise.<br>Switch cruises using the menu."), no_update]
    else:
        return [no_update, no_update]


@app.callback(
    [
        Output('trace-graph', 'figure', allow_duplicate=True),
        Output('trace-graph-header', 'title', allow_duplicate=True),
    ],
    [
        Input('plot-data-change','data'),
        Input('trace-variable', 'value'),
    ],
    [
        State('plot-expocode', 'value'), 
    ], prevent_initial_call=True
)
def update_trace(in_change, trace_in_variable, trace_in_expocode):
    fname = inspect.currentframe().f_code.co_name
    if trace_in_variable is None or len(trace_in_variable) < 1:
        trace_in_variable = 'fCO2_recommended'
    
    logger.debug(f'__{fname}__ remaking trace plot')


    if trace_in_expocode is not None and len(trace_in_expocode) > 0:
        if redis_instance.hexists(str(trace_in_expocode), FULL_CRUISE_DATA_FIELD_NAME):
            df = read_cache_for_key(str(trace_in_expocode))
        else:
            # Wait for the plot-data-change to load the cache
            raise exceptions.PreventUpdate
    else:
        return [get_blank("Don't see the cruises you expect?<br>Go back and click the Find Cruises button."), no_update]  

    df = df.loc[df[trace_in_variable].notna()]
    if df.shape[0] > 1:
        title = f'All {trace_in_variable} data from {str(trace_in_expocode)}'
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
        figure.update_layout(legend={'xanchor':'left', 'x': 0})
        figure.update_geos(fitbounds='locations', lonaxis_range=[-180,180], lataxis_range=[-90,90])
        figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan', resolution=50)

    else:
        figure = get_blank(f'No data found for {trace_in_variable}.')
        title = f'No data found for {trace_in_variable}.'
    logger.debug(f'__{fname}__ returning value from trace of {trace_in_expocode}')
    return [figure, title]



@app.callback(
    [
        Output('crossover-trace-graph', 'figure', allow_duplicate=True),
        Output('crossover-trace-graph-header', 'title', allow_duplicate=True),
    ],
    [
        Input('crossover-endpoints', 'data')
    ],
    [
        State('crossover-trace-variable', 'value'),
        State('plot-expocode', 'value'), 
        State('crossover-expocode', 'value')
    ], prevent_initial_call=True
)
def update_crossover_trace(trace_in_endpoints, trace_in_variable, trace_in_expocode, trace_in_crossover_expocode):
    if trace_in_variable is None or len(trace_in_variable) < 1:
        trace_in_variable = 'fCO2_recommended'

    if trace_in_endpoints:
        ends = json.loads(trace_in_endpoints)
        tmin = ends['tmin']
        tmax = ends['tmax']    
    else:
        return no_update
    
    return _make_crossover_trace_helper(trace_in_variable, trace_in_expocode, trace_in_crossover_expocode, tmin=tmin, tmax=tmax)


def _make_crossover_trace_helper(trace_in_variable, trace_in_expocode, trace_in_crossover_expocode, tmin=None, tmax=None):
    fname = inspect.currentframe().f_code.co_name
    if trace_in_variable is None or len(trace_in_variable) < 1:
        trace_in_variable = 'fCO2_recommended'
    
    logger.debug(f'__{fname}__ making trace plot with filter {tmin} and {tmax}')


    if trace_in_expocode is not None and len(trace_in_expocode) > 0:
        if redis_instance.hexists(str(trace_in_expocode), FULL_CRUISE_DATA_FIELD_NAME):
            df = read_cache_for_key(str(trace_in_expocode))
        else:
            # Wait for the plot-data-change to load the cache
            raise exceptions.PreventUpdate
    else:
        return [get_blank("Don't see the cruises you expect?<br>Go back and click the Find Cruises button."), no_update]  

    cdf = None
    if trace_in_crossover_expocode is not None and len(trace_in_crossover_expocode) > 0:
        logger.debug(f'__{fname}__ reading crossover cache')
        if redis_instance.hexists(str(trace_in_crossover_expocode), FULL_CRUISE_DATA_FIELD_NAME):
            logger.debug(f'__{fname}__ reading crossover cache')
            cdf = read_cache_for_key(str(trace_in_crossover_expocode))
        else:
            raise exceptions.PreventUpdate

    df = df.loc[df[trace_in_variable].notna()]
    if df.shape[0] > 1:
        if tmax is None:
            tmax = df['time'].max()
        if tmin is None:
            tmin = df['time'].min()
        tmin_obj = parser.parse(tmin)
        tmax_obj = parser.parse(tmax)
        logger.debug(f'__{fname}__ filtering cruise between {tmin_obj.isoformat()} and {tmax_obj.isoformat()}')
        df = df[df['time'].between(tmin_obj.isoformat(), tmax_obj.isoformat(), inclusive='both')]
        logger.debug(f'__{fname}__ filter left {df.shape[0]} points the the cruise.')
        title = f'All {trace_in_variable} data from {str(trace_in_expocode)}'
        if cdf is not None:
            cdf = cdf.loc[cdf[trace_in_variable].notna()]
            if tmax is None:
                tmax = cdf['time'].max()
            if tmin is None:
                tmin = cdf['time'].min()
            tmin_obj = parser.parse(tmin)
            tmax_obj = parser.parse(tmax)
            logger.debug(f'__{fname}__ filtering crossover between {tmin_obj.isoformat()} and {tmax_obj.isoformat()}')
            cdf = cdf[cdf['time'].between(tmin_obj.isoformat(), tmax_obj.isoformat(), inclusive='both')]
            logger.debug(f'__{fname}__ filter left {cdf.shape[0]} points the the crossover.')
            if cdf.shape[0] > 1:
                df = pd.concat([df, cdf])
                title = title + f' and {trace_in_crossover_expocode}'
        df['time_str'] = df['time'].astype(str)
        rmin = df[trace_in_variable].min()
        rmax = df[trace_in_variable].max()
        figure = px.scatter_geo(df,
                                lat='latitude',
                                lon='longitude',
                                color=trace_in_variable,
                                color_continuous_scale='Viridis',
                                hover_data=['expocode','time','latitude','longitude',trace_in_variable],
                                range_color=[rmin,rmax], custom_data=['time_str', 'expocode'],)
        figure.update_traces(marker={'size':6})
        figure.update_coloraxes(colorbar={'orientation':'v', 'title_side':'right'})
        figure.update_layout(legend={'xanchor':'left', 'x': 0})
        figure.update_geos(fitbounds='locations', lonaxis_range=[-180,180], lataxis_range=[-90,90])
        figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan', resolution=50)

        if cdf is not None:
            if redis_instance.hexists(str(trace_in_expocode), CROSSOVER_DATA_FIELD_NAME):
                crosses_json_string = redis_instance.hget(str(trace_in_expocode), CROSSOVER_DATA_FIELD_NAME)
                crosses = json.loads(crosses_json_string)
                crossing_lat = crosses[trace_in_crossover_expocode]['crossing_lat']
                crossing_lon = crosses[trace_in_crossover_expocode]['crossing_lon']
                # Add a cross at the crossover
                figure.add_trace(
                    go.Scattergeo(
                        lat=[crossing_lat],
                        lon=[crossing_lon],
                        mode="markers",
                        name='Crossover',
                        marker=dict(
                            symbol="x", # Set the marker symbol to 'x'
                            line_color="black",
                            color="grey",
                            line_width=2,
                            opacity=0.5,
                            size=16,         
                        ),
                        hoverinfo="name", # Only show the name on hover for this point
                    )
                )
        # figure.update_coloraxes(colorbar={'orientation':'h', 'thickness':20, 'y': -.175, 'title': None})
    else:
        figure = get_blank(f'No data found for {trace_in_variable}.')
        title = f'No data found for {trace_in_variable}.'
    logger.debug(f'__{fname}__ returning trace of {trace_in_expocode}')
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
    fname = inspect.currentframe().f_code.co_name
    map_type = 'geo'
    logger.debug(f'__{fname}__ firing update map')
    if ctx.triggered_id == 'top_tab_value' and top_tab_value != 'map':
        logger.debug(f'__{fname}__  not updating map {ctx.triggered_id} and {top_tab_value}')
        return no_update

    try:
        df = pd.read_sql('SELECT * from map_counts', con=postgres_engine)
    except Exception as e:
        logger.error(e)
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
        logger.debug(f'__{fname}__ using region_id of {region_id}')
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
            logger.debug(f'__{fname}__ adding mask')
            figure.add_traces(list(mask_trace.select_traces()))
    else:
        figure = px.scatter_map(df, lat='latitude', lon='longitude', color='fCO2_recommended', 
                                hover_data=['latitude', 'longitude'], 
                                hover_name='expocode', color_continuous_scale='Viridis')
        figure.update_traces(marker={'size': marker_size})
        figure.update_coloraxes(colorbar={'orientation':'v', 'title_side':'right', })

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
        # Output('ll_lat', 'value'),
        # Output('ll_lon', 'value'),
        # Output('ur_lat', 'value'),
        # Output('ur_lon', 'value'),
        Output('map-info', 'data'),
        Output('region', 'value'),
    ],
    [
        Input('map-graph','selectedData')
    ], prevent_initial_call=True
)
def selectData(selectData):
    fname = inspect.currentframe().f_code.co_name
    logger.debug('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
    logger.debug(f"__{fname}__ {selectData}")
    logger.debug('-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-')
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
    # This is either changing to the whole globe or a selected region. Reset the region menu to nothing in either case.
    if map_info is None:
        raise exceptions.PreventUpdate
    else:
        # return [map_info['ll']['latitude'], map_info['ll']['longitude'], map_info['ur']['latitude'], map_info['ur']['longitude'], json.dumps(map_info), '']
        return [json.dumps(map_info), '']


@app.callback(
    [
        Output('show-data-grid', 'columnDefs'),
        Output('show-data-grid', 'rowData'),
        Output('show-data-header', 'title')
    ],
    [
        Input('show-button', 'n_clicks'),
    ],
    [
        State('plot-expocode', 'value'),
        State('plot-data-change', 'data'),
    ], prevent_initial_call=True
)
def show_cruise_or_grid(click, plot_in_expocode, plot_data_store):
    fname = inspect.currentframe().f_code.co_name
    if plot_in_expocode is None or len(plot_in_expocode) == 0:
        raise exceptions.PreventUpdate

    if plot_data_store == 'no':
        logger.debug(f'__{fname}__ no new data')
        raise exceptions.PreventUpdate
    
    
    if plot_in_expocode is not None and len(plot_in_expocode) > 0:
        logger.debug(f'__{fname}__ showing data for {plot_in_expocode}')
        if redis_instance.hexists(str(plot_in_expocode), FULL_CRUISE_DATA_FIELD_NAME):
            df = read_cache_for_key(str(plot_in_expocode))
        else:
            raise exceptions.PreventUpdate
        df.dropna(axis=1, how='all', inplace=True)
        columnDefs = []
        for column in df.columns:
            columnDefs.append({'field': column, 'headerName': column})
        logger.debug(f'__{fname}__ returning data for cruise')
        return [columnDefs, df.to_dict("records"), f'Data for {plot_in_expocode}']

    else:
        return [[], {}, 'No cruse found']
   

@app.callback(
    [
        Output('grid-show-data-grid', 'columnDefs'),
        Output('grid-show-data-grid', 'rowData'),
        Output('grid-show-data-header', 'title')
    ],
    [
        Input('grid-show-button', 'n_clicks')
    ],
    [
        State('grid-data-key', 'data')
    ], prevent_initial_call=True
)
def show_grid(grid_click, in_grid_data_key):
    fname = inspect.currentframe().f_code.co_name
    if in_grid_data_key is not None and len(in_grid_data_key) > 0:
        if redis_instance.hexists(str(in_grid_data_key), CURRENT_GRID_DATA):
            df_json_string = redis_instance.hget(in_grid_data_key, CURRENT_GRID_DATA).decode('utf-8')
            df = pd.read_json(StringIO(json.loads(df_json_string)), convert_dates=['time']) # TODO type definitions?
            columnDefs = []
            for column in df.columns:
                columnDefs.append({'field': column, 'headerName': column})
            logger.debug(f'__{fname}__ returning data for gridded summary')
            return [columnDefs, df.to_dict("records"), f'Data for gridded summary plot.']
        else:
            return [[], {}, 'No gridded data found']
    else:
        return [[], {}, 'No gridded data found']



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
    fname = inspect.currentframe().f_code.co_name
    out_expocode = None
    # DEBUG 
    # print('=-=-=-=-=- starting set_platform_code_from_map =-=-=-=-=-=')
    # print('printing click')
    # print(str(in_click))
    if in_click is not None:
        logger.debug(f'__{fname}__ getting first point')
        fst_point = in_click['points'][0]
        logger.debug(f'__{fname}__ First clicked point = {fst_point}')
        if 'customdata' in fst_point:
            out_value = fst_point['customdata'][0]
            logger.debug(f'__{fname}__ expo to add because of click {out_value}')
        else:
            logger.debug(f'__{fname}__ no custom data in click')
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
def make_property_property(plot_data_store, in_prop_prop_x, in_prop_prop_y, in_prop_prop_colorby, plot_in_expocode):
    fname = inspect.currentframe().f_code.co_name
    in_map_variable = 'fCO2_recommended'
    x_label = None
    y_label = None
    legend_title = None
    logger.debug(f'__{fname}__ updating the property-propery plot {plot_in_expocode}')

    if plot_in_expocode is None or len(plot_in_expocode) == 0:
        logger.debug(f'__{fname}__ data-plot: no expo')
        return [get_blank('Choose an expocode from the menu at right.'), 'No expocode selected.']
    if in_map_variable is None or len(in_map_variable) == 0:
        logger.debug(f'__{fname}__ data-plot: no variable')
        raise exceptions.PreventUpdate
    if plot_data_store == 'no':
        logger.debug(f'__{fname}__ no new data')
        raise exceptions.PreventUpdate
    if plot_in_expocode is not None and len(plot_in_expocode) > 0:
        card_title = f'{in_prop_prop_y} vs {in_prop_prop_x} colored by {in_prop_prop_colorby} from {plot_in_expocode}'
        key = str(plot_in_expocode)
        if redis_instance.hexists(key, FULL_CRUISE_DATA_FIELD_NAME):
            df = read_cache_for_key(key)
        else:
            return exceptions.PreventUpdate
        plot_data = df[columns_for_WOCE_edits]
        redis_instance.hset(str(plot_in_expocode), COLUMNS_FOR_WOCE_EDIT_TABLE_FIELD_NAME, json.dumps(plot_data.to_json()))
        redis_instance.expire(str(plot_in_expocode), TIME_TO_LIVE)
                
                
    else:
        return [get_blank('Select an expocode form the menu.'), no_update] 


    if df.shape[0] < 1:
        raise exceptions.PreventUpdate

    df['expocode'] = df['expocode'].astype(str)
    df['WOCE_CO2_water'] = df['WOCE_CO2_water'].astype(str)
    df['WOCE_CO2_atm'] = df['WOCE_CO2_atm'].astype(str)

    if in_prop_prop_colorby == 'expocode':
        cmap = px.colors.qualitative.Light24
    else:
        cmap = px.colors.qualitative.Dark24

    logger.debug(f'__{fname}__ making property-property-plot')

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

    figure.update_layout(margin={'t': 40})
    return[figure, card_title]


@app.callback(
    [
        Output('crossover-timeseries', 'figure'),
        Output('crossover-endpoints', 'data')
    ],
    [
        Input('plot-data-change', 'data'),
        Input('crossover-trace-variable', 'value'),
    ],
    [

        State('plot-expocode', 'value'),
        State('crossover-expocode', 'value'),
    ], prevent_initial_call=True
)
def make_crossover_timeseries(plot_data_store, in_trace_variable, plot_in_expocode, plot_in_crossover_expocode):
    fname = inspect.currentframe().f_code.co_name
    in_trace_variable = 'fCO2_recommended'
    legend_title = None
    logger.debug(f'__{fname}__ updating the property-propery plot ' + str(plot_in_expocode))

    if plot_in_expocode is None or len(plot_in_expocode) == 0:
        logger.debug(f'__{fname}__ data-plot: no expo')
        return [get_blank('Choose an expocode from the menu at right.'), 'No expocode selected.']
    if in_trace_variable is None or len(in_trace_variable) == 0:
        logger.debug(f'__{fname}__ data-plot: no variable')
        raise exceptions.PreventUpdate
    if plot_data_store == 'no':
        logger.debug(f'__{fname}__ no new data')
        raise exceptions.PreventUpdate
    if plot_in_expocode is not None and len(plot_in_expocode) > 0:
        card_title = f'Timeseries of {in_trace_variable} from {plot_in_expocode}'
        key = str(plot_in_expocode)
        if redis_instance.hexists(key, FULL_CRUISE_DATA_FIELD_NAME):
            df = read_cache_for_key(key)
        else:
            return exceptions.PreventUpdate
        plot_data = df[columns_for_WOCE_edits]
        redis_instance.hset(str(plot_in_expocode), COLUMNS_FOR_WOCE_EDIT_TABLE_FIELD_NAME, json.dumps(plot_data.to_json()))
        redis_instance.expire(str(plot_in_expocode), TIME_TO_LIVE)
        crosses = None
        if plot_in_crossover_expocode is not None and len(plot_in_crossover_expocode) > 0:
            card_title = card_title + f' and {plot_in_crossover_expocode}'
            if redis_instance.hexists(str(plot_in_expocode), FULL_CRUISE_DATA_FIELD_NAME):
                cdf = read_cache_for_key(str(plot_in_crossover_expocode))
                if cdf.shape[0] > 0:
                    df = pd.concat([df, cdf])
            else:
                raise exceptions.PreventUpdate

            if redis_instance.hexists(str(plot_in_expocode), CROSSOVER_DATA_FIELD_NAME):
                logger.debug(f'__{fname}__ crosses found for {plot_in_expocode}')
                crosses_json_string = redis_instance.hget(str(plot_in_expocode), CROSSOVER_DATA_FIELD_NAME)
                crosses = json.loads(crosses_json_string)
                
                
    else:
        return [get_blank('Select an expocode form the menu.'), no_update] 


    if df.shape[0] < 1:
        raise exceptions.PreventUpdate

    df['expocode'] = df['expocode'].astype(str)
    df['WOCE_CO2_water'] = df['WOCE_CO2_water'].astype(str)
    df['WOCE_CO2_atm'] = df['WOCE_CO2_atm'].astype(str)
    df['time_str'] = df['time'].astype(str)
    logger.debug(f'__{fname}__ making crossover-timeseries plot')

    # for now no choice of color for plot dots
    cmap = px.colors.qualitative.Light24
    pd.set_option('display.max_rows', None)
    print(df.dtypes)
    figure = px.scatter(df,
                        x='time',
                        y=in_trace_variable,
                        color='expocode',
                        hover_name='expocode',
                        hover_data=['time',in_trace_variable, 'expocode'],
                        custom_data=['time_str', 'expocode'],
                        color_discrete_sequence=cmap,
                        category_orders={"WOCE_CO2_water": ["2", "3", "4", "5", "1"]},
                        color_continuous_scale=px.colors.sequential.Viridis,
        )

    if crosses:
        logger.debug(f'__{fname}__ plotting cross vertical line.')
        crossing_date_time = crosses[plot_in_crossover_expocode]['crossing_date']
        logger.debug(f'__{fname}__ This is the crossing date/time {crossing_date_time}')
        crossing_time = datetime.fromisoformat(crossing_date_time)
        crossing_time.replace(tzinfo=timezone.utc)
        logger.debug(f'__{fname}__ time in millis {crossing_time.timestamp()*1000}')
        logger.debug(f'__{fname}__ ISO string passed to add_shame {crossing_time.isoformat()}')
        # Adding a collapsed shape instead of a vline works.
        figure.add_shape(
            type="line",
            xref="x",
            yref="paper", # 'paper' makes the line span the entire height (0 to 1) of the plot area
            x0=crossing_time.isoformat(), # Pass as an unambiguous string
            y0=0,
            x1=crossing_time.isoformat(),
            y1=1,
            line=dict(
                color="black",
                width=2,
                dash="dash",
            ),
        )
        figure.add_annotation(
            x=crossing_time.isoformat(),
            y=1.09, # Position at the top (relative to paper yref)
            xref="x",
            yref="paper",
            text=f"Crossover Time = {crossing_time.isoformat()}",
            showarrow=False,
            # yshift=-15,
            xshift=135
        )

    figure.update_layout(margin={'t': 40})
    endpoints = {'tmin': df['time'].min(), 'tmax': df['time'].max()}
    endpoints_string = json.dumps(endpoints)
    return[figure, endpoints_string]


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
        State('crossover-expocode', 'value')
 
    ], prevent_initial_call=True
)
def make_thumbnails(plot_data_store, plot_in_expocode, plot_in_crossover_expocode):
    fname = inspect.currentframe().f_code.co_name
    logger.debug(f'__{fname}__ updating the thumbnail plots ' + str(plot_in_expocode))

    if plot_in_expocode is None or len(plot_in_expocode) == 0:
        logger.debug(f'__{fname}__ data-plot: no expo')
        return [get_blank('Choose an expocode from the menu.'), 'No expocode seledcted.']

    if plot_data_store == 'no':
        logger.debug(f'__{fname}__ no new data')
        raise exceptions.PreventUpdate
    

    card_title = ""
    if plot_in_expocode is not None and len(plot_in_expocode) > 0:
        card_title = f'Property Property Thumbnails for {plot_in_expocode}'
        if redis_instance.hexists(str(plot_in_expocode), FULL_CRUISE_DATA_FIELD_NAME):
            df = read_cache_for_key(str(plot_in_expocode))
        else:
            logger.debug(f'__{fname}__ cache hit failed -=-=-=-=-=-=-=-=-=-=-')
            raise exceptions.PreventUpdate

    else:
        return [get_blank('Select an expocode form the menu.'), no_update] 

    crosses = None
    if plot_in_crossover_expocode is not None and len(plot_in_crossover_expocode) > 0:
        card_title = card_title + '  (Crossover data is not shown in the thumbnail plots.)'


    if df.shape[0] < 1:
        print('No data to plot')
        raise exceptions.PreventUpdate

    df['expocode'] = df['expocode'].astype(str)
    df['WOCE_CO2_water'] = df['WOCE_CO2_water'].astype(str)

    cmap = px.colors.qualitative.Light24

    
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
    logger.debug(f'__{fname}__ returning figure and title')
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
    fname = inspect.currentframe().f_code.co_name
    if cell is not None:
        logger.debug(f"__{fname}__ clicked on cell value:  {cell['value']}, column:   {cell['colId']}, row index:   {cell['rowIndex']}")
        if cell['colId'] == 'prop':
            return [cell['value'], 'plots', 'prop-prop-plot',]
        elif cell['colId'] == 'thumbnails':
            return [cell['value'], 'plots', 'prop-prop-thumbs',]
        elif cell['colId'] == 'CruiseQC':
            # TODO this isn't correct
            return [cell['value'], 'plots', 'cruise-qc',]
        else:
            raise exceptions.PreventUpdate
    else:
        raise exceptions.PreventUpdate


@app.callback(
    [
        Output('table-of-cruises', 'rowData'),
        Output('table-of-cruises', 'columnDefs'),
        Output('plot-expocode', 'options'),
        Output('plot-expocode', 'value'),
        Output('make-cruise-tracks', 'data'),
        Output('track-data-loading', 'children'),
        Output('crossover-expocode', 'options', allow_duplicate=True),
        Output('crossover-expocode', 'value', allow_duplicate=True),
        Output('crossover-message', 'children', allow_duplicate=True),
        Output('cruise_table_url', 'data')
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
        State('valid_data', 'value'),
        State('organization', 'value'),
        State('socat-version', 'value'),
        State('qc-flag', 'value'),
        State('platform-name', 'value'),
        State('platform-type', 'value'),
        State('map-info', 'data')
    ], prevent_initial_call=True, background=True
)
def make_table_of_crusies(da_click, mt_in_expocodes, mt_in_start_date, mt_in_end_date, mt_in_woce_water, mt_in_regions, mt_in_investigator, mt_in_valid_data, mt_in_org, mt_in_version, mt_in_qc_flag, mt_in_platform_name, mt_in_platform_type, mt_in_map_info):
    fname = inspect.currentframe().f_code.co_name
    if da_click == "map" or da_click == 'plots':
        return [no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update]
    vars_to_get = ['expocode', 'platform_name',	'platform_type', 'investigators', 'qc_flag', 'socat_version']
    valid_con = ''
    if mt_in_valid_data is not None and len(mt_in_valid_data)>0:
        for var in mt_in_valid_data:
            valid_con = valid_con + f'&{var}!=NaN'
    expo_con = util.make_con('expocode', mt_in_expocodes)
    time_con = '&time>='+mt_in_start_date+'&time<='+mt_in_end_date
    investigator_con = util.make_con('investigators', mt_in_investigator)
    ver_con = util.make_con('socat_version', mt_in_version)
    name_con = util.make_con('platform_name', mt_in_platform_name)
    org_con = util.make_con('organization', mt_in_org)
    if org_con:
        vars_to_get.append('organization')
    qc_flag_con = util.make_con('qc_flag', mt_in_qc_flag)
    platform_type_con = util.make_con('platform_type', mt_in_platform_type)
    woce_water_con = util.make_con('WOCE_CO2_water', mt_in_woce_water)
    region_con = util.make_con('region_id', mt_in_regions)
    if region_con:
        vars_to_get.append('region_id')

    base_url = decimated_url
    if socat_mode == "QC_EDITOR":
        base_url = full_url
    url = base_url + '.csv?' + ','.join(vars_to_get) + expo_con + valid_con + region_con + time_con + woce_water_con + investigator_con + org_con + ver_con + qc_flag_con + name_con + platform_type_con + region_con
    if not region_con and mt_in_map_info is not None and len(mt_in_map_info) > 3:
        bounds = json.loads(mt_in_map_info)
        cons = maputil.get_socat_subset(bounds['ll']['longitude'], bounds['ur']['longitude'],bounds['ll']['latitude'],bounds['ur']['latitude'])
        url = url + cons['lat'] + cons['lon']
    url = url + '&distinct()'
    expo_options = []
    logger.debug(f'__{fname}__ table URL: ' + url)
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
        {'field':'investigators', 'headerName': 'Investigators', 'filter': True},
        {'field': 'platform_name', 'headerName': 'Platform Name', 'filter': True},
        {'field': 'qc_flag', 'headerName': 'QC Flag', 'filter': True},
        {'field': 'socat_version', 'headerName': 'SOCAT Version', 'filter': True}
    ]  

    encoded_url = url.encode('utf-8')
    # Create a SHA-256 hash object
    hash_object = hashlib.sha256(encoded_url)
    hash_digest = hash_object.hexdigest()
    if df.shape[0] > 0:
        expos = sorted(list(df['expocode']))
        for code in expos:
            expo_options.append({'label': code, 'value': code})
        expo_value = expos[0]
        # Cache the table so we can extract the version number when/if we go to save a QC entry
        # and so we can make a map from the locations database
        # TODO how to make this unique (with a hash of the URL)
        redis_instance.hset(hash_digest, TABLE_OF_CRUISES_URL_FIELD_NAME, json.dumps(df.to_json()))
        return [df.to_dict("records"), table_of_cruises_columnDefs, expo_options, expo_value, 'go', '', [], '', 'Use button to check for crossovers.', hash_digest]
    else:
        df = pd.DataFrame(columns=['no_data'])
        redis_instance.hset('no_data', 'table-of-cruises', json.dumps(df.to_json()))
        tcd = [{'field': 'no_data', 'headerName': 'No matching cruises found...'}]
        return [df.to_dict("records"), tcd, {}, '', 'go', '', [], '', 'Use button to check for crossovers.', hash_digest]


@app.callback(
    [
        Output('cruise-tracks', 'figure'),
        Output('cruise-tracks-header', 'title')
    ],
    [
        Input('make-cruise-tracks', 'data')
    ],
    [
        State('cruise_table_url', 'data')
    ], prevent_initial_call = True
)
def make_cruise_tracks(trigger, hash_digest):
    track_list_string = redis_instance.hget(hash_digest, TABLE_OF_CRUISES_URL_FIELD_NAME).decode('utf-8')
    crusies_to_track = pd.read_json(StringIO(json.loads(track_list_string)), dtype=dtype_definitions)
    if crusies_to_track.shape[0] > 0:
        expos = list(crusies_to_track['expocode'].unique())
        in_set = "','".join(expos)
        in_set = "'" + in_set + "'"
        ex_con = util.make_query('expocode', expos, False)
        track_data = pd.read_sql(f"SELECT * FROM tracks WHERE {ex_con} AND platform_type != 'Mooring'", con=postgres_engine)
        stations = pd.read_sql(f"SELECT * FROM tracks WHERE {ex_con} AND platform_type = 'Mooring'", con=postgres_engine)
        if track_data.shape[0] > 100_000:
            track_data = track_data.sample(n=100_000)
        track_data = track_data.sort_values(['expocode', 'time'])
        figure = px.line_geo(track_data, lat='latitude', lon='longitude', color='expocode', 
                            hover_data=['expocode', 'time', 'latitude', 'longitude'],
                            )
        figure.update_traces(line={'width': 3})
        stat_fig = px.scatter_geo(stations, lat='latitude', lon='longitude', color='expocode', 
                                hover_data=['expocode', 'time', 'latitude', 'longitude'])
        figure.add_traces(list(stat_fig.select_traces()))
        if len(expos) > 500:
            figure.update_layout(showlegend=False)
            figure.add_annotation(
                text="Too many cruises<br>to show legend...",
                showarrow=False,
                yref="paper",
                yanchor="top",
                y=.98,
                xanchor="right",
                xref="paper",
                x=.98    
            )
        else:
            figure.update_layout(legend={'orientation' : "v", 'x': 1, 'y': 1, 'xanchor': 'right', 'yanchor': 'top'})
        figure.update_geos(fitbounds='locations', lonaxis_range=[-180,180], lataxis_range=[-90,90])
        figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan', resolution=50)
        title=f'Approximate Cruise Tracks for the Selected Cruises. {str(len(expos))} total tracks.'
        return [figure, title]
    else:
        return[get_blank('No matching cruises found...'), 'No matching cruises found...']


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
    fname = inspect.currentframe().f_code.co_name
    map_info = None
    if region_id is not None and len(region_id) > 0:
        if isinstance(region_id, list):
            region_id = region_id[0]
        map_info = regions[region_id]
        logger.debug(f'__{fname}__ {map_info}')
    if map_info is None:
        raise exceptions.PreventUpdate
    else:
        logger.debug(f'__{fname}__ fired the single region {region_id} and {map_info}')
        return [json.dumps(map_info)]
        # return [map_info['ll']['latitude'], map_info['ll']['longitude'], map_info['ur']['latitude'], map_info['ur']['longitude'], json.dumps(map_info),]


@app.callback(
    [
        Output("crossover-expocode", "options"),
        Output('crossover-message', "children")
    ],
    [
        Input("check-crossovers", "n_clicks")
    ],
    [
        State("plot-expocode", "value")
    ], prevent_initial_call=True, background=True
)
def check_crossovers(button_click, plot_expo):
    options = []
    message = "No crossovers found."
    if plot_expo is not None:
        if redis_instance.hexists(str(plot_expo), CROSSOVER_DATA_FIELD_NAME):
            crosses_json_string = redis_instance.hget(str(plot_expo), CROSSOVER_DATA_FIELD_NAME)
            crosses = json.loads(crosses_json_string)
        else:
            crosses = crossover("expocode", plot_expo, decimated_url)
        if crosses:
            options = []
            for cross in crosses:
                options.append({"label": cross, "value": cross})
            if len(options) == 1:
                message = f"{str(len(options))} crossover found."
            else:
                message = f"{str(len(options))} crossovers found."

            redis_instance.hset(plot_expo, CROSSOVER_DATA_FIELD_NAME, json.dumps(crosses))
            redis_instance.expire(str(plot_expo), TIME_TO_LIVE)

    return [options, message]


@app.callback(
    [
        Output('map-info', 'data', allow_duplicate=True),                      # 1
        Output('region', 'value', allow_duplicate=True),                       # 2
        Output('woce-co2-water', 'value', allow_duplicate=True),               # 3
        Output('start-date-picker', 'value', allow_duplicate=True),            # 4
        Output('end-date-picker', 'value', allow_duplicate=True),              # 5
        Output('investigator', 'value', allow_duplicate=True),                 # 6
        Output('organization', 'value', allow_duplicate=True),                 # 7
        Output('socat-version', 'value', allow_duplicate=True),                # 8
        Output('qc-flag', 'value', allow_duplicate=True),                      # 9
        Output('platform-name', 'value', allow_duplicate=True),                #10
        Output('platform-type', 'value', allow_duplicate=True),                #11
        Output('expocode', 'value'),                                           #12
        Output('ll_lat', 'value', allow_duplicate=True),                       #13
        Output('ll_lon', 'value', allow_duplicate=True),                       #14
        Output('ur_lat', 'value', allow_duplicate=True),                       #15
        Output('ur_lon', 'value', allow_duplicate=True),                       #16
        Output('active-constraints', 'children', allow_duplicate=True)         #17
    ],
    [
        Input('reset', 'n_clicks'),
        State('start-date-picker', 'min'),
        State('end-date-picker', 'max')
    ], prevent_initial_call=True
)
def reset_map(click, min_date, max_date):
    return [[], [], [], min_date, max_date, [], [], [], [], [], [], [], -90, -180, 90, 180, []]


@app.callback(
    [
        Output('active-constraints', 'children', allow_duplicate=True),
        Output('reset', 'style')
    ],
    [
        Input('map-info', 'data'),
        Input('region', 'value'),
        Input('woce-co2-water', 'value'),
        Input('start-date-picker', 'value'),
        Input('end-date-picker', 'value'),
        Input('investigator', 'value'),
        Input('organization', 'value'),
        Input('socat-version', 'value'),
        Input('qc-flag', 'value'),
        Input('platform-name', 'value'),
        Input('platform-type', 'value'),
        Input('expocode', 'value'),
        Input('ll_lat', 'value'),
        Input('ll_lon', 'value'),
        Input('ur_lat', 'value'),
        Input('ur_lon', 'value'),
    ], prevent_initial_call=True
)
def show_active_constraints(
    in_map_info, 
    in_region, 
    in_woce_water, 
    in_start_date, 
    in_end_date, 
    in_investigator, 
    in_organization, 
    in_socat_version, 
    in_qc_flag, 
    in_platform_name, 
    in_platform_type, 
    in_expocode,
    in_ll_lat,
    in_ll_lon,
    in_ur_lat,
    in_ur_lon):
    
    display_elements = []
    if in_ur_lon:
        if in_ur_lon != 180:
            p = html.P(children=f"Longitude is West of {in_ur_lon}")
            display_elements.append(p)
    if in_ur_lat:
        if in_ur_lat != 90:
            p = html.P(children=f"Latitude is South of {in_ur_lat}")
            display_elements.append(p)
    if in_ll_lon:
        if in_ll_lon != -180:
            p = html.P(children=f"Longitude is East of {in_ll_lon}")
            display_elements.append(p)
    if in_ll_lat:
        if in_ll_lat != -90:
            p = html.P(children=f"Latitude is North of {in_ll_lat}")
            display_elements.append(p)
    if in_platform_type:
        p = html.P(children=[f"Platform type is one of {in_platform_type}"])
        display_elements.append(p)
    if in_platform_name:
        p = html.P(children=[f"Platform name is one of {in_platform_name}"])
        display_elements.append(p)
    if in_qc_flag:
        p = html.P(children=[f"QC Flag is one of {in_qc_flag}"])
        display_elements.append(p)
    if in_socat_version:
        p = html.P(children=[f"SOCAT version is one of {in_socat_version}"])
        display_elements.append(p)
    if in_organization:
        p = html.P(children=[f"Organization is {in_organization}"])
        display_elements.append(p)
    if in_investigator:
        p = html.P(children=[f"Investigators match one of {in_investigator}"])
        display_elements.append(p)
    if (in_start_date):
        if in_start_date != start_date:
            p = html.P(children=[f"Start date: {in_start_date}"])
            display_elements.append(p)
    if (in_end_date):
        if in_end_date != end_date:
            p = html.P(children=[f"End date: {in_end_date}"])
            display_elements.append(p)
    if in_woce_water:
        p = html.P(children=[f"WOCE Water flag is one of {in_woce_water}"])
        display_elements.append(p)
    if in_region:
        p = html.P(children=[f"Region is one of {in_region}"])
        display_elements.append(p)
    if in_expocode:
        p = html.P(children=[f"Expocodes: {in_expocode}"])
        display_elements.append(p)

    if len(display_elements) == 0:
        style = {'background-color': "#7A76FF"}
        
    else:
        style = {'background-color': "#FFD107"}
        
    return [html.Div(children=display_elements), style]


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
    fname = inspect.currentframe().f_code.co_name
    if relayout_data and 'geo.projection.scale' in relayout_data:
        #you can set these however you like
        min_zoom_scale = 0.5
        max_zoom_scale = 10.0 

        current_zoom_scale = relayout_data['geo.projection.scale']

        logger.debug( f'__{fname}__ limiting if {current_zoom_scale} is outside min {min_zoom_scale} and max {max_zoom_scale}')

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


@app.callback(
    Output('crossover-endpoints', 'data', allow_duplicate=True),
    Input('crossover-timeseries', 'relayoutData'),
    prevent_initial_call=True
)
def filter_crossover_map(timeseries_extents):
    fname = inspect.currentframe().f_code.co_name
    if not timeseries_extents:
        return no_update
    tmin = None
    tmax = None
    if "xaxis.range[0]" in timeseries_extents:
        tmin = timeseries_extents["xaxis.range[0]"]
    if "xaxis.range[1]" in timeseries_extents:
        tmax = timeseries_extents["xaxis.range[1]"]

    if tmin is not None or tmax is not None:
        endpoints = {'tmin': tmin, 'tmax': tmax}
        endpoints_s = json.dumps(endpoints)
        return endpoints_s
   
    return no_update



clientside_callback(
    "window.dash_clientside.clientside.sync_plots",
    Output("no-action", "data"),
    Input("crossover-timeseries", "hoverData"),
    State("crossover-trace-graph", "id"),
)


if __name__ == '__main__':
    app.run(debug=True, dev_tools_props_check=False)
