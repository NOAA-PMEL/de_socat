from dash import Dash, dcc, html, Input, Output, State, exceptions, callback_context, ALL
import dash_design_kit as ddk
import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as colors
from plotly.subplots import make_subplots
import os
import io
import colorcet as cc
from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
import pandas as pd
import datashader as ds
import datashader.transfer_functions as tf
import PIL
from pyproj import Transformer
import dash_mantine_components as dmc
from datetime import datetime, date
import numpy as np
import pprint
import hashlib
import json
import redis
import maputil
from sdig.erddap.info import Info

# When there will be more than 50,000 (???) points on the property property panel
# either use the decimated data set or
# segement by time to display to show the first 50000 with a time selector menu to see the remaning segments
#

pp = pprint.PrettyPrinter(indent=4)

redis_instance = redis.StrictRedis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))

visible = {'visibility': 'visible'}
hidden = {'visibility': 'hidden'}

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

thumbnail_vars = []
for sub_list in thumbnail_pairs:
    thumbnail_vars.extend(sub_list)

thumbnail_vars = list(set(thumbnail_vars))

ESRI_API_KEY = os.environ.get('ESRI_API_KEY')

zoom = 1
center = {'lon': 0.0, 'lat': 0.0}
map_limits = {"west": -180, "east": 180, "south": -89, "north": 89}

map_height = 525
map_width = 1050

agg_x = 270
agg_y = 135

map_title_base = 'Trajectory from the SOCAT v2022 Decimated Data Set '
decimated_url = 'https://data.pmel.noaa.gov/socat/erddap/tabledap/socat_v2020_decimated'
full_url = 'https://data.pmel.noaa.gov/socat/erddap/tabledap/socat_v2022_fulldata'

# Define Dash application structure
app = Dash(__name__)
server = app.server  # expose server variable for Procfile

app.layout = ddk.App(children=[
    ddk.Header([
        ddk.Logo(src='https://www.socat.info/wp-content/uploads/2017/06/cropped-socat_cat.png'),
        ddk.Title('Surface Ocean CO\u2082 Atlas'),
    ]),
    html.Div(id='kick'),
    dcc.Store(id='plot-data-change'),
    dcc.Store(id='map-info'),
    ddk.Block(width=.3, children=[
        ddk.ControlCard(width=1., children=[
            ddk.CardHeader(title='Map Controls'),
            html.Button(id='reset', children='Reset', style={'width': '120px'}),
            ddk.ControlItem(label='Expocode', children=[
                dcc.Dropdown(id='expocode', placeholder='Select Cruises by Expocode', multi=True)
            ]),
            ddk.ControlItem(label='Variable:', children=[
                dcc.Dropdown(id='map-variable', multi=False)
            ]),
            ddk.ControlItem(label="Time Range", children=[
                dmc.DatePicker(
                    id="start-date-picker",
                    label="Start Date",
                    minDate=date(1957, 1, 1),
                    maxDate=datetime.now().date(),
                    value=date(1957, 1, 1),
                    inputFormat='YYYY-MM-DD',
                    style={"width": 200},
                ),
                dmc.DatePicker(
                    id="end-date-picker",
                    label="End Date",
                    minDate=date(1957, 1, 1),
                    maxDate=date(2022, 12, 31),
                    value=datetime.now().date(),
                    inputFormat='YYYY-MM-DD',
                    style={"width": 200, 'padding-left': '30px'},
                )
            ]),
            
        ]),
    ]),
    ddk.Card(width=.7, children=[
        dcc.Loading(
            ddk.CardHeader(
                modal=True, 
                modal_config={'height': 90, 'width': 95}, 
                fullscreen=True,
                id='map-graph-header',
            ),
        ),
        dcc.Loading(
            dcc.Graph(id='map-graph', config={'modeBarButtonsToAdd':['zoom2d',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape'
                                       ]}),
        )
    ]),    
    ddk.ControlCard(id='plot-controls', width=.3, style={'visibility': 'hidden'}, children=[
        ddk.CardHeader(title='Plot Controls'),
        ddk.ControlItem(label='Plot type', children=[
            dcc.Dropdown(id='plot-type',
            value='timeseries',
            multi=False,
            options=[
                {'label': 'Timeseries', 'value': 'timeseries'},
                {'label': 'Property-Property', 'value': 'prop-prop'},
                {'label': 'Property-Property Thumbnails', 'value': 'prop-prop-thumbs', 'disabled': False}
            ])
        ]),
        ddk.ControlItem(id='prop-prop-x-item', style={'visibility':'hidden'}, label='Property-property X-axis', children=[
            dcc.Dropdown(id='prop-prop-x',
            value='fCO2_recommended',
            multi=False,
            disabled=False,)
        ]),
        ddk.ControlItem(id='prop-prop-y-item', style={'visibility':'hidden'}, label='Property-property Y-axis', children=[
            dcc.Dropdown(id='prop-prop-y',
            value='latitude',
            multi=False,
            disabled=False,)
        ]),
        ddk.ControlItem(id='prop-prop-colorby-item', style={'visibility':'hidden'}, label='Property-property Color-by', children=[
            dcc.Dropdown(id='prop-prop-colorby',
            value='expocode',
            multi=False,
            disabled=False,)
        ]),
    ]),  
    ddk.Card(width=.7, id='one-graph-card', style={'visibility': 'hidden'}, children=[
        dcc.Loading(
            ddk.CardHeader(
                modal=True, 
                modal_config={'height': 90, 'width': 95}, 
                fullscreen=True,
                id='one-graph-header',
            ),
        ),
        dcc.Loading(dcc.Graph(id='one-graph'))
    ]),
])


@app.callback(
    [
        Output('map-variable', 'options'),
        Output('map-variable', 'value'),
        Output('prop-prop-x', 'options'),
        Output('prop-prop-y','options'),
        Output('prop-prop-colorby', 'options')
    ],
    [
        Input('kick', 'n_clicks')
    ]
)
def set_variable_options(click_in):
    dinfo = Info(decimated_url)
    variables, long_names, standard_name, units = dinfo.get_variables()
    variable_options = []
    for var in variables:   
        if var != 'lat_meters' and var != 'lon_meters':
            variable_options.append({'label':var, 'value': var})
    return [variable_options, 'fCO2_recommended', variable_options, variable_options, variable_options]


@app.callback(
    [
        Output('map-graph', 'figure'),
        Output('map-graph-header', 'title'),
        Output('expocode', 'options'),
        Output('plot-data-change', 'data')
    ],
    [
        Input('expocode', 'value'),
        Input('map-variable', 'value'),
        Input('start-date-picker', 'value'),
        Input('end-date-picker', 'value'),
        Input('map-info', 'data')
    ]
)
def update_map(map_in_expocode, map_in_variable, in_start_date, in_end_date, map_info):
    data_store='no'
    expo_con = {'con':''}
    expo_options = []
    if map_in_expocode is not None and len(map_in_expocode) > 0:
        expo_con = Info.make_platform_constraint('expocode', map_in_expocode)
    vars_to_get = ['latitude','longitude','time','expocode',map_in_variable] 
    time_con = '&time>='+in_start_date+'&time<='+in_end_date
    if len(expo_con['con']) > 0:
        vars_to_get.extend(thumbnail_vars)
        vars_to_get = list(set(vars_to_get))
        url = full_url + '.csv?' + ','.join(vars_to_get) + time_con+'&'+expo_con['con']
        # if there is an expo set, use the list previously set
        expo_store = redis_instance.hget("cache", "expocodes")
        expo_options = json.loads(expo_store)
        data_store = 'yes'
    else:
        url = decimated_url + '.csv?' + ','.join(vars_to_get) + time_con
    if map_info is not None and len(map_info) > 3 and expo_con['con'] == '':
        bounds = json.loads(map_info)
        cons = maputil.get_socat_subset(bounds['ll']['longitude'], bounds['ur']['longitude'],bounds['ll']['latitude'],bounds['ur']['latitude'])
        url = url + cons['lat'] + cons['lon']

    print(url)
    df = pd.read_csv(url, skiprows=[1])

    # if there is no expo set in the menu, reset the options to the current map
    if len(expo_options) == 0:
        expocodes = df['expocode'].unique()
        for code in sorted(expocodes):
            expo_options.append({'value': code, 'label': code})
    # DEBUG print('found ' + str(df.shape[0]) + ' observations')

    if (df.shape[0]<50000):
        title = map_in_variable + ' from ' + in_start_date + ' to ' + in_end_date
        # DEBUG print('making a scatter geo plot')
        df = df.loc[df[map_in_variable].notna()]
        if 'fCO2' in map_in_variable:
            rmin = 160
            rmax = 560
        else:
            rmin = df[map_in_variable].min()
            rmax = df[map_in_variable].max()
        figure = px.scatter_geo(df,
                                lat='latitude',
                                lon='longitude',
                                color=map_in_variable,
                                color_continuous_scale='Viridis',
                                hover_data=['expocode','time','latitude','longitude',map_in_variable],
                                range_color=[rmin,rmax], custom_data=['expocode'], fitbounds='locations')
        figure.update_traces(marker=dict(size=6))
    else:
        title = 'Mean of ' + map_in_variable + ' from ' + in_start_date + ' to ' + in_end_date
        # DEBUG print('making a datashader plot')
        cvs = ds.Canvas(plot_width=agg_x, plot_height=agg_y, x_range=[-180,180], y_range=[-90,90],)
        agg = cvs.points(df, 'longitude', 'latitude', ds.mean(map_in_variable))
        sdf = agg.to_pandas()
        pdf = sdf.unstack()
        qdf = pdf.to_frame().reset_index()
        qdf.columns=['longitude','latitude',map_in_variable]
        qdf = qdf.loc[qdf[map_in_variable].notna()]
        if 'fCO2' in map_in_variable:
            rmin = 160
            rmax = 560
        else:
            rmin = qdf[map_in_variable].min()
            rmax = qdf[map_in_variable].max()
        figure = px.scatter_geo(qdf, lat='latitude', lon='longitude', color=map_in_variable, range_color=[rmin, rmax], color_continuous_scale='Viridis')
        figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan')
        figure.update_traces(marker={'size':3})
        figure.update_layout(margin={'t':25, 'b':25, 'l':0, 'r':0})
    figure.update_layout(margin={'t':25, 'b':25, 'l':0, 'r':0})
    figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan', resolution=50)
    if data_store == 'yes':
        redis_instance.hset("cache", 'plot-data', df.to_json())
    redis_instance.hset("cache", "expocodes",json.dumps(expo_options))
    return [figure, title, expo_options, data_store]


@app.callback(
    [
        Output('map-info','data')
    ],
    [
        Input('map-graph','selectedData')
    ]
)
def selectData(selectData):
    map_info = {}
    if selectData is not None:
        geo_range = selectData['range']['geo']
        ll_longitude = geo_range[0][0]
        ll_latitude = geo_range[1][1]
        ur_longitude = geo_range[1][0]
        ur_latitude = geo_range[0][1]
        map_info = {
            'll': {
                'longitude': ll_longitude,
                'latitude': ll_latitude
            },
            'ur': {
                'longitude': ur_longitude,
                'latitude': ur_latitude
            }
        }
    return [json.dumps(map_info)]

@app.callback(
    [
        Output('expocode', 'value')
    ],
    [
        Input('map-graph', 'clickData')
    ],
    [
        State('expocode', 'value')
    ], prevent_initial_call=True
)
def set_platform_code_from_map(in_click, state_in_expovalue):
    out_expocode = None
    # DEBUG print('printing click')
    # DEBUG print(str(in_click))
    if in_click is not None:
        # DEBUG print('getting first point')
        fst_point = in_click['points'][0]
        # DEBUG print(fst_point)
        if 'customdata' in fst_point:
            out_expocode = fst_point['customdata']
            out_value = out_expocode[0]
        else:
            raise exceptions.PreventUpdate
    if state_in_expovalue is not None and len(state_in_expovalue) > 0:
        if isinstance(state_in_expovalue, str):
            out_value = [state_in_expovalue, out_expocode]
        elif isinstance(state_in_expovalue, list):
            state_in_expovalue.append(out_expocode)
            out_value = state_in_expovalue
    if isinstance(out_value, list):
        new_lst = [item for item in out_value if item is not None]
        return [new_lst]
    else:
        return [out_value]


@app.callback(
    [
        Output('one-graph-card', 'style'),
        Output('plot-controls', 'style')
    ],
    [
        Input('expocode', 'value'),
    ]
)    
def set_visibility(in_expocode):
    if in_expocode is None or len(in_expocode) < 1:
        return [hidden, hidden]
    else:
        return [visible, visible]    
    

@app.callback(
    [
        Output('one-graph', 'figure'),
        Output('one-graph-header', 'title')
    ],
    [
        Input('plot-data-change', 'data'),
        Input('plot-type', 'value'),
        Input('prop-prop-x', 'value'),
        Input('prop-prop-y', 'value'),
        Input('prop-prop-colorby', 'value'),
    ],
    [
        State('expocode', 'value'),
        State('map-variable', 'value'),
 
    ], prevent_initial_call=True
)
def update_plots(plot_data_store, in_plot_type, in_prop_prop_x, in_prop_prop_y, in_prop_prop_colorby, plot_in_expocode, in_map_variable):

    x_label = None
    y_label = None
    legend_title = None
    print(str(plot_in_expocode))
    if plot_in_expocode is None or len(plot_in_expocode) == 0:
        print('data-plot: no expo')
        raise exceptions.PreventUpdate
    if in_map_variable is None or len(in_map_variable) == 0:
        print('data-plot: no variable')
        raise exceptions.PreventUpdate
    if plot_data_store == 'no':
        print('no new data')
        raise exceptions.PreventUpdate
    
    to_plot = pd.read_json(redis_instance.hget("cache","plot-data"))

    

    if to_plot.shape[0] < 1:
        raise exceptions.PreventUpdate

    to_plot['expocode'] = to_plot['expocode'].astype(str)
    to_plot['WOCE_CO2_water'] = to_plot['WOCE_CO2_water'].astype(str)

    if in_prop_prop_colorby == 'expocode':
        cmap = px.colors.qualitative.Light24
    else:
        cmap = px.colors.qualitative.Dark24

    if in_plot_type == 'timeseries':
        # DEBUG print('timeseries plot with ' + str(to_plot.shape[0]) + ' data points.')
        ts_sub = to_plot[['time', 'latitude', 'longitude', in_map_variable, 'expocode']]
        ts_sort = ts_sub.sort_values(['time','expocode'])
        ts_clean = 
        card_title = 'Time seris of ' + in_map_variable + ' from ' + str(plot_in_expocode)
        # DEBUG print('start sort')
        # DEBUG print('end sort -- plotting now')
        figure = px.line(ts_plot,
                    x='time', 
                    y=in_map_variable, 
                    color='expocode', 
                    hover_name='expocode',
                    hover_data=['time','latitude','longitude', in_map_variable],
                    color_discrete_sequence=px.colors.qualitative.Light24,
                )
        figure.update_layout(height=450)
        #DEBUG print('plot done height set')
        # figure.update_traces(connectgaps=False)
    elif in_plot_type == 'prop-prop':
        card_title = in_prop_prop_y + ' vs ' + in_prop_prop_x + ' colored by ' + in_prop_prop_colorby
        figure = px.scatter(to_plot,
                            x=in_prop_prop_x,
                            y=in_prop_prop_y,
                            color=in_prop_prop_colorby,
                            hover_name='expocode',
                            hover_data=['time',in_prop_prop_x,in_prop_prop_y,in_prop_prop_colorby],
                            color_discrete_sequence=cmap,
                            color_continuous_scale=px.colors.sequential.Viridis
        )
        figure.update_layout(height=450)
    elif in_plot_type == 'prop-prop-thumbs':
        card_title = 'Property Property Thumbnails for ' + 'EXPO JOE'
        plots = []
        subplot_titles = []
        num_plots = len(thumbnail_pairs)
        num_rows = int(num_plots/3)
        if num_rows == 0:
            num_rows = num_rows + 1
        if num_plots > 3 and num_plots%3 > 0:
            num_rows = num_rows + 1
        for pair in thumbnail_pairs:
            subplot_title = pair[1] + ' vs ' + pair[0] + ' colored by ' + pair[2]
            x = pair[0]
            y = pair[1]
            color_by = pair[2]
            subplot_titles.append(subplot_title)
            if color_by == 'expocode':
                cmap = px.colors.qualitative.Light24
            else:
                cmap = px.colors.qualitative.Dark24
            subplot = px.scatter(to_plot,
                        x=x,
                        y=y,
                        color=color_by,
                        hover_name='expocode',
                        hover_data=['time','latitude','longitude','expocode', x, y, color_by],
                        color_discrete_sequence=cmap,
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
            plots.append(next(subplot.select_traces()))
        figure = make_subplots(cols=3, rows=num_rows, shared_xaxes=False, shared_yaxes=False, subplot_titles=subplot_titles)
        i = 1
        j = 1
        for d, plot in enumerate(plots):
            figure.add_trace(plot, i, j)
            j = (j + 1)%4
            if j == 0:
                j = j + 1
            if j == 1 and d > 0:
                i = i + 1
            figure.update_xaxes(title_text=thumbnail_pairs[d][0], showticklabels=True, row=i, col=j)
            figure.update_yaxes(title_text=thumbnail_pairs[d][1], showticklabels=True, row=i, col=j)
        figure.update_layout(height=num_rows*450, margin=dict( l=80, r=80, b=80, t=80, ))
    # DEBUG print('returning figure and title')
    return[figure, card_title]


@app.callback(
    [
        Output('prop-prop-x-item', 'style'),
        Output('prop-prop-y-item', 'style'),
        Output('prop-prop-colorby-item', 'style')
    ],
    [
        Input('plot-type', 'value')
    ]
)
def set_prop_prop_display(in_plot_type):
    if in_plot_type is not None and in_plot_type == "prop-prop":
        return [{'display': 'block'}, {'display': 'block'}, {'display':'block'}]
    return [{'display':'none'}, {'display':'none'}, {'display':'none'}]


@app.callback(
    [
        Output('expocode', 'value', allow_duplicate=True),
        Output('map-info', 'data', allow_duplicate=True)
    ],
    [
        Input('reset', 'n_clicks')
    ], prevent_initial_call=True
)
def reset_map(click):
    return ['', '']


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

if __name__ == '__main__':
    app.run_server(debug=True)
