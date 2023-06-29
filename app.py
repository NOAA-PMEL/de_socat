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
import util
import dash_ag_grid as dag
from sdig.erddap.info import Info
import db

from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

# When there will be more than 50,000 (???) points on the property property panel
# either use the decimated data set or
# segement by time to display to show the first 50000 with a time selector menu to see the remaning segments
#

edits_table = 'socat_edits'

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

redis_instance = redis.StrictRedis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))

visible = {'visibility': 'visible'}
hidden = {'visibility': 'hidden'}

no_display = {'display': 'none'}
display_block = {'display': ''}

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

app.layout = dmc.Container(fluid=True, children=[
    dmc.Header(height=90, children=[
        dmc.Group(children=[
            dmc.Image(src='https://www.socat.info/wp-content/uploads/2017/06/cropped-socat_cat.png', height='70px', width='140px'),
            dmc.Text('Surface Ocean CO\u2082 QC Editor', size="xl", weight=700),
            dmc.Button(id='table-of-cruises-button', children=["Table of Cruises"]),
            dmc.Button(id='edited-rows-button', children=['DEBUG: See rows that have been edited']),
        ],style={'margin-top': '8px'} )
    ], withBorder=True),
    html.Div(id='kick'),
    dcc.Store(id='plot-data-change'),
    dcc.Store(id='map-info'),
    dmc.Grid(children=[
        dmc.Col(span=3, children=[
            dmc.Group(position='apart', children=[
                dmc.Text('Map Controls', size='lg', weight=500, ml='lg'),
                dmc.Button(id='reset', children='Reset', style={'float': 'right'}, mr='lg'),
            ]),
            dmc.AccordionMultiple(style={'height': '480px', 'overflow': 'auto', 'overflow-x': 'hidden'},  chevronPosition="left", variant='contained', 
                                  value=['variable-accordion', 'expocode-accordion', 'woce-accordion', 'region-accordion'], children=[
                dmc.AccordionItem(value='variable-accordion', children=[
                    dmc.AccordionControl('Variable:'),
                    dmc.AccordionPanel(children=[
                        dmc.Select(id='map-variable', searchable=True)
                    ]),
                ]),
                dmc.AccordionItem(value='region-accordion', children=[
                    dmc.AccordionControl('Region:'),
                    dmc.AccordionPanel(children=[
                        dmc.MultiSelect(id='region', placeholder='Select Region', searchable=True, data=[
                            {'value': "A", "label":'North Atlantic'},
                            {'value': "C", "label": "Coastal"},
                            {'value': "I", "label":'Indian'},
                            {'value': "N", "label":"North Pacific"},
                            {'value': "O", "label": "Southern Oceans"},
                            {'value': "R", "label": "Arctic"},
                            {'value': "T", "label": "Tropical Pacific"},
                            {'value': "Z", "label": "Tropical Atlantic"}
                        ],),
                    ]),
                ]),
                dmc.AccordionItem(value='woce-accordion', children=[
                    dmc.AccordionControl('WOCE Flag for Water CO2:'),
                    dmc.AccordionPanel(children=[
                        dmc.MultiSelect(id='woce-co2-water', placeholder='Select WOCE Flag', searchable=True, data=[
                            {'value': "2", "label":'2'},
                            {'value': "3", "label": "3"},
                            {'value': "4", "label":'4'},
                        ]),
                    ]),
                ]),
                dmc.AccordionItem(value='time-accordian', children=[
                    dmc.AccordionControl("Time Range:"),
                    dmc.AccordionPanel(children=[
                        dmc.Group([
                            dmc.DatePicker(
                                id="start-date-picker",
                                label="Start Date",
                                minDate=date(1957, 1, 1),
                                maxDate=datetime.now().date(),
                                value=date(2019, 1, 1),
                                inputFormat='YYYY-MM-DD',
                                clearable=False,
                                # style={"width": 200},
                            ),
                            dmc.DatePicker(
                                id="end-date-picker",
                                label="End Date",
                                minDate=date(1957, 1, 1),
                                maxDate=date(2022, 12, 31),
                                value=date(2019, 12, 31),
                                inputFormat='YYYY-MM-DD',
                                clearable=False,
                                # style={"width": 200, 'padding-left': '30px'},
                            )
                        ])
                    ]),
                ]),
                dmc.AccordionItem(value='metadata-accordian',
                children=[
                    dmc.AccordionControl("Other Metadata Constraints:",),
                    dmc.AccordionPanel(children=[
                        dmc.AccordionMultiple(chevronPosition="left", variant='contained', 
                        styles={"control": {"backgroundColor": dmc.theme.DEFAULT_COLORS["blue"][0], ':hover':{'background-color': dmc.theme.DEFAULT_COLORS["blue"][1]}}},
                        children=[
                            dmc.AccordionItem(value='investigator-item', children=[
                                dmc.AccordionControl("Investigators:"),
                                dmc.AccordionPanel(children=[
                                    dmc.MultiSelect(id='investigator', placeholder='Select investigators', searchable=True, data=[
                                    ]),                                
                                ]),
                            ]),
                            dmc.AccordionItem(value='organization-item', children=[
                                dmc.AccordionControl("Organization:"),
                                dmc.AccordionPanel(children=[
                                    dmc.MultiSelect(id='organization', placeholder='Select organization', searchable=True, data=[
                                    ]),                                
                                ])
                            ]),
                            dmc.AccordionItem(value='qc-flag-item', children=[
                                dmc.AccordionControl("QC Flag:"),
                                dmc.AccordionPanel(children=[
                                    dmc.MultiSelect(id='qc-flag', placeholder='Select QC Flag', searchable=True, data=[
                                        {'label': 'A', 'value': 'A'},
                                        {'label': 'B', 'value': 'B'},
                                        {'label': 'C', 'value': 'C'},
                                        {'label': 'D', 'value': 'D'},
                                        {'label': 'E', 'value': 'E'}
                                    ]),                                
                                ])
                            ]),
                            dmc.AccordionItem(value='platform-type-item', children=[
                                dmc.AccordionControl("Platform Type:"),
                                dmc.AccordionPanel(children=[
                                    dmc.MultiSelect(id='platform-type', placeholder='Select Platform Type', searchable=True, data=[
                                        {'label': "Autonomous Surface Vehicle", 'value': "Autonomous Surface Vehicle"},
                                        {'label': "Boat", 'value': "Boat"},
                                        {'label': "Drifting Buoy", 'value': "Drifting Buoy"},
                                        {'label': "Mooring", 'value': "Mooring"},
                                        {'label': "Ship", 'value': "Ship"}
                                    ]),                                
                                ])
                            ])
                        ],
                        )
                    ]),
                ],                         
                )
            ])
        ]),
        dmc.Col(span=5, children=[
            dmc.Card(children=[
                dmc.CardSection(mt='xl', mx='sm', children=[
                    dcc.Loading(color='white', type='dot', children=[
                        dmc.Text(id='map-graph-header', size='lg', weight=500, ml='lg', mt='lg'),
                    ])
                ]),
                dmc.CardSection(
                    dcc.Loading(
                        dcc.Graph(id='map-graph', config={'modeBarButtonsToAdd':['zoom2d',
                                                    'drawopenpath',
                                                    'drawclosedpath',
                                                    'drawcircle',
                                                    'drawrect',
                                                    'eraseshape'
                                                ]}),
                    )
                )
            ], withBorder=True, shadow="sm", radius="md",),
        ]),
        dmc.Col(span=4, children=[
            dmc.Card(id='trace-card',children=[
                dmc.CardSection(mt='lg', mx='sm', children=[
                    dcc.Loading(type='dot', color='white', children=[
                        dmc.Text(id='trace-graph-header', size='lg', weight=500, ml='lg', mt='lg'),
                    ]),
                ]),
                dmc.CardSection(children=[
                    dcc.Loading(
                        dcc.Graph(id='trace-graph', config={'modeBarButtonsToAdd':['zoom2d',
                                                    'drawopenpath',
                                                    'drawclosedpath',
                                                    'drawcircle',
                                                    'drawrect',
                                                    'eraseshape'
                                                ]}),
                    )
                ])
            ], withBorder=True, shadow="sm", radius="md",),   
        ])
    ]),
    dmc.Grid(id='plot-grid', children=[
        dmc.Col(id='plot-controls', span=3, children=[
            dmc.Group(position='apart', children=[
                dmc.Text('Plot Controls', size='lg', weight=500, ml='lg'),
            ]),
            dmc.AccordionMultiple(style={'height': '350px', 'overflow': 'auto', 'overflow-x': 'hidden'},  chevronPosition="left", variant='contained', 
                                  value=['plot-type-accordion', 'expocode-accordion'],
                                children=[
                dmc.AccordionItem(value='expocode-accordion', children=[
                    dmc.AccordionControl('Expocode:'),
                    dmc.AccordionPanel(children=[
                        dmc.MultiSelect(id='expocode', placeholder='Select Cruises by Expocode', searchable=True)
                    ]),
                ]),
                dmc.AccordionItem(value='plot-type-accordion', children=[
                    dmc.AccordionControl('Plot Type:'),
                    dmc.AccordionPanel([
                        dmc.Select(id='plot-type', value='timeseries',
                            data=[
                                {'label': 'Timeseries', 'value': 'timeseries'},
                                {'label': 'Property-Property', 'value': 'prop-prop'},
                                {'label': 'Property-Property Thumbnails', 'value': 'prop-prop-thumbs', 'disabled': False}
                            ]
                        )
                    ])
                ]),
                dmc.AccordionItem(id='prop-prop-x-item', style={'visibility':'hidden'}, value='prop-prop-x-accordion', children=[
                    dmc.AccordionControl('Property-property X-axis'),
                    dmc.AccordionPanel(children=[
                        dmc.Select(id='prop-prop-x', value='time', searchable=True)
                    ]),
                ]),
                dmc.AccordionItem(id='prop-prop-y-item', style={'visibility':'hidden'}, value='prop-prop-y-accordion', children=[
                    dmc.AccordionControl('Property-property Y-axis'),
                    dmc.AccordionPanel(children=[
                        dmc.Select(id='prop-prop-y', value='fCO2_recommended', searchable=True)
                    ]),
                ]),
                dmc.AccordionItem(id='prop-prop-colorby-item', style={'visibility':'hidden'}, value='prop-prop-colorby-accordion', children=[
                    dmc.AccordionControl('Property-property Color-by'),
                    dmc.AccordionPanel(children=[
                        dmc.Select(id='prop-prop-colorby', value='expocode', searchable=True)
                    ]),
                ])
            ])
        ]),
        dmc.Col(id='one-graph-card', style={'visibility': 'hidden'}, span=9, children=[
            dmc.Card(children=[
                dmc.CardSection(mt='lg', mx='sm', children=[
                    dcc.Loading(type='dot', color='white', children=[
                        dmc.Text(id='one-graph-header',  size='lg', weight=500, ml='lg', mt='lg')
                    ])
                ]),
                dmc.CardSection(
                    dcc.Loading(dcc.Graph(id='one-graph'))
                )
            ], withBorder=True, shadow="sm", radius="md")
        ])
    ]),
    dmc.Modal(id="modal-cruise-table", title="Table of Cruises", overflow="inside", size="95%", zIndex=10000, children=[
        dmc.Grid(id='table-grid', children=[
            dmc.Col(span=12, children=[
                dcc.Loading(dag.AgGrid(id='table-of-cruises', dashGridOptions={'pagination':True, "paginationAutoPageSize": True}, style={'height': '80vh'})),
            ])
        ])
    ]),
    dmc.Modal(id="modal-edit-table", title="Selected Points", overflow="hidden", size="95%", zIndex=10000, style={'height': '95%'}, children=[
        dmc.Grid(id='edit-grid', children=[
            dmc.Col(span=12, children=[
                dmc.Button("Save", id='edit-save'),
                dmc.Button("Cancel", id='edit-cancel', color='yellow'),
                dcc.Loading(dag.AgGrid(id='selected-points', dashGridOptions={'pagination':True, "paginationAutoPageSize": True}, style={'height': '80vh'})),
            ])
        ]),
    ]),
    dmc.Modal(id="modal-review-table", title="Edited Rows Points", overflow="hidden", size="95%", zIndex=10000, children=[
        dmc.Grid(id='edited-display', children=[
            dmc.Col(span=12, children=[
                dmc.Button("DELETE All Rows", id='edit-delete'),
                dcc.Loading(dag.AgGrid(id='edited-points', dashGridOptions={'pagination':True}, style={'height': '80vh'})),
            ])
        ])
    ]),
])


@app.callback(
    Output("modal-review-table", "opened"),
    Output('edited-points', 'rowData'),
    Output('edited-points', 'columnDefs'),
    Input('edited-rows-button', 'n_clicks'),
    Input('edit-delete', 'n_clicks'),
    State("modal-edit-table", "opened"),
    prevent_initial_call=True,
)
def modal_open_debug(show_button, delete_button, opened):
    triggered_id = callback_context.triggered_id
    if triggered_id == 'edited-rows-button':
        edited_rows = db.show_saves()
        columnDefs=[{"field": i, "headerName": i} for i in sorted(edited_rows.columns, key=str.casefold)]      
    elif triggered_id == 'edit-delete':
        db.delete_all_rows()
        return [not opened, None, None]
    return [not opened, edited_rows.to_dict("records"), columnDefs]



@app.callback(
    Output("modal-edit-table", "opened"),
    Input('one-graph', 'selectedData'),
    Input('edit-save', 'n_clicks'),
    Input('edit-cancel', 'n_clicks'),
    State('selected-points', 'rowData'),
    State("modal-edit-table", "opened"),
    prevent_initial_call=True,
)
def modal_open_edit(in_selected_data, save_button, cancel_button, rowData, opened):
    if in_selected_data is None: 
        raise exceptions.PreventUpdate
    if len(in_selected_data['points']) == 0:
        raise exceptions.PreventUpdate
    triggered_id = callback_context.triggered_id
    if triggered_id == 'edit-save':
        selected_data = pd.read_json(redis_instance.hget("cache","edit-table-data"))
        as_edited = pd.DataFrame(rowData)
        edits = pd.concat([selected_data, as_edited]).drop_duplicates(keep=False)
        start = int(edits.shape[0]/2)
        d = datetime.utcnow()
        d = str(d)
        d = d.replace(d[-7:], 'Z')
        edits.loc[:, 'edit_timestamp'] = d
        save_edits = edits.iloc[start:]
        save_edits.to_sql(edits_table, postgres_engine, if_exists='append', index=False)
    return not opened


@app.callback(
    Output("modal-cruise-table", "opened"),
    Input('table-of-cruises-button', "n_clicks"),
    State("modal-cruise-table", "opened"),
    prevent_initial_call=True,
)
def modal_open_cruise(header_button, opened):
    return not opened


@app.callback(
    [
        Output('map-variable', 'data'),
        Output('map-variable', 'value'),
        Output('prop-prop-x', 'data'),
        Output('prop-prop-y','data'),
        Output('prop-prop-colorby', 'data'),
        Output('start-date-picker', 'minDate'),
        Output('start-date-picker', 'maxDate'),
        Output('start-date-picker', 'value'),
        Output('end-date-picker', 'minDate'),
        Output('end-date-picker', 'maxDate'),
        Output('end-date-picker', 'value'),
        Output('investigator', 'data'),
        Output('organization', 'data'),
    ],
    [
        Input('kick', 'n_clicks')
    ]
)
def set_up(click_in):
    dinfo = Info(decimated_url)
    variables, long_names, standard_name, units, v_d_types = dinfo.get_variables()
    variable_options = []
    for var in variables:   
        if var != 'lat_meters' and var != 'lon_meters':
            variable_options.append({'label':var, 'value': var})
    start_date, end_date, start_seconds, end_seconds = dinfo.get_times()
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
    return [variable_options, 'fCO2_recommended', variable_options, variable_options, variable_options, start_date, end_date, start_date, start_date, end_date, end_date, investigator_options, org_options]


@app.callback(
    [
        Output('trace-graph', 'figure'),
        Output('trace-graph-header', 'children'),
        Output('plot-data-change', 'data')
    ],
    [
        Input('expocode','value')
    ],
    [
        State('map-variable', 'value')
    ]
)
def update_trace(trace_in_expocode, trace_in_variable):
    expo_con = {'con':''}
    expo_options = []
    vars_to_get = ['latitude', 'longitude', 'time', 'expocode', trace_in_variable,] 
    if trace_in_expocode is not None and len(trace_in_expocode) > 0:
        expo_con = Info.make_platform_constraint('expocode', trace_in_expocode)
        print(expo_con['con'])
    if len(expo_con['con']) > 0:
        vars_to_get.extend(thumbnail_vars)
        vars_to_get = list(set(vars_to_get))
        url = full_url + '.csv?' + ','.join(vars_to_get) +'&'+expo_con['con']
        # if there is an expo set, use the list previously set
        expo_store = redis_instance.hget("cache", "expocodes")
        expo_options = json.loads(expo_store)
    else:
        raise exceptions.PreventUpdate
    print('plot url = ' + url)
    df = pd.read_csv(url, skiprows=[1])
    df = df.loc[df[trace_in_variable].notna()]
    rmin = df[trace_in_variable].min()
    rmax = df[trace_in_variable].max()
    lat_min, lat_max, lon_min, lon_max, fitbounds = get_map_ranges(df)
    print(lat_min, lat_max, lon_min, lon_max, fitbounds)
    figure = px.scatter_geo(df,
                            lat='latitude',
                            lon='longitude',
                            color=trace_in_variable,
                            color_continuous_scale='Viridis',
                            hover_data=['expocode','time','latitude','longitude',trace_in_variable],
                            range_color=[rmin,rmax], custom_data=['expocode'],)
    figure.update_traces(marker=dict(size=6))
    if fitbounds:
        figure.update_geos(fitbounds='locations')
    else:
        figure.update_geos(lonaxis_range=[lon_min,lon_max], lataxis_range=[lat_min,lat_max])
    figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan', resolution=50)
    figure.update_coloraxes(colorbar={'orientation':'h', 'thickness':20, 'y': -.175, 'title': None})
    title = 'All ' + trace_in_variable + ' data from crusies ' + str(trace_in_expocode)
    redis_instance.hset("cache", 'plot-data', df.to_json())
    return [figure, title, 'yes']

@app.callback(
    [
        Output('map-graph', 'figure'),
        Output('map-graph-header', 'children'),
        Output('expocode', 'data'),     
    ],
    [
        Input('map-variable', 'value'),
        Input('region', 'value'),
        Input('woce-co2-water', 'value'),
        Input('start-date-picker', 'value'),
        Input('end-date-picker', 'value'),
        Input('investigator', 'value'),            # These could be done with pattern matching, but...
        Input('organization', 'value'),
        Input('qc-flag', 'value'),
        Input('platform-type', 'value'),
        Input('map-info', 'data')
    ],
    [
        State('expocode', 'value')
    ]
)
def update_map(map_in_variable, in_regions, in_woce_water, in_start_date, in_end_date, in_investigator, in_org, in_qc_flag, in_platform_type, map_info, map_in_expocode):

    vars_to_get = ['latitude','longitude','time','expocode',map_in_variable] 
    time_con = '&time>='+in_start_date+'&time<='+in_end_date
    investigator_con = util.make_con('investigators', in_investigator)
    if investigator_con:
        vars_to_get.append('investigators')
    org_con = util.make_con('organization', in_org)
    if org_con:
        vars_to_get.append('organization')
    qc_flag_con = util.make_con('qc_flag', in_qc_flag)
    if qc_flag_con:
        vars_to_get.append('qc_flag')
    platform_type_con = util.make_con('platform_type', in_platform_type)
    if platform_type_con:
        vars_to_get.append('platform_type')
    woce_water_con = util.make_con('WOCE_CO2_water', in_woce_water)
    if woce_water_con:
        vars_to_get.append('WOCE_CO2_water')
    region_con = util.make_con('region_id', in_regions)
    if region_con:
        vars_to_get.append('region_id')
    url = decimated_url + '.csv?' + ','.join(vars_to_get) + time_con + region_con + woce_water_con + investigator_con + org_con + qc_flag_con + platform_type_con
    if map_info is not None and len(map_info) > 3:
        bounds = json.loads(map_info)
        cons = maputil.get_socat_subset(bounds['ll']['longitude'], bounds['ur']['longitude'],bounds['ll']['latitude'],bounds['ur']['latitude'])
        url = url + cons['lat'] + cons['lon']
    expo_options = []
    print('map URL: ' + url)
    df = pd.read_csv(url, skiprows=[1])
    if map_in_expocode is not None and len(map_in_expocode) > 0:
        expo_store = redis_instance.hget("cache", "expocodes")
        expo_options = json.loads(expo_store)

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
        lat_min, lat_max, lon_min, lon_max, fitbounds = get_map_ranges(df)
        figure = px.scatter_geo(df,
                                lat='latitude',
                                lon='longitude',
                                color=map_in_variable,
                                color_continuous_scale='Viridis',
                                hover_data=['expocode','time','latitude','longitude',map_in_variable],
                                range_color=[rmin,rmax], 
                                custom_data=['expocode'], 
                                projection='equirectangular')
        if fitbounds:
            figure.update_geos(fitbounds='locations')
        else:
            figure.update_geos(lonaxis_range=[lon_min,lon_max], lataxis_range=[lat_min,lat_max])
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
        figure.update_traces(marker={'size':3})
        figure.update_layout(margin={'t':25, 'b':25, 'l':0, 'r':0})
    figure.update_layout(margin={'t':25, 'b':25, 'l':0, 'r':0})
    figure.update_coloraxes(colorbar={'orientation':'h', 'thickness':20, 'y': -.175, 'title': None})
    figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan', resolution=50)

    redis_instance.hset("cache", "expocodes",json.dumps(expo_options))
    return [figure, title, expo_options]


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
        Input('one-graph', 'selectedData')
    ]
)
def show_selected_points(in_points):
    if in_points is not None:
        all_data = pd.read_json(redis_instance.hget("cache","plot-data"))
        column_names = sorted(all_data.columns, key=str.casefold)
        column_names.remove('WOCE_CO2_water')
        column_names.insert(0, 'WOCE_CO2_water')
        columnDefs=[{"field": i, "headerName": i, 'editable': True, 'sortable': True, 'cellEditor': 'agSelectCellEditor', 'cellEditorParams':
                    {'values': ['2', '3', '4']}
                    } if 'WOCE' in i else {"field": i, "headerName": i} for i in column_names]
        selected_points = in_points['points']
        times = []
        for point in selected_points:
            customs = point['customdata']
            times.append(customs[0])
        to_show = all_data.loc[all_data['time'].isin(times)]
        redis_instance.hset("cache", 'edit-table-data', to_show.to_json())
        return [to_show.to_dict("records"), columnDefs]
    else:
        raise exceptions.PreventUpdate



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
    print('=-=-=-=-=- starting set_platform_code_from_map =-=-=-=-=-=')
    # DEBUG 
    print('printing click')
    # DEBUG 
    print(str(in_click))
    if in_click is not None:
        # DEBUG 
        print('getting first point')
        fst_point = in_click['points'][0]
        # DEBUG 
        print(fst_point)
        if 'customdata' in fst_point:
            out_expocode = fst_point['customdata']
            out_value = out_expocode[0]
            print('expo to add because of click ' + out_value)
            print('existing expo ' + str(state_in_expovalue))
        else:
            print('no custom data in click')
            raise exceptions.PreventUpdate
    if state_in_expovalue is not None and len(state_in_expovalue) > 0:
        if isinstance(state_in_expovalue, str):
            out_value = [state_in_expovalue, out_value]
        elif isinstance(state_in_expovalue, list):
            state_in_expovalue.append(out_value)
            out_value = [state_in_expovalue]
    if isinstance(out_value, list):
        new_lst = [item for item in out_value if item is not None]
        print('map selection is returning expo value of ' + str(new_lst))
        print('=-=-=-=-=- Finished (new_lst) set_platform_code_from_map =-=-=-=-=-=')
        return new_lst
    else:
        print('map selection is returning expo value of ' + str(out_value))
        print('=-=-=-=-=- Finished (out_value) set_platform_code_from_map =-=-=-=-=-=')
        return [[out_value]]


@app.callback(
    [
        Output('trace-card', 'style')
    ],
    [
        Input('expocode', 'value'),
    ]
)    
def set_visibility_trace(in_expocode):
    if in_expocode is None or len(in_expocode) < 1:
        return [hidden]
    else:
        return [visible]    


@app.callback(
    [
        Output('one-graph-card', 'style'),
    ],
    [
        Input('plot-data-change', 'data'),
        Input('expocode', 'value')
    ]
)    
def set_visibility_plot(plot_data, in_expocode):
    if (in_expocode is None or len(in_expocode) < 1) or (plot_data is None or plot_data == 'no'):
        return [hidden]
    else:
        return [visible]

    

@app.callback(
    [
        Output('one-graph', 'figure'),
        Output('one-graph-header', 'children')
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
    print('updating the timeseries plot ' + str(plot_in_expocode))
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
        ts_plot = ts_sub.sort_values(['time','expocode'])
        card_title = 'Timeseries of ' + in_map_variable + ' from ' + ', '.join(plot_in_expocode)
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
        card_title = in_prop_prop_y + ' vs ' + in_prop_prop_x + ' colored by ' + in_prop_prop_colorby + 'from' + ', '.join(plot_in_expocode)
        figure = px.scatter(to_plot,
                            x=in_prop_prop_x,
                            y=in_prop_prop_y,
                            color=in_prop_prop_colorby,
                            hover_name='expocode',
                            hover_data=['time',in_prop_prop_x,in_prop_prop_y,in_prop_prop_colorby],
                            custom_data=['time'],
                            color_discrete_sequence=cmap,
                            color_continuous_scale=px.colors.sequential.Viridis
        )
        figure.update_layout(height=450)
    elif in_plot_type == 'prop-prop-thumbs':
        card_title = 'Property Property Thumbnails for ' + ', '.join(plot_in_expocode)
        plots = []
        subplot_titles = []
        num_plots = len(thumbnail_pairs)
        num_rows = int(num_plots/3)
        if num_rows == 0:
            num_rows = num_rows + 1
        if num_plots > 3 and num_plots%3 > 0:
            num_rows = num_rows + 1
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
            subplot = px.scatter(to_plot,
                        x=x,
                        y=y,
                        color=color_by,
                        hover_name='expocode',
                        # labels = {'color': color_by},
                        hover_data=['time','latitude','longitude','expocode', x, y, color_by],
                        custom_data=['time'],
                        color_discrete_sequence=cmap,
                        # color_continuous_scale=px.colors.sequential.Viridis,
                    )
            subplot.update_traces(legend=leg)
            plots.append(subplot)
            legend = legend + 1
        figure = make_subplots(cols=3, rows=num_rows, shared_xaxes=False, shared_yaxes=False)
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

        figure.update_layout(height=num_rows*450, margin=dict( l=80, r=80, b=80, t=80, ))
    # DEBUG print('returning figure and title')
    return[figure, card_title]


@app.callback(
    [
        Output('table-of-cruises', 'rowData'),
        Output('table-of-cruises', 'columnDefs')
    ],
    [
        Input('table-of-cruises-button', 'n_clicks')
    ],
    [
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
def make_table_of_crusies(da_click, mt_in_start_date, mt_in_end_date, mt_in_woce_water, mt_in_regions, mt_in_investigator, mt_in_org, mt_in_qc_flag, mt_in_platform_type, mt_in_map_info):
    vars_to_get = ['expocode', 'platform_name',	'platform_type', 'investigators', 'qc_flag', 'socat_version'] 
    time_con = '&time>='+mt_in_start_date+'&time<='+mt_in_end_date
    investigator_con = util.make_con('investigators', mt_in_investigator)
    org_con = util.make_con('organization', mt_in_org)
    if org_con:
        vars_to_get.append('organization')
    qc_flag_con = util.make_con('qc_flag', mt_in_qc_flag)
    platform_type_con = util.make_con('platform_type', mt_in_platform_type)
    woce_water_con = util.make_con('WOCE_CO2_water', mt_in_woce_water)
    if woce_water_con:
        woce_water_con = '&' + woce_dict['con']
    region_con = util.make_con('region_id', mt_in_regions)
    if region_con:
        vars_to_get.append('region_id')
    url = decimated_url + '.csv?' + ','.join(vars_to_get) + time_con + region_con + woce_water_con + investigator_con + org_con + qc_flag_con + platform_type_con + region_con
    if mt_in_map_info is not None and len(mt_in_map_info) > 3:
        bounds = json.loads(mt_in_map_info)
        cons = maputil.get_socat_subset(bounds['ll']['longitude'], bounds['ur']['longitude'],bounds['ll']['latitude'],bounds['ur']['latitude'])
        url = url + cons['lat'] + cons['lon']
    url = url + '&distinct()'
    expo_options = []
    print('table URL: ' + url)
    df = pd.read_csv(url, skiprows=[1])
    columnDefs = [{"field": i, "headerName": i} for i in sorted(df.columns, key=str.casefold)]      
    return [df.to_dict("records"), columnDefs]


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
        Output('map-info', 'data', allow_duplicate=True),
        Output('region', 'value', allow_duplicate=True),
        Output('woce-co2-water', 'value', allow_duplicate=True),
        Output('start-date-picker', 'value', allow_duplicate=True),
        Output('end-date-picker', 'value', allow_duplicate=True),
        Output('investigator', 'value', allow_duplicate=True),
    ],
    [
        Input('reset', 'n_clicks'),
        State('start-date-picker', 'minDate'),
        State('end-date-picker', 'maxDate')
    ], prevent_initial_call=True
)
def reset_map(click, min_date, max_date):
    return ['', [], [], min_date, max_date, '']


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
