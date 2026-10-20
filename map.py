from dash import Dash, dcc, html, Input, Output, State, exceptions, callback_context, ALL, no_update
import dash_design_kit as ddk
import plotly.express as px
import pandas as pd
import theme
import datashader as ds

url = 'https://data.pmel.noaa.gov/socat/erddap/tabledap/socat_v2025_decimated.csv?latitude,longitude,time,expocode,fCO2_recommended&time>=2024-01-01&time<=2025-01-31'
df = pd.read_csv(url, skiprows=[1])

turl = 'https://data.pmel.noaa.gov/socat/erddap/tabledap/socat_v2025_decimated.csv?latitude,longitude,time,expocode,fCO2_recommended&expocode="069920240501"'
tdf = pd.read_csv(turl, skiprows=[1])

app = Dash(__name__)
server = app.server 
app.layout = ddk.App(show_editor=True, children=[
    ddk.Header(children=[
        ddk.Logo(src='https://www.socat.info/wp-content/uploads/2017/06/cropped-socat_cat.png'),
        ddk.Title('Surface Ocean CO\u2082 QC Editor'),
    ]),
    dcc.Tabs(children=[
        dcc.Tab(id='1', value='1', label='A Tab with a Map', children=[
            ddk.Card(width=.25, children=[
                html.Button(id='plot', children="Plot Map"),
                html.Button(id='timeseries', children="Plot Timeseries")
            ]),
            ddk.Card(width=.75, style={'height': '75vh'}, children=[
                dcc.Loading(color='white', type='dot', children=[
                    ddk.CardHeader(id='map-graph-header', title='Summary Plot'),
                ]),
                # dcc.Loading(style={'height':'95%', 'width':'95%'}, children=[
                    ddk.Graph(id='map-graph', style={'height':'95%', 'width':'95%'},)
                # ]),
            ]),
        ]),
        dcc.Tab(id="2", value="2", label="Another Tab", children=[
            html.H1("this is the other tab")
        ])
    ])

])

@app.callback(
    [
        Output('map-graph', 'figure', allow_duplicate=True),
        Output('map-graph-header', 'title', allow_duplicate=True)
    ],
    [
        Input('plot', 'n_clicks')
    ], prevent_initial_call=True
)
def make_map(click):
    agg_x = 72
    agg_y = 36
    map_in_variable = 'fCO2_recommended'
    cvs = ds.Canvas(plot_width=agg_x, plot_height=agg_y, x_range=[-180, 180], y_range=[-90, 190],)
    agg = cvs.points(df, 'longitude', 'latitude', ds.mean(map_in_variable))
    sdf = agg.to_pandas()
    pdf = sdf.unstack()
    qdf = pdf.to_frame().reset_index()
    qdf.columns=['longitude','latitude',map_in_variable]
    qdf = qdf.loc[qdf[map_in_variable].notna()]
    rmin = 160
    rmax = 560

    figure = px.scatter_geo(qdf, lat='latitude', lon='longitude', color=map_in_variable, range_color=[rmin, rmax], color_continuous_scale='Viridis')
    figure.update_traces(marker={'size':16})
    figure.update_layout(margin={'t':25, 'b':0, 'l':0, 'r':0}, title='Mean of FCO\u2082 Recommended (per 5\u00B0x5\u00B0 box)', uirevision='x9999', dragmode='select')
    figure.update_yaxes(fixedrange=True)
    figure.update_coloraxes(colorbar={'lenmode':'fraction', 'len':.85, 'y':.5, 'title':'FCO\u2082 Recommended (uatm)', 'title_side':'right'})
    figure.update_geos(showland=True, coastlinecolor='black', coastlinewidth=1, landcolor='tan', resolution=50, fitbounds='locations')
    return[figure, 'Geo Map']


@app.callback(
    [
        Output('map-graph', 'figure', allow_duplicate=True),
        Output('map-graph-header', 'title', allow_duplicate=True)
    ],
    [
        Input('timeseries', 'n_clicks')
    ], prevent_initial_call=True
)
def make_timeseries(click):
   

    map_in_variable = 'fCO2_recommended'

    figure = px.scatter(tdf, x='time', y=map_in_variable, title='FCO\u2082 Recommended for cruise Expocode=069920240501')
    figure.update_yaxes(title='FCO\u2082 Recommended (uatm)')
    figure.update_xaxes(title='Observation time')
    return[figure, 'Time series plot']

if __name__ == '__main__':
    app.run(debug=True, dev_tools_props_check=False)