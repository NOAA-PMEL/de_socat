from dash import Dash, dcc, html, Input, Output, State, exceptions, callback_context
import plotly.express as px
import plotly.graph_objects as go
import dash_design_kit as ddk
import numpy as np
import pandas as pd
import random
import PIL
import math


zoom = .5

center180 = {'lon': 180.0, 'lat': 0.0}
map_limits360 = {"west": 0, "east": 360, "south": -89, "north": 89}
title180 = 'Map Centered at the International Dateline'

lat_lon0 = [[180.0, -66.51326], [359.0, 66.5132]]
image_coordinates0 = [
    [lat_lon0[0][0], lat_lon0[1][1]],
    [lat_lon0[1][0], lat_lon0[1][1]],
    [lat_lon0[1][0], lat_lon0[0][1]],
    [lat_lon0[0][0], lat_lon0[0][1]],
]
lat_lon1 = [[0, -66.51326], [180, 66.5132]]
image_coordinates1 = [
    [lat_lon1[0][0], lat_lon1[1][1]],
    [lat_lon1[1][0], lat_lon1[1][1]],
    [lat_lon1[1][0], lat_lon1[0][1]],
    [lat_lon1[0][0], lat_lon1[0][1]],
]

with PIL.Image.open('i0.png') as img0:
    img0.load()

with PIL.Image.open('i1.png') as img1:
    img1.load()

initial_map_layers = [
    {"sourcetype": "image", "source": img0, "coordinates": image_coordinates0, 'below': 'traces'},
    {"sourcetype": "image", "source": img1, "coordinates": image_coordinates1, 'below': 'traces'}
]

map_height = 520
map_width = 1100

lon = range(160, 275, 5)
lat = np.random.default_rng().uniform(low=-6.0, high=6.0, size=(len(lon)))
temp = np.random.default_rng().uniform(low=24, high=30, size=(len(lon)))

cats = ['ship', 'glider', 'UCV', 'drifting buoy']
types = []
for i in range(0, len(lon)):
    types.append(cats[random.randint(0, 3)])

df = pd.DataFrame(zip(lon, lat, temp, types), columns=['longitude', 'latitude', 'sst', 'platform'])


# Define Dash application structure
app = Dash(__name__)
server = app.server  # expose server variable for Procfile

app.layout = ddk.App(children=[
    ddk.Header([
        ddk.Logo(src='https://www.socat.info/wp-content/uploads/2017/06/cropped-socat_cat.png'),
        ddk.Title('Marker Test'),
    ]),
    html.Div(id='kick'),
    ddk.Card(width=.7,
        children=[
            dcc.Loading(
                ddk.CardHeader(
                    modal=True, 
                    modal_config={'height': 90, 'width': 95}, 
                    fullscreen=True,
                    id='map-graph-header',
                    title='Invisible Markers',
                ),
            ),
            ddk.Graph(
                id='map-graph',
            ),
        ]
    ),
])


@app.callback(
    [
        Output('map-graph', 'figure'),
    ],
    [
        Input('kick','n_clicks'),
        Input('map-graph', 'relayoutData')
    ]
)
def update_map(in_map_center, map_state):

    my_center = center180
    my_zoom = zoom
    if map_state is not None and 'mapbox.center' in map_state:
        my_center = map_state['mapbox.center']

    if map_state is not None and 'mapbox.zoom' in map_state:
        my_zoom = map_state['mapbox.zoom']

    my_limits = map_limits360

    if math.isclose(zoom, my_zoom):
        map_layers = initial_map_layers
        figure = go.Figure()
        map_b = go.Scattermapbox()
        figure.add_traces(map_b)
    else:
        map_layers = None
        figure = px.scatter_mapbox(
            df, 
            lat='latitude', 
            lon='longitude', 
            color='sst',
            hover_name='platform',
            hover_data=['platform'], 
            color_continuous_scale=px.colors.sequential.Viridis)

    figure.update_layout(
        height=map_height,
        mapbox_style="open-street-map",
        mapbox_zoom=my_zoom,
        mapbox_center=my_center,
        mapbox_layers=map_layers,
        mapbox_bounds=my_limits,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        legend=dict(
            orientation="v",
            xanchor='left',
            x=.99,
        ),
        modebar_orientation='v',
    )
    return [figure]

if __name__ == '__main__':
    app.run_server(debug=True)
