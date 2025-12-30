import theme
import constants
import os
import dash_design_kit as ddk
from dash import dcc, html
import dash_ag_grid as dag
import dash_daq as daq
from theme import theme, tabs_styles, tab_style, tab_selected_style, second_tab_style, second_tab_selected_style


plot_config = {"displaylogo": False}

main_map_config = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d"]}

map_plot_config = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}


socat_mode = constants.socat_mode

# Refactored to eliminate repeated code by Gemini

# Define a base header
header_children_base = [
    ddk.Logo(
        src="https://www.socat.info/wp-content/uploads/2017/06/cropped-socat_cat.png",
        style={'width': '110px'}
    ),
    ddk.Title("Surface Ocean CO\u2082 Atlas Data Viewer"),
    ddk.Block(width=1, style={'max-width':'450px'}, children=[
        dcc.Dropdown(id='viewer', options=[
            {'label':"SOCAT Cruises", 'value':'cruises'},
            {'label': "SOCAT Gridded Summaries", "value":"grids"}
        ], value="cruises", clearable=False)
    ])
]

# Define base property controls
prop_prop_controls_base = [
    ddk.CardHeader("Property-Property Controls", style={'width':'10%'}),
    ddk.ControlItem(
        label="X-axis",
        children=[
            dcc.Dropdown(
                id="prop-prop-x",
                value="time",
                clearable=False,
            )
        ],
    ),
    ddk.ControlItem(
        label="Y-axis",
        children=[
            dcc.Dropdown(
                id="prop-prop-y",
                value="fCO2_recommended",
                clearable=False,
            )
        ],
    ),
    ddk.ControlItem(
        label="Color By",
        children=[
            dcc.Dropdown(
                id="prop-prop-colorby",
                value="expocode",
                clearable=False,
            )
        ],
    ),
]

# Define base cruise QC children
cruise_qc_children_base = [
    ddk.CardHeader(
        id="cruise-qc-card-header",
        title="Cruise QC for ...",
    ),
    dag.AgGrid(
        id="cruise-qc-grid",
        dashGridOptions={"pagination": True},
        columnSize="sizeToFit",
        defaultColDef={"resizable": True},
        style={
            "height": 650,
            "width": "100%",
        },
    ),
]

# Conditionally add QC-specific elements
if socat_mode == "QC_EDITOR":
    title = "Surface Ocean CO\u2082 Atlas QC Editor"

    # Add QC-specific header children
    header_children_base.extend(
        [
            ddk.Modal(
                id="debug-woce-flag",
                target_id="show-woce-edits-card",
                hide_target=True,
                children=[
                    html.Button(id="show-edits", children="DEBUG: See edited rows."),
                ],
            ),
            ddk.Modal(
                id="debug-qc-entries",
                target_id="show-qc-entries-card",
                hide_target=True,
                children=[
                    html.Button(
                        id="show-qc-entries", children="DEBUG: See added QC entries."
                    ),
                ],
            ),
        ]
    )

    woce_edits_card = ddk.Card(
        id="show-woce-edits-card", children=[dag.AgGrid(id="edited-points")]
    )
    qc_entries_card = ddk.Card(
        id="show-qc-entries-card", children=[dag.AgGrid(id="qc-entries")]
    )

    # Add QC-specific property controls
    prop_prop_controls_base.insert(
        1,
        ddk.ControlItem(
            label="Flag Selected Points",
            children=[
                ddk.Block(
                    children=[
                        ddk.Modal(
                            id="edit-flags-modal",
                            target_id="selected-points-card",
                            hide_target=True,
                            children=[
                                html.Button(
                                    id="flag",
                                    children="Set Flags for",
                                )
                            ],
                        ),
                        dcc.Dropdown(
                            id="woce-flag-to-set",
                            options=[
                                {
                                    "label": "WOCE_CO2_water",
                                    "value": "WOCE_CO2_water",
                                },
                                {
                                    "label": "WOCE_CO2_atm",
                                    "value": "WOCE_CO2_atm",
                                },
                            ],
                            value="WOCE_CO2_water",
                            clearable=False,
                        ),
                    ]
                )
            ],
        ),
    )
    prop_prop_controls_base.insert(
        2,
        ddk.Block(
            width=1,
            id="selected-points-card",
            style={
                "height": "85vh",
                "width": "85vw",
            },
            children=[
                ddk.ControlCard(
                    width=1,
                    style={"height": "35vh"},
                    orientation="h",
                    children=[
                        ddk.CardHeader(title="Set WOCE Flags"),
                        ddk.ControlItem(
                            id="set-row-item",
                            width=0.3,
                            label="Set WOCE_CO2_water Checked Rows:",
                            children=[
                                html.Button(
                                    "Set",
                                    id="set-woce-flag",
                                    style={"width": "95px"},
                                ),
                                dcc.Dropdown(
                                    id="qc-woce-co2-flag-value",
                                    placeholder="Pick a Flag Value",
                                    multi=False,
                                    style={"width": "200px"},
                                    options=[
                                        {"value": "2", "label": "2"},
                                        {"value": "3", "label": "3"},
                                        {"value": "4", "label": "4"},
                                    ],
                                ),
                            ],
                        ),
                        ddk.ControlItem(
                            label="Save Flags",
                            children=[
                                html.Button(
                                    id="save-woce-flags",
                                    children="Save Flags",
                                )
                            ],
                        ),
                        ddk.ControlItem(
                            label="Comment",
                            children=[
                                dcc.Textarea(
                                    id="comment",
                                    rows=4,
                                    cols=65,
                                )
                            ],
                        ),
                    ],
                ),
                ddk.Card(
                    style={
                        "position": "absolute",
                        "bottom": 0,
                    },
                    children=[
                        dag.AgGrid(
                            id="selected-points",
                            style={"height": "55vh"},
                            dashGridOptions={
                                "rowSelection": "multiple",
                                "suppressRowClickSelection": True,
                            },
                            rowClassRules=constants.water_edit_style,
                        )
                    ],
                ),
            ],
        ),
    )

    # Add QC-specific cruise QC children
    cruise_qc_children_base.insert(
        1,
        ddk.ControlCard(
            width=1.0,
            id="expo-menu",
            orientation="h",
            children=[
                ddk.ControlItem(
                    label="Add Item",
                    label_position="left",
                    id="cruise-qc-button-item",
                    children=[
                        ddk.Modal(
                            id="add-qc-dialog",
                            target_id="add-qc-dialog-container",
                            hide_target=True,
                            customModalToggle="cruise-qc-cancel",
                            children=[
                                html.Button(
                                    id="add-cruise-qc",
                                    children=["Add QC"],
                                )
                            ],
                        )
                    ],
                )
            ],
        ),
    )
    cruise_qc_children_base.append(
        html.Div(
            id="add-qc-dialog-container",
            style={
                "margin": "50px",
                "padding": "25px",
                "width": "58vw",
                "height": 650,
            },
            children=[
                ddk.ControlCard(
                    id="add-qc-card",
                    style={
                        "height": 600,
                        "width": "55vw",
                        "overflow-y": "scroll",
                    },
                    children=[
                        ddk.CardHeader(
                            id="cruise-qc-card-title",
                            title="Cruise QC for ...",
                        ),
                        ddk.ControlItem(
                            label="Regions",
                            children=[
                                dcc.Checklist(
                                    id="qc-region-value",
                                    options=[
                                        {"label": "Costal", "value": "costal"},
                                        {"label": "Arctic", "value": "arctic"},
                                        {
                                            "label": "Global (Override regional QC flags)",
                                            "value": "global",
                                        },
                                    ],
                                    value=["global"],
                                )
                            ],
                        ),
                        ddk.ControlItem(
                            label="Accuracy of calculated aqueous fCO2 at SST:",
                            children=[
                                dcc.RadioItems(
                                    id="fco2-comment",
                                    options=[
                                        {
                                            "label": "< 2 μatm (A, B)",
                                            "value": "fco2two",
                                        },
                                        {"label": "< 5 μatm (C)", "value": "fco2five"},
                                        {"label": "< 10 μatm (E)", "value": "fco2ten"},
                                        {
                                            "label": "> 10 μatm (F, S)",
                                            "value": "fco2bad",
                                        },
                                        {"label": "(no comment)", "value": "fco2no"},
                                    ],
                                    value="fco2no",
                                )
                            ],
                        ),
                        ddk.ControlItem(
                            label="Followed approved methods/SOP criteria:",
                            children=[
                                dcc.RadioItems(
                                    id="sop-comment",
                                    options=[
                                        {"label": "true (A, B)", "value": "soptrue"},
                                        {
                                            "label": "false (C, E) - specify not followed in additional comments",
                                            "value": "sopfalse",
                                        },
                                        {"label": "(no comment)", "value": "sopno"},
                                    ],
                                    value="sopno",
                                )
                            ],
                        ),
                        ddk.ControlItem(
                            label="Metadata documentation:",
                            children=[
                                dcc.RadioItems(
                                    id="meta-comment",
                                    options=[
                                        {
                                            "label": "complete (A, B, C, E)",
                                            "value": "metacomplete",
                                        },
                                        {"label": "(no comment)", "value": "metano"},
                                    ],
                                    value="metano",
                                )
                            ],
                        ),
                        ddk.ControlItem(
                            label="Data quality:",
                            children=[
                                dcc.RadioItems(
                                    id="data-comment",
                                    options=[
                                        {
                                            "label": "acceptable (A, B, C, E)",
                                            "value": "datagood",
                                        },
                                        {
                                            "label": "significant amount of unacceptable data (F, S)",
                                            "value": "databad",
                                        },
                                        {"label": "(no comment)", "value": "datano"},
                                    ],
                                    value="datano",
                                )
                            ],
                        ),
                        ddk.ControlItem(
                            label="High-quality cross-over:",
                            children=[
                                dcc.RadioItems(
                                    id="xover-comment",
                                    options=[
                                        {
                                            "label": "found with dataset (A)",
                                            "value": "crossfound",
                                        },
                                        {
                                            "label": "none found (B, C, E)",
                                            "value": "crossnone",
                                        },
                                        {"label": "(no comment)", "value": "crossno"},
                                    ],
                                    value="crossno",
                                )
                            ],
                        ),
                        ddk.ControlItem(
                            label="Quality Control Flag to Assign",
                            children=[
                                dcc.Dropdown(
                                    id="added-qc-flag",
                                    options=[
                                        {"label": "Comment", "value": "comment"},
                                        {"label": "A", "value": "A"},
                                        {"label": "B", "value": "B"},
                                        {"label": "C", "value": "C"},
                                        {"label": "E", "value": "E"},
                                        {"label": "F", "value": "F"},
                                        {"label": "Suspend", "value": "suspend"},
                                        {"label": "Exclude", "value": "Exclude"},
                                    ],
                                    value="comment",
                                    style={"width": "120px"},
                                    multi=False,
                                )
                            ],
                        ),
                        ddk.ControlItem(
                            label="Additional Comment",
                            children=[
                                dcc.Textarea(
                                    id="qc-additional-comment",
                                    rows=10,
                                    cols=80,
                                )
                            ],
                        ),
                        ddk.Block(
                            width=0.90,
                            children=[
                                html.Button(
                                    "Save",
                                    id="cruise-qc-save",
                                ),
                                html.Button(
                                    "Cancel!",
                                    id="cruise-qc-cancel",
                                    style={
                                        "background-color": theme["accent_negative"]
                                    },
                                ),
                            ],
                        ),
                    ],
                )
            ],
        )
    )
else:
    title = "Surface Ocean CO\u2082 Atlas Data Viewer"
    woce_edits_card = html.Div(id="show-woce-edits-card", style={"display": "none"})
    qc_entries_card = html.Div(id="show-qc-entries-card", style={"display": "none"})

# Assign the final lists
header_children = header_children_base
prop_prop_controls = prop_prop_controls_base
cruise_qc_children = cruise_qc_children_base


def get_layout(
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
):
    layout = ddk.App(
        show_editor=False,
        theme=theme,
        children=[
            dcc.Store(id="plot-data-change"),
            dcc.Store(id="map-info"),
            dcc.Store(id='map-selected-date'),
            dcc.Store(id="make-cruise-tracks"),
            dcc.Store(id="cruise_table_url"),
            dcc.Store(id='grid-data-key'),
            html.Div(id="kick", style={"visibility": "none"}),
            ddk.Header(children=header_children),
            woce_edits_card,
            qc_entries_card,
            html.Div(id='cruise-view', children=[
                dcc.Tabs(
                    id="top-level-tabs",
                    value="map",
                    style=tabs_styles,
                    children=[
                        dcc.Tab(
                            id="map-tab",
                            value="map",
                            label="Cruise Selection",
                            style=tab_style,
                            selected_style=tab_selected_style,
                            children=[
                                ddk.Card(
                                    width=0.25,
                                    children=[
                                        html.Div(
                                            style={"height": "82vh", "overflow": "scroll"},
                                            children=[
                                                ddk.ControlCard(
                                                    children=[
                                                        ddk.CardHeader(
                                                            "Selection Constraints"
                                                        ),
                                                        dcc.Loading(
                                                            children=[
                                                                html.Button(
                                                                    id="reset",
                                                                    children=["Reset"],
                                                                    disabled=True,
                                                                ),
                                                                html.Button(
                                                                    id="search",
                                                                    style={'margin-left': '5px'},
                                                                    children=[
                                                                        "Find Cruises"
                                                                    ],
                                                                    disabled=False,
                                                                ),
                                                            ]
                                                        ),
                                                    ]
                                                ),
                                                ddk.ControlCard(
                                                    children=[
                                                        ddk.CardHeader(
                                                            "Latitude/Longitude Contraint"
                                                        ),
                                                        ddk.Block(
                                                            width=1,
                                                            children=[
                                                                ddk.Block(width=0.3),
                                                                ddk.Block(
                                                                    width=0.3,
                                                                    children=[
                                                                        dcc.Input(
                                                                            id="ur_lat",
                                                                            type="text",
                                                                            value=90,
                                                                            style={
                                                                                "width": "12ch"
                                                                            },
                                                                        )
                                                                    ],
                                                                ),
                                                                ddk.Block(width=0.3),
                                                                ddk.Block(
                                                                    width=0.3,
                                                                    children=[
                                                                        dcc.Input(
                                                                            id="ll_lon",
                                                                            type="text",
                                                                            value=-180,
                                                                            style={
                                                                                "width": "12ch"
                                                                            },
                                                                        )
                                                                    ],
                                                                ),
                                                                ddk.Block(
                                                                    width=0.3,
                                                                ),
                                                                ddk.Block(
                                                                    width=0.3,
                                                                    children=[
                                                                        dcc.Input(
                                                                            id="ur_lon",
                                                                            type="text",
                                                                            value=180,
                                                                            style={
                                                                                "width": "12ch"
                                                                            },
                                                                        )
                                                                    ],
                                                                ),
                                                                ddk.Block(width=0.3),
                                                                ddk.Block(
                                                                    width=0.3,
                                                                    children=[
                                                                        dcc.Input(
                                                                            id="ll_lat",
                                                                            type="text",
                                                                            value=90,
                                                                            style={
                                                                                "width": "12ch"
                                                                            },
                                                                        )
                                                                    ],
                                                                ),
                                                                ddk.Block(width=0.3),
                                                            ],
                                                        ),
                                                    ]
                                                ),
                                                # ddk.ControlCard(children=[
                                                #     ddk.CardHeader("Variable on the Map"),
                                                #     dcc.Dropdown(id='map-variable', placeholder='Color Variable on Map', options=variable_options, value='fCO2_recommended')
                                                # ]),
                                                ddk.ControlCard(
                                                    children=[
                                                        ddk.CardHeader("Expocode"),
                                                        dcc.Dropdown(
                                                            id="expocode",
                                                            placeholder="Select expocodes",
                                                            multi=True,
                                                            clearable=True,
                                                            options=initial_expo_options,
                                                        ),
                                                    ]
                                                ),
                                                ddk.Block(
                                                    width=1,
                                                    children=[
                                                        ddk.Block(
                                                            width=0.5,
                                                            children=[
                                                                ddk.ControlCard(
                                                                    children=[
                                                                        ddk.CardHeader(
                                                                            "Region"
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="region",
                                                                            multi=True,
                                                                            placeholder="Region",
                                                                            options=[
                                                                                {
                                                                                    "value": "A",
                                                                                    "label": "North Atlantic",
                                                                                },
                                                                                {
                                                                                    "value": "C",
                                                                                    "label": "Coastal",
                                                                                },
                                                                                {
                                                                                    "value": "I",
                                                                                    "label": "Indian",
                                                                                },
                                                                                {
                                                                                    "value": "N",
                                                                                    "label": "North Pacific",
                                                                                },
                                                                                {
                                                                                    "value": "O",
                                                                                    "label": "Southern Oceans",
                                                                                },
                                                                                {
                                                                                    "value": "R",
                                                                                    "label": "Arctic",
                                                                                },
                                                                                {
                                                                                    "value": "T",
                                                                                    "label": "Tropical Pacific",
                                                                                },
                                                                                {
                                                                                    "value": "Z",
                                                                                    "label": "Tropical Atlantic",
                                                                                },
                                                                            ],
                                                                        ),
                                                                    ]
                                                                ),
                                                            ],
                                                        ),
                                                        ddk.Block(
                                                            width=0.5,
                                                            children=[
                                                                ddk.ControlCard(
                                                                    children=[
                                                                        ddk.CardHeader(
                                                                            "WOCE Flag"
                                                                        ),
                                                                        dcc.Dropdown(
                                                                            id="woce-co2-water",
                                                                            placeholder="WOCE CO\u2082 Water",
                                                                            multi=True,
                                                                            options=[
                                                                                {
                                                                                    "value": "2",
                                                                                    "label": "2",
                                                                                },
                                                                                {
                                                                                    "value": "3",
                                                                                    "label": "3",
                                                                                },
                                                                                {
                                                                                    "value": "4",
                                                                                    "label": "4",
                                                                                },
                                                                            ],
                                                                        ),
                                                                    ]
                                                                ),
                                                            ],
                                                        ),
                                                    ],
                                                ),
                                                # https://stackoverflow.com/questions/70714819/dash-plotly-datetime-selection
                                                ddk.ControlCard(
                                                    children=[
                                                        ddk.CardHeader("Date Range"),
                                                        ddk.ControlItem(
                                                            label="Start Date",
                                                            label_position="left",
                                                            children=[
                                                                dcc.Input(
                                                                    id="start-date-picker",
                                                                    value=start_date,
                                                                    type="date",
                                                                )
                                                            ],
                                                        ),
                                                        ddk.ControlItem(
                                                            label="End Date",
                                                            label_position="left",
                                                            children=[
                                                                dcc.Input(
                                                                    id="end-date-picker",
                                                                    value=end_date,
                                                                    type="date",
                                                                )
                                                            ],
                                                        ),
                                                    ]
                                                ),
                                                ddk.ControlCard(
                                                    children=[
                                                        ddk.CardHeader("Investigators"),
                                                        dcc.Dropdown(
                                                            id="investigator",
                                                            placeholder="Investigators",
                                                            clearable=True,
                                                            multi=True,
                                                            options=investigaors_options,
                                                        ),
                                                    ]
                                                ),
                                                ddk.ControlCard(
                                                    children=[
                                                        ddk.CardHeader("Valid Data"),
                                                        dcc.Dropdown(
                                                            id="valid_data",
                                                            placeholder="Must contain data for...",
                                                            searchable=True,
                                                            options=variable_options,
                                                            multi=True,
                                                        ),
                                                    ]
                                                ),
                                                ddk.ControlCard(
                                                    children=[
                                                        ddk.CardHeader("Organization"),
                                                        dcc.Dropdown(
                                                            id="organization",
                                                            placeholder="Organizations",
                                                            searchable=True,
                                                            options=organization_options,
                                                        ),
                                                    ]
                                                ),
                                                ddk.ControlCard(
                                                    children=[
                                                        ddk.CardHeader("SOCAT Version"),
                                                        dcc.Dropdown(
                                                            id="socat-version",
                                                            placeholder="Select SOCAT Version",
                                                            clearable=True,
                                                            multi=True,
                                                            options=socat_version_options,
                                                        ),
                                                    ]
                                                ),
                                                ddk.ControlCard(
                                                    children=[
                                                        ddk.CardHeader("QC Flag"),
                                                        dcc.Dropdown(
                                                            id="qc-flag",
                                                            placeholder="Select QC Flag",
                                                            clearable=True,
                                                            multi=True,
                                                            options=[
                                                                {
                                                                    "label": "A",
                                                                    "value": "A",
                                                                },
                                                                {
                                                                    "label": "B",
                                                                    "value": "B",
                                                                },
                                                                {
                                                                    "label": "C",
                                                                    "value": "C",
                                                                },
                                                                {
                                                                    "label": "E",
                                                                    "value": "E",
                                                                },
                                                                {
                                                                    "label": "Q",
                                                                    "value": "Q",
                                                                },
                                                                {
                                                                    "label": "U",
                                                                    "value": "U",
                                                                },
                                                                {
                                                                    "label": "N",
                                                                    "value": "N",
                                                                },
                                                            ],
                                                            # value=["Q", "U", "N"]
                                                        ),
                                                    ]
                                                ),
                                                ddk.ControlCard(
                                                    children=[
                                                        ddk.CardHeader("Platform Name"),
                                                        dcc.Dropdown(
                                                            id="platform-name",
                                                            placeholder="Select Platform Name",
                                                            clearable=True,
                                                            multi=True,
                                                            options=platform_name_options,
                                                        ),
                                                    ]
                                                ),
                                                ddk.ControlCard(
                                                    children=[
                                                        ddk.CardHeader("Platform Type"),
                                                        dcc.Dropdown(
                                                            id="platform-type",
                                                            placeholder="Select Platform Type",
                                                            clearable=True,
                                                            multi=True,
                                                            options=[
                                                                {
                                                                    "label": "Autonomous Surface Vehicle",
                                                                    "value": "Autonomous Surface Vehicle",
                                                                },
                                                                {
                                                                    "label": "Boat",
                                                                    "value": "Boat",
                                                                },
                                                                {
                                                                    "label": "Drifting Buoy",
                                                                    "value": "Drifting Buoy",
                                                                },
                                                                {
                                                                    "label": "Mooring",
                                                                    "value": "Mooring",
                                                                },
                                                                {
                                                                    "label": "Ship",
                                                                    "value": "Ship",
                                                                },
                                                            ],
                                                        ),
                                                    ]
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                ddk.Card(
                                    width=0.75,
                                    style={"height": "86vh"},
                                    children=[
                                        ddk.CardHeader(
                                            id="map-graph-header",
                                            title="Select search latitude and longitude range",
                                        ),
                                        ddk.Graph(
                                            id="map-graph",
                                            style={"height": "95%", "width": "95%"},
                                            config=main_map_config,
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        dcc.Tab(
                            id="table-tab",
                            value="table",
                            label="Table and Map of Selected Cruises",
                            style=tab_style,
                            selected_style=tab_selected_style,
                            children=[
                                dcc.Tabs(
                                    id="selected-cruises-tabs",
                                    value="table-sub-tab",
                                    style=tabs_styles,
                                    children=[
                                        dcc.Tab(
                                            id="table-sub-tab",
                                            label="Table of Selected Cruises",
                                            value="table-sub-tab",
                                            style=second_tab_style,
                                            selected_style=second_tab_selected_style,
                                            children=[
                                                ddk.Card(
                                                    children=[
                                                        # ddk.CardHeader(fullscreen=True),
                                                        dcc.Loading(
                                                            children=[
                                                                dag.AgGrid(
                                                                    id="table-of-cruises",
                                                                    dashGridOptions={
                                                                        "pagination": True,
                                                                        "paginationAutoPageSize": True,
                                                                    },
                                                                    columnSize="sizeToFit",
                                                                    defaultColDef={
                                                                        "resizable": True
                                                                    },
                                                                    style={
                                                                        "height": "80vh"
                                                                    },
                                                                ),
                                                            ]
                                                        )
                                                    ]
                                                )
                                            ],
                                        ),
                                        dcc.Tab(
                                            id="tracks-sub-tab",
                                            label="Map of Selected Cruises",
                                            value="tracks-sub-tab",
                                            style=second_tab_style,
                                            selected_style=second_tab_selected_style,
                                            children=[
                                                ddk.Card(
                                                    style={"height": "86vh"},
                                                    children=[
                                                        dcc.Loading(
                                                            children=[
                                                                ddk.CardHeader(
                                                                    id="cruise-tracks-header",
                                                                    title="Select search search criteria on the first tab.",
                                                                ),
                                                                html.Div(
                                                                    id="track-data-loading",
                                                                    style={
                                                                        "display": "none"
                                                                    },
                                                                ),
                                                            ]
                                                        ),
                                                        ddk.Graph(
                                                            id="cruise-tracks",
                                                            style={
                                                                "height": "95%",
                                                                "width": "95%",
                                                            },
                                                            config=map_plot_config,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        ),
                                    ],
                                )
                            ],
                        ),
                        dcc.Tab(
                            id="plots-tab",
                            value="plots",
                            label="Plots and QC",
                            style=tab_style,
                            selected_style=tab_selected_style,
                            children=[
                                ddk.Card(
                                    width=0.25,
                                    children=[
                                        ddk.ControlCard(
                                            children=[
                                                ddk.CardHeader("Download Data"),
                                                ddk.Block(
                                                    width=1,
                                                    children=[
                                                        dcc.Loading(
                                                            children=[
                                                                html.Div(style={'display':'flex'}, children=[
                                                                ddk.Modal(
                                                                    id="show-data-modal",
                                                                    target_id="show-data-card",
                                                                    hide_target=True,
                                                                    children=[
                                                                        html.Button(
                                                                            "Show",
                                                                            id="show-button",
                                                                        )
                                                                    ],
                                                                ),
                                                                html.A(
                                                                    id="csv",
                                                                    children=[
                                                                        html.Button(
                                                                            "CSV",
                                                                            id="csv-button",
                                                                            style={'margin-top': '5px', 'margin-right': '5px'}
                                                                        )
                                                                    ],
                                                                    href=full_url,
                                                                    target="_blank",
                                                                ),
                                                                html.A(
                                                                    id="netcdf",
                                                                    children=[
                                                                        html.Button(
                                                                            "netCDF",
                                                                            id="netcdf-button",
                                                                            style={'margin-top': '5px'}
                                                                        )
                                                                    ],
                                                                    href=full_url,
                                                                    target="_blank",
                                                                ),
                                                            ]),
                                                            ]
                                                        )
                                                    ],
                                                ),
                                                ddk.Card(
                                                    id="show-data-card",
                                                    children=[
                                                        ddk.CardHeader(id='show-data-header', title="Data Table for Cruise"),
                                                        dag.AgGrid(
                                                            id="show-data-grid",
                                                            dashGridOptions={
                                                                "pagination": True
                                                            },
                                                            columnSize="sizeToFit",
                                                            defaultColDef={
                                                                "resizable": True
                                                            },
                                                            style={
                                                                "height": "80vh",
                                                                # "width": "100%",
                                                            },
                                                        ),
                                                    ],
                                                ),
                                            ]
                                        ),
                                        ddk.ControlCard(
                                            children=[
                                                ddk.CardHeader("Expocode to Plot"),
                                                dcc.Dropdown(
                                                    id="plot-expocode",
                                                    multi=False,
                                                    clearable=False,
                                                ),
                                            ]
                                        ),
                                        ddk.ControlCard(
                                            children=[
                                                html.Button(id="check-crossovers", children=["Check for Crossovers"])
                                            ]
                                        ),
                                        ddk.ControlCard(
                                            children=[
                                                ddk.CardHeader("Crossover to Plot"),
                                                dcc.Loading(
                                                    dcc.Dropdown(
                                                        id="crossover-expocode",
                                                        multi=False,
                                                        clearable=True,
                                                    ),
                                                )
                                            ]
                                        ),
                                        ddk.Card(
                                            children=[
                                                html.P(id="crossover-message", children="Use button to check for crossovers.")
                                            ]
                                        ),
                                        ddk.Card(
                                            id="save-full-message-card",
                                            style={"visibility": "hidden"},
                                            children=[
                                                ddk.CardHeader(
                                                    title="These changes have been saved..."
                                                ),
                                                html.Div(id="save-full-message"),
                                                html.Button(
                                                    "OK", id="close-save-full-message"
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                                ddk.Card(
                                    width=0.75,
                                    children=[
                                        dcc.Tabs(
                                            id="plot-qc-level-tabs",
                                            style=tabs_styles,
                                            children=[
                                                dcc.Tab(
                                                    id="trajectories",
                                                    value="trajectories",
                                                    label="Map of Selected Cruise",
                                                    style=second_tab_style,
                                                    selected_style=second_tab_selected_style,
                                                    children=[
                                                        ddk.Card(
                                                            style={"height": "85vh"},
                                                            children=[
                                                                dcc.Loading(
                                                                    children=[
                                                                        ddk.CardHeader(
                                                                            id="trace-graph-header",
                                                                            title="Selected Cruise                                       ",
                                                                            children=[
                                                                                dcc.Dropdown(
                                                                                    id="trace-variable",
                                                                                    options=variable_options,
                                                                                    value="fCO2_recommended",
                                                                                    multi=False,
                                                                                )
                                                                            ],
                                                                        ),
                                                                    ]
                                                                ),
                                                                # dcc.Loading(
                                                                ddk.Graph(
                                                                    id="trace-graph",
                                                                    style={
                                                                        "height": "95%",
                                                                        "width": "95%",
                                                                    },
                                                                    config=map_plot_config,
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
                                                            ],
                                                        )
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    id="prop-prop",
                                                    value="prop-prop-plot",
                                                    label="Property-Property Plot",
                                                    style=second_tab_style,
                                                    selected_style=second_tab_selected_style,
                                                    children=[
                                                        ddk.ControlCard(
                                                            id="prop-prop-controls",
                                                            orientation="h",
                                                            children=prop_prop_controls,
                                                        ),
                                                        ddk.Card(
                                                            children=[
                                                                dcc.Loading(
                                                                    children=[
                                                                        ddk.CardHeader(
                                                                            id="prop-prop-graph-header",
                                                                            title="Property-proptery plot",
                                                                        ),
                                                                        dcc.Graph(
                                                                            id="prop-prop-graph",
                                                                            style={
                                                                                "height": "60vh"
                                                                            },
                                                                            config=plot_config,
                                                                        ),
                                                                        html.Div(
                                                                            id="prop-prop-loading"
                                                                        ),  # Hides the card while the data is being pulled from ERDDAP
                                                                    ]
                                                                )
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    id="thumbnails-tab",
                                                    value="prop-prop-thumbs",
                                                    label="Thumbnail Plots",
                                                    style=second_tab_style,
                                                    selected_style=second_tab_selected_style,
                                                    children=[
                                                        ddk.Card(
                                                            children=[
                                                                dcc.Loading(
                                                                    color="white",
                                                                    type="dot",
                                                                    children=[
                                                                        ddk.CardHeader(
                                                                            id="thumbnails-header",
                                                                            title="Thumbnail Plots",
                                                                        ),
                                                                    ],
                                                                ),
                                                                dcc.Loading(
                                                                    dcc.Graph(
                                                                        id="thumbnails-graph",
                                                                        style={
                                                                            "height": image_height
                                                                            + 40
                                                                        },
                                                                        config=plot_config,
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
                                                                ),
                                                            ]
                                                        ),
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    id="cruise-qc-tab",
                                                    value="cruise-qc",
                                                    label="Cruise QC",
                                                    style=tab_style,
                                                    selected_style=tab_selected_style,
                                                    children=[
                                                        ddk.Card(
                                                            width=1,
                                                            id="cruise-qc-card",
                                                            style={"height": "90vh"},
                                                            children=cruise_qc_children,
                                                        ),
                                                    ],
                                                ),
                                            ],
                                        )
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ]),
            html.Div(id='grid-view', style={'display': 'none'}, children=[
                ddk.Block(width=.25, children=[
                    dcc.Tabs(
                    id="grid-control-tabs",
                    value="dataset",
                    style=tabs_styles,
                    children=[
                        dcc.Tab(                            
                            id="dataset-tab",
                            value="dataset",
                            label="Dataset Selection",
                            style=tab_style,
                            selected_style=tab_selected_style,
                            children=[
                                ddk.ControlCard(children=[
                                    ddk.CardHeader("SOCAT Release"),
                                    dcc.Dropdown(id='grid-socat-release', options=socat_release_options)
                                ]),
                                ddk.ControlCard(children=[
                                    ddk.CardHeader("Data set"),
                                    dcc.Dropdown(id='grid-dataset', options=grid_dataset_options)
                                ]),
                         ]),
                        dcc.Tab(id="plot-tab", value="grid-plot", label="Plot Controls", style=tab_style, selected_style=tab_selected_style, children=[
                            ddk.ControlCard(children=[
                                ddk.CardHeader("Variable"),
                                dcc.Dropdown(id='grid-variable')
                            ]),
                            ddk.ControlCard(children=[
                                ddk.CardHeader(children=[
                                    dcc.Checklist(id='time-aggregations-switch', options=[
                                        {'label':'Apply mean, min, max, or sum over time range.', 'value': 'on'}
                                    ]),
                                ]),
                                dcc.Dropdown(id='aggregation-type', style={'display': 'none'}, options=[
                                    {'label': 'Mean', 'value': 'mean'},
                                    {'label': 'Min', 'value': 'min'},
                                    {'label': 'Max', 'value': 'max'},
                                    {'label': 'Sum', 'value': 'sum'},
                                ])
                            ]),
                            ddk.ControlCard(children=[
                                ddk.CardHeader(style={'margin-top': '10px'}, children="Start Date"),
                                ddk.Block(width=1, children=[
                                    ddk.Block(width=.3, children=[
                                        dcc.Dropdown(id='grid-year', placeholder='Year'),
                                    ]),
                                    ddk.Block(width=.7, children=[
                                        dcc.Dropdown(id='grid-month', placeholder="Month", options=[
                                            {'label': 'January', 'value': '01'},
                                            {'label': 'February', 'value': '02'},
                                            {'label': 'March', 'value': '03'},
                                            {'label': 'April', 'value': '03'},
                                            {'label': 'May', 'value': '05'},
                                            {'label': 'June', 'value': '06'},
                                            {'label': 'July', 'value': '07'},
                                            {'label': 'August', 'value': '08'},
                                            {'label': 'September', 'value': '09'},
                                            {'label': 'October', 'value': '10'},
                                            {'label': 'November', 'value': '11'},
                                            {'label': 'December', 'value': '12'},
                                        ])
                                    ])
                                ]), 
                            ]),
                            html.Div(id='aggregation-controls', style={'display':'none'}, children=[
                                ddk.ControlCard(id='grid-end-date', children=[
                                    ddk.CardHeader(style={'margin-top': '10px'}, children="End Date"),
                                    ddk.Block(width=1, children=[
                                        ddk.Block(width=.3, children=[
                                            dcc.Dropdown(id='grid-year-end', placeholder='Year'),
                                        ]),
                                        ddk.Block(width=.7, children=[
                                            dcc.Dropdown(id='grid-month-end', placeholder="Month", options=[
                                                {'label': 'January', 'value': '01'},
                                                {'label': 'February', 'value': '02'},
                                                {'label': 'March', 'value': '03'},
                                                {'label': 'April', 'value': '03'},
                                                {'label': 'May', 'value': '05'},
                                                {'label': 'June', 'value': '06'},
                                                {'label': 'July', 'value': '07'},
                                                {'label': 'August', 'value': '08'},
                                                {'label': 'September', 'value': '09'},
                                                {'label': 'October', 'value': '10'},
                                                {'label': 'November', 'value': '11'},
                                                {'label': 'December', 'value': '12'},
                                            ])
                                        ])
                                    ]),
                                ]),
                            ]),
                            ddk.ControlCard(id='grid-download', children=[
                                ddk.CardHeader("Grid Data Download"),
                                dcc.Loading(children=[
                                    html.Div(style={'display':'flex'}, children=[
                                        ddk.Modal(
                                            id="grid-show-data-modal",
                                            target_id="grid-show-data-card",
                                            hide_target=True,
                                            children=[
                                                html.Button(id='grid-show-button', children="Show"),
                                            ]),
                                        html.A(id='grid-csv', href='', target="_blank", referrerPolicy="no-referrer", children=[html.Button(id='grid-csv-button', children="CSV", style={'margin-left': '5px', 'margin-right': '5px', 'margin-top': '5px'})]),
                                        html.A(id='grid-netcdf', href='', target="_blank", children=[html.Button(id='grid-netcdf-button', children="netCDF", style={'margin-top': '5px'})]),
                                    ])
                                ])
                            ]),
                        ]),
                    ]),
                ]),
                ddk.Card(width=.75, style={'height': '86vh'}, children=[
                    dcc.Loading(ddk.CardHeader(id='grid-title')),
                    ddk.Graph(id='grid-map', style={'height':'95%', 'width': '95%'})
                ]),
                    ddk.Card(
                        id="grid-show-data-card",
                        children=[
                            ddk.CardHeader(id='grid-show-data-header', title="Data Table for Gridded Summary"),
                            dag.AgGrid(
                                id="grid-show-data-grid",
                                dashGridOptions={
                                    "pagination": True
                                },
                                columnSize="sizeToFit",
                                defaultColDef={
                                    "resizable": True
                                },
                                style={
                                    "height": "80vh",
                                    # "width": "100%",
                                },
                            ),
                        ],
                    ),
            ]),
            ddk.Footer(
                children=[
                    html.Hr(),
                    ddk.Block(
                        children=[
                            ddk.Block(
                                width=0.3,
                                children=[
                                    html.Div(
                                        children=[
                                            dcc.Link(
                                                "National Oceanic and Atmospheric Administration",
                                                href="https://www.noaa.gov/",
                                                style={"font-size": ".8em"},
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        children=[
                                            dcc.Link(
                                                "Pacific Marine Environmental Laboratory",
                                                href="https://www.pmel.noaa.gov/",
                                                style={"font-size": ".8em"},
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        children=[
                                            dcc.Link(
                                                "oar.pmel.webmaster@noaa.gov",
                                                href="mailto:oar.pmel.webmaster@noaa.gov",
                                                style={"font-size": ".8em"},
                                            )
                                        ]
                                    ),
                                    dcc.Link(
                                        "DOC |",
                                        href="https://www.commerce.gov/",
                                        style={"font-size": ".8em"},
                                    ),
                                    dcc.Link(
                                        " NOAA |",
                                        href="https://www.noaa.gov/",
                                        style={"font-size": ".8em"},
                                    ),
                                    dcc.Link(
                                        " OAR |",
                                        href="https://www.research.noaa.gov/",
                                        style={"font-size": ".8em"},
                                    ),
                                    dcc.Link(
                                        " PMEL |",
                                        href="https://www.pmel.noaa.gov/",
                                        style={"font-size": ".8em"},
                                    ),
                                    dcc.Link(
                                        " Privacy Policy |",
                                        href="https://www.noaa.gov/disclaimer",
                                        style={"font-size": ".8em"},
                                    ),
                                    dcc.Link(
                                        " Disclaimer |",
                                        href="https://www.noaa.gov/disclaimer",
                                        style={"font-size": ".8em"},
                                    ),
                                    dcc.Link(
                                        " Accessibility",
                                        href="https://www.pmel.noaa.gov/accessibility",
                                        style={"font-size": ".8em"},
                                    ),
                                ],
                            ),
                            ddk.Block(
                                width=0.7,
                                children=[
                                    html.Img(
                                        src=footer_image,
                                        style={"height": "90px", "padding": "14px"},
                                    )
                                ],
                            ),
                        ]
                    ),
                ],
                style={"color": "white"},
            ),
        ],
    )
    return layout
