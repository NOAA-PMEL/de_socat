from theme import theme
from dash import html
import dash_design_kit as ddk
import os
import redis
from sqlalchemy import all_, create_engine, engine
from sqlalchemy.pool import NullPool

QUEUE_URL = "https://sqs.us-east-1.amazonaws.com/807604749487/dash-sqs"

TIME_TO_LIVE = 3600 # time to live af an entry in the redis cache

FULL_CRUISE_DATA_FIELD_NAME = 'cruise_data'
COLUMNS_FOR_WOCE_EDIT_TABLE_FIELD_NAME = 'woce_edit_columns'
CROSSOVER_DATA_FIELD_NAME = "crossovers"
TABLE_OF_CRUISES_URL_FIELD_NAME = 'table_of_cruises'
CURRENT_GRID_DATA = 'grid_data'

short_format = '%Y-%m-%d'

zoom = 1
center = {'lon': 0.0, 'lat': 0.0}
map_limits = {"west": -180, "east": 180, "south": -89, "north": 89}

map_height = 600
map_width = 1200

# This variable by default with have the value "VIEWER"
# If the environment variable is set and the value is "QC_EDITOR"
# then the UI will be configured to expose the buttons which allow
# QC flags and WOCE flags to be set.
socat_mode = os.environ.get("SOCAT_MODE", "VIEWER")

VERSION_ID = "socat_v2026"

if socat_mode == "VIEWER":
    map_title_base = 'Trajectory from the latest SOCAT Decimated Data Set '
    decimated_url = f'https://data.pmel.noaa.gov/socat/erddap/tabledap/{VERSION_ID}_decimated'
    full_url = f'https://data.pmel.noaa.gov/socat/erddap/tabledap/{VERSION_ID}_fulldata'
    grid_url = 'https://data.pmel.noaa.gov/socat/erddap/griddap'
    qc_flags = []
    save_socat_edits = ''
elif socat_mode == "QC_EDITOR":
    map_title_base = 'Trajectory from the latest SOCAT Decimated Data Set '
    # decimated_url = 'https://datalocal.pmel.noaa.gov/erddap/tabledap/socat_latest_decimated'
    # full_url = 'https://datalocal.pmel.noaa.gov/erddap/tabledap/socat_latest_fulldata'
    decimated_url = f'https://data.pmel.noaa.gov/socat/erddap/tabledap/{VERSION_ID}_decimated'
    full_url = f'https://data.pmel.noaa.gov/socat/erddap/tabledap/{VERSION_ID}_fulldata'
    grid_url = 'https://data.pmel.noaa.gov/socat/erddap/griddap'
    qc_flags = ["Q", "U", "N"]
    save_socat_edits = 'https://sour.pmel.noaa.gov:8940/SaveSocatEdits'



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


dtype_definitions = {'expocode': str, 'organization': str, 'investigators': str, 'platform_name': str, 'platform_type': str, 'qc_flag': str, 'socat_version': str}


water_edit_style = {
                "green-cell": "params.data.WOCE_CO2_water == 2",
                "yellow-cell": "params.data.WOCE_CO2_water == 3",
                "red-cell": "params.data.WOCE_CO2_water == 4"
            }

atm_edit_style = {
                "green-cell": "params.data.WOCE_CO2_atm == 2",
                "yellow-cell": "params.data.WOCE_CO2_atm == 3",
                "red-cell": "params.data.WOCE_CO2_atm == 4"
            }


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


edits_table = 'socat_edits'
qc_entries_table = 'qc_entries'

regions = {
    "A": {
        'll': {
            'longitude': -80,
            'latitude': 20
        },
        'ur': {
            'longitude': 0,
            'latitude': 70
        },
        "name": 'North Atlantic'
    },
    "C": {
        'll': {
            'longitude': -180,
            'latitude': -90
        },
        'ur': {
            'longitude': 180,
            'latitude': 90
        }
    },
    "I": {
        'll': {
            'longitude': 20,
            'latitude': -60
        },
        'ur': {
            'longitude': 130,
            'latitude': 30
        }
    },
    "N": {
        'll': {
            'longitude': 110,
            'latitude': 70
        },
        'ur': {
            'longitude': -120,
            'latitude': 0
        }
    },
    "O": {
        'll': {
            'longitude': -180,
            'latitude': -90
        },
        'ur': {
            'longitude': 180,
            'latitude': -60
        }
    },
    "R": {
        'll': {
            'longitude': -180,
            'latitude': 66
        },
        'ur': {
            'longitude': 180,
            'latitude': 90
        }
    },
    "T": {
        'll': {
            'longitude': 120,
            'latitude': -30
        },
        'ur': {
            'longitude': -70,
            'latitude': 30
        }
    },
    "Z": {
        'll': {
            'longitude': -55,
            'latitude': 5
        },
        'ur': {
            'longitude': -15,
            'latitude': 25
        }
    }   
}
region_names = {
    "A" : 'North Atlantic',
    "C" : "Coastal",
    "I" : 'Indian',
    "N" : "North Pacific",
    "O" : "Southern Oceans",
    "R" : "Arctic",
    "T" : "Tropical Pacific",
    "Z" : "Tropical Atlantic"
}

redis_instance = redis.StrictRedis.from_url(os.environ.get("REDIS_URL", "redis://127.0.0.1:6379"))

if __name__ == '__main__':
    for id in region_names:
        print(f'{region_names[id]} covers:')
        print(regions[id])         