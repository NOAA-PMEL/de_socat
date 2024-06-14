import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
import pymysql
import os

edits_table = 'socat_edits'

# Create a SQLAlchemy connection string from the environment variable `DATABASE_URL`
# automatically created in your dash app when it is linked to a postgres container
# on Dash Enterprise. If you're running locally and `DATABASE_URL` is not defined,
# then this will fall back to a connection string for a local postgres instance
#  with username='postgres' and password='password'
connection_string = "postgresql+pg8000" + os.environ.get(
    "DATABASE_URL", "postgresql://postgres:password@127.0.0.1:5432"
).lstrip("postgresql")


params = {
    'host': 'sour.pmel.noaa.gov',
    'user': 'scientist_pd',
    'password': os.environ['MYSQL_PASSWORD'],
    'port': 3306,
    'db': 'SOCATv2022_pd',
    'charset': 'utf8mb4'
}

# Returns a new connection to the database
def getconn():
    return pymysql.connect(host=params['host'],
                             user=params['user'],
                             password=params['password'],
                             db=params['db'],
                             charset=params['charset'],
                             port=params['port'])


mysql_pool = NullPool(getconn)
mysql_engine = create_engine("mysql+pymysql://", pool=mysql_pool)


def get_cruise_qc(expocode):
    with mysql_engine.connect() as connection:
    # In line with best practices, we use SQLAlchemy's `text` object so that user inputs are escaped, and the risk
    # a SQL injection attack is minimized.
        qc_query = '''select q.expocode as "Expocode", r.region_name as "Region", qc_flag as "Flag", FROM_UNIXTIME(q.qc_time) as "Flag Timestamp", v.realname as "Reviewer", ROUND(q.socat_version,1) as "Version", q.qc_comment as "Comments"
                        FROM

                        QCEvents q
                        LEFT JOIN Regions r
                        ON
                        q.region_id = r.region_id
                        LEFT JOIN Reviewers v
                        ON
                        q.reviewer_id = v.reviewer_id
                        where expocode="{}"
                        and q.qc_flag <> 'R'
                        order by q.qc_time
                        '''
        qc_query = qc_query.format(expocode)
        print(qc_query)
        qc_records = pd.read_sql(qc_query, connection)
        return qc_records


# Create a SQLAlchemy engine object. This object initiates a connection pool
# so we create it once here and import into app.py.
# `poolclass=NullPool` prevents the Engine from using any connection more than once. You'll find more info here:
# https://docs.sqlalchemy.org/en/14/core/pooling.html#using-connection-pools-with-multiprocessing-or-os-fork
postgres_engine = create_engine(connection_string, poolclass=NullPool)

def show_saves():
    updated_df = pd.read_sql(
        "SELECT * FROM {};".format(edits_table), postgres_engine
    )
    return updated_df


def delete_all_rows():
    delete = 'DELETE FROM {};'
    postgres_engine.execute(delete.format(edits_table))


def drop_edits():
    delete = 'DROP TABLE {};'
    postgres_engine.execute(delete.format(edits_table))