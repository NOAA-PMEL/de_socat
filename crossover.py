from urllib.parse import quote
import pandas as pd
import math
from datetime import datetime, timedelta
import numpy as np
import sys

EARTH_AUTHALIC_RADIUS_KM = 6371.007
CUTOFF = 80.
SPEED = 30.
MIN_FCO2_DIFF = 5.0
MIN_TEMP_DIFF = 0.3

# Max allowable difference in time in milliseconds
time_delta =  math.ceil(24.0 * 60.0 * 60.0 * 1000.0 * CUTOFF / SPEED)

# Max allowable difference in latitude in degrees
lat_delta = (CUTOFF / EARTH_AUTHALIC_RADIUS_KM) * (180.0 / math.pi)

def crossover(traj_id_name, tid, dataurl):
    crossovers = {}
    laturl = dataurl + ".csv?"+quote(traj_id_name+",latitude,longitude,time&"+traj_id_name+"=\""+tid+"\"&distinct()&orderByMinMax(\"latitude\")")
    latdf = pd.read_csv(laturl, skiprows=[1])
    lat_min = latdf['latitude'].iloc[0].astype(np.float64)
    lat_max = latdf['latitude'].iloc[1].astype(np.float64)

    lonurl = dataurl + ".csv?"+quote(traj_id_name+",longitude,latitude,time&"+traj_id_name+"=\""+tid+"\"&distinct()&orderByMinMax(\"longitude\")")
    londf = pd.read_csv(lonurl, skiprows=[1])
    lon_min = londf['longitude'].iloc[0].astype(str)
    lon_max = londf['longitude'].iloc[1].astype(str)

    timeurl = dataurl + ".csv?"+quote(traj_id_name+",time,latitude,longitude&"+traj_id_name+"=\""+tid+"\"&distinct()&orderByMinMax(\"time\")")
    timedf = pd.read_csv(timeurl, skiprows=[1])

    tmin = timedf['time'].iloc[0]
    tmax = timedf['time'].iloc[1]

    tmin_obj = datetime.fromisoformat(tmin) - timedelta(milliseconds=time_delta*2)
    tmax_obj = datetime.fromisoformat(tmax) + timedelta(milliseconds=time_delta*2)

    tmin = tmin_obj.isoformat()
    tmax = tmax_obj.isoformat()

    lat_min = lat_min - lat_delta*2
    if lat_min < -90.:
         lat_min = -90.
    lat_max = lat_max + lat_delta*2
    if lat_max > 90:
        lat_max = 90.

    crossoversurl = dataurl + ".csv?"+quote(traj_id_name+"&"+traj_id_name+"!=\""+tid+"\"&distinct()&time>="+tmin+"&time<="+tmax+
                                                "&latitude>="+str(lat_min)+"&latitude<="+str(lat_max)+
                                                "&longitude>="+str(lon_min)+"&longitude<="+str(lon_max), "UTF-8")

    cross_df = None
    try:
        cross_df = pd.read_csv(crossoversurl, skiprows=[1])
    except:
        pass

    filter = tid[0:4]

    if cross_df is not None and not cross_df.empty:
    
        selected_cruise_url = dataurl + ".csv?"+quote(traj_id_name+",time,latitude,longitude,temp,fCO2_recommended&"+traj_id_name+"=\""+tid+"\"&orderBy(\"time\")")
        selected_cruise_df = pd.read_csv(selected_cruise_url, skiprows=[1], dtype={'expocode': 'str'}) # Some of them look like numbers
        for index, row in cross_df.iterrows():
            cid = str(row['expocode'])
            if not cid.startswith(filter):
                potential_cross_url = dataurl + ".csv?"+quote(traj_id_name+",time,latitude,longitude,temp,fCO2_recommended&"+traj_id_name+"=\""+cid+"\"&orderBy(\"time\")")
                potential_cross_df = pd.read_csv(potential_cross_url, skiprows=[1], dtype={'expocode': 'str'}) 
                cross = check_crossover(selected_cruise_df, potential_cross_df)
                if cross:
                    crossovers.update(cross)
    import json       
    # DEBUG print(json.dumps(crossovers, indent=4))
    return crossovers


# This could be implemented using pandas by taking the cross product of the two dataframes
# and computing the crossing criterial for each row.
# It might be faster, but might need some memory managment
# For now implementing with the highly frowned upon looping of the dataframe.
def check_crossover(selected_cruise_df, potential_cross_df):
    cross = {}
    min_distance = sys.float_info.max
    # Save the min date for the crossing cruise.
    cruise_min_date = None
    cruise_max_date = None
    for i, row in selected_cruise_df.iterrows():
        tid = str(row['expocode'])
        date = row['time']
        dt = datetime.fromisoformat(date)
        time = dt.timestamp() * 1000 # convert to milliseconds
        lat = row['latitude']
        lon = row['longitude']
        temp = row['temp']
        if pd.isna(temp):
            temp = sys.float_info.max - 10000.
        
        fCO2 = row['fCO2_recommended']
        if pd.isna(fCO2):
            fCO2 = sys.float_info.max - 10000.

        for j, crossing_row in potential_cross_df.iterrows():
            crossing_id = crossing_row['expocode']
            # Don't care that it crosses itself
            if tid == crossing_id:
                return None
            
            crossing_date = crossing_row['time']
            if j == 0:
                cruise_min_date = crossing_date
                cruise_max_date = potential_cross_df['time'].max()
            cdt = datetime.fromisoformat(crossing_date)
            crossing_time = cdt.timestamp()*1000 # Want millis not seconds 
            crossing_lat = crossing_row['latitude']
            crossing_lon = crossing_row['longitude']

            crossing_temp = crossing_row['temp']
            if pd.isna(crossing_temp):
                crossing_temp = sys.float_info.max - 10000.
                       
            crossing_fCO2 = crossing_row['fCO2_recommended']
            if pd.isna(crossing_fCO2):
                crossing_fCO2 = sys.float_info.max - 10000.
            

            if crossing_time > (time + time_delta):
                # The rest of the second cruise occurred far
                # later than the point of first cruise.
                # Go on to the next point of the first cruise.
                break
            
            # if crossing_time < time - time_delta:
            #     # This point of the second cruise occurred far
            #     # earlier than the point of the first cruise.
            #     # Go on to the next point of the second cruise.
            #     continue
            
            # if abs(lat - crossing_lat) > lat_delta:
            #     # Differences in latitudes are too large.
            #     # Go on to the next point of the second cruise.
            #     continue
            
            if (crossing_time >= (time - time_delta)) and not (abs(lat - crossing_lat) > lat_delta):
                loc_time_dist = distance_to(lat, lon, time, crossing_lat, crossing_lon, crossing_time, SPEED, EARTH_AUTHALIC_RADIUS_KM)
                if min_distance > loc_time_dist and loc_time_dist <= CUTOFF:
                    temp_diff = abs(crossing_temp - temp)
                    fCO2_diff = abs(crossing_fCO2 - fCO2)
                    if temp_diff < MIN_TEMP_DIFF and fCO2_diff  < MIN_FCO2_DIFF:
                        min_distance = loc_time_dist
                        cross ={
                            crossing_id: {
                                'loc_time_dist': loc_time_dist, 
                                'crossing_lat': crossing_lat, 
                                'crossing_lon': crossing_lon, 
                                'crossing_date': crossing_date, 
                                'cruise_min_date': cruise_min_date, 
                                'cruise_max_date': cruise_max_date
                            }
                        }


    return cross


def distance_to(lat, lon, time, crossing_lat, crossing_lon, crossing_time, speed, radius):
    # Returns the location-time "distance" to another location-time point
    #  using the provided conversion factor for time to distance and the
    # given radius for a spherical Earth.  Uses the haversine formula to
    # compute the great circle distance from the longitudes and latitudes.
    #
    # @param other
    #      the other location-time point to use
    # @param speed
    #      the number of kilometers to use for 24 hours of time;
    #      can be zero to obtain a distance without a time contribution
    # @param radius
    #      the radius of a spherical Earth in kilometers
    # @return
    #      the location-time distance between this location-time point
    #      and other in kilometers
     
    # Convert longitude and latitude degrees to radians
    lat1 = lat * math.pi / 180.0
    lat2 = crossing_lat * math.pi / 180.0
    lon1 = lon * math.pi / 180.0
    lon2 = crossing_lon * math.pi / 180.0
   
    # Use the haversine formula to compute the great circle distance,
    # in radians, between the two (longitude, latitude) points.
        
    dellat = math.sin(0.5 * (lat2 - lat1))
    dellat *= dellat
    dellon = math.sin(0.5 * (lon2 - lon1))
    dellon *= dellon * math.cos(lat1) * math.cos(lat2)
    distance = 2.0 * math.asin(math.sqrt(dellon + dellat))
    # Convert the great circle distance from radians to kilometers
    distance *= radius

    if speed != 0.0:
        # Get the time difference in days (24 hours)
        deltime = (crossing_time - time) / (24.0 * 60.0 * 60.0 * 1000.0)
        # Convert to the time difference to kilometers
        deltime *= speed
        # Combine the time distance with the surface distance
        distance = math.sqrt(distance * distance + deltime * deltime)

    return distance
    