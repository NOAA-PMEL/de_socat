import math
#
# Takes, ll and ur lat and lon and creates a data frame with the 
#


def get_socat_subset(ll_longitude, ur_longitude, ll_latitude, ur_latitude):
    print('Starting coordinate constraint processing...')
    lat_con = '&latitude>='+str(ll_latitude)+'&latitude<='+str(ur_latitude)
    center_lat = (ur_latitude - ll_latitude)/2.0
    center_lon = (ur_longitude - ll_longitude)/2.0
    if ll_longitude < ur_longitude:
        # Going west to east does not cross dateline, normal constraint
        print('east west normal')
        if (-180 <= ll_longitude <= 180) and (-180 <= ur_longitude <= 180):
            lon_con = '&longitude>='+str(ll_longitude)+'&longitude<='+str(ur_longitude)
    elif ll_longitude > ur_longitude:
        if (-180 <= ll_longitude <= 180.0) and (-180 <= ur_longitude <= 180.0):
            center_lon = (ll_longitude - ur_longitude)/2.0
            print('west to east')
            # Going west to east over dateline, but not greenwich, convert to lon360 from -180 to 180 input
            if ur_longitude < 0 < ll_longitude:
                print('has 360')
                print('xlo=', str(ll_longitude), ' xhi=', ur_longitude, ' xhi360=', angle0360(ur_longitude))
                lon_con = '&lon360>='+str(angle0360(ll_longitude)) + '&lon360<='+str(angle0360(ur_longitude))
            else:
                lon_con = '&lon360<='+str(angle0360(ll_longitude)) + '&lon360>='+str(angle0360(ur_longitude))

    return {'lon': lon_con, 'lat': lat_con, 'center': {'lat': center_lat, 'lon':center_lon}}


##
# This converts an angle (in degrees) into the range >=0 to <360.
#
def angle0360(degrees):
    if not math.isfinite(degrees):
        return 0;

    while degrees < 0:
        degrees += 360.
    while degrees >= 360.:
        degrees -= 360.
        
    return degrees
