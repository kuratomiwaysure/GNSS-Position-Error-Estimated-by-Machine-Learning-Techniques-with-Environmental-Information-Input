"------------------------------------------------------------------------------------------"
"This file is used as the reader for all data coming from GNSS receiver and post processing"
"------------Cooridnates conversion done with previously built code at Waysure-------------"
"-----------------------------2st Stage: Sky consideration---------------------------------"


import json
import numpy as np
import math
import random
import time


"Reads sky obstruction information from file"
def readSkyFile(text):

    sky_list=json.load(open(text))

    return sky_list


"Finds the nearest sky obstruction information input from the file"
def findNearestSky(sky_input,time):

    diff_prev = 10000000
    total_seconds_ref = timeSeconds(time)
    sky_current = 1

    for i in sky_input:

        time_vec = i['time']
        sky = i['sky']
        total_seconds_vec = timeSeconds(time_vec)

        diff = abs(total_seconds_ref-total_seconds_vec)

        if(diff < diff_prev):

            diff_prev = diff
            sky_current = sky

    return sky_current


"Returns the list of used satellites' id as a single row list"
def makeListOfSatellites(used_satellites_area,length_area,lsq_satellites,length_lsq_satellites):

    satellitesList = list()
    all_inf_sat = list()

    for i in range(0,length_area):

        sat_id = used_satellites_area[i]['satellite_id']
        sat_const = used_satellites_area[i]['constellation']
        sat_multipath = used_satellites_area[i]['multipath']
        sat_raim = used_satellites_area[i]['raim']
        sat_cycle = used_satellites_area[i]['cycle_slip']
        sat_amb = used_satellites_area[i]['amb_type']

        sat_id_const = str(sat_id)+str(sat_const)
        sat_used = used_satellites_area[i]['used']

        if sat_used == 1:

            if length_lsq_satellites == 0:

                sat_lsq_residual = 0

                sat_sample = {
                    'satellite_id_const': sat_id_const,
                    'multipath': sat_multipath,
                    'raim': sat_raim,
                    'cycle_slip': sat_cycle,
                    'amb_type': sat_amb,
                    'lsq_residual': sat_lsq_residual
                    }

            else:

                for j in range(0,length_lsq_satellites):

                    sat_id_lsq = lsq_satellites[j]['satellite_id']
                    sat_const_lsq = lsq_satellites[j]['constellation']
                    sat_lsq_residual = lsq_satellites[j]['lsq_residual']

                    if sat_id == sat_id_lsq and sat_const == sat_const_lsq:

                        sat_sample = {
                            'satellite_id_const': sat_id_const,
                            'multipath': sat_multipath,
                            'raim': sat_raim,
                            'cycle_slip': sat_cycle,
                            'amb_type': sat_amb,
                            'lsq_residual': sat_lsq_residual
                            }

            satellitesList.append(sat_id_const)
            all_inf_sat.append(sat_sample)

    return satellitesList,all_inf_sat


"Gets info from a used satellite"
def getUsedSatellite(satellites_used_all_info,individual_id_const):

    for i in satellites_used_all_info:

        sat_id_const = i['satellite_id_const']

        if individual_id_const == sat_id_const:

            sat = i

    return sat


"Verifies that the satellite was used so as to save its data into the vector input"
def checkSatelliteUsed(sat_list,satellite):

    if satellite in sat_list:

        return 1

    else:

        return 0


"Parses Trimble ENU data into readable data for coordinate transformation"
def parseTrimble(data):

    time_list = list()
    east_list = list()
    north_list = list()
    up_list = list()

    vectordata = data.split()

    vectordata.pop(7)
    vectordata.pop(6)
    vectordata.pop(5)
    vectordata.pop(4)
    vectordata.pop(3)
    vectordata.pop(2)
    vectordata.pop(1)
    vectordata.pop(0)

    vectordata = np.reshape(vectordata,(int(len(vectordata)/5),5))

    east = vectordata[:,0]
    north = vectordata[:,1]
    up = vectordata[:,2]
    time = vectordata[:,4]
    [time_list.append(x) for x in time]
    [east_list.append(float(x)) for x in east]
    [north_list.append(float(x)) for x in north]
    [up_list.append(float(x)) for x in up]

    return time_list,east_list,north_list,up_list


"Parses Trimble ENU data into readable data for coordinate transformation"
def parseTrimble2(data):

    time_list = list()
    east_list = list()
    north_list = list()
    up_list = list()

    vectordata = data.split()

    vectordata.pop(4)
    vectordata.pop(3)
    vectordata.pop(2)
    vectordata.pop(1)
    vectordata.pop(0)

    vectordata = np.reshape(vectordata,(int(len(vectordata)/5),5))

    east = vectordata[:,0]
    north = vectordata[:,1]
    up = vectordata[:,2]
    time = vectordata[:,4]

    for x in range(0,len(time)):

        time_list.append(time[x])

    for x in range(0,len(east)):

        east_list.append(float(east[x]))

    for x in range(0,len(north)):

        north_list.append(float(north[x]))

    for x in range(0,len(up)):

        up_list.append(float(up[x]))

    [time_list.append(x) for x in time]
    [east_list.append(float(x)) for x in east]
    [north_list.append(float(x)) for x in north]
    [up_list.append(float(x)) for x in up]

    return time_list,east_list,north_list,up_list


"Parses NMEA LLA data into readable data for coordinate transformation"
def parseNMEA(data):

    time_list = list()
    lat_list = list()
    lon_list = list()
    alt_list = list()

    data = data.splitlines()
    num_lines = len(data)

    for i in range(1,num_lines):

        line = data[i].split(',')

        "Put time in comparable format"
        time_str = line[1]
        hour = time_str[0:2]
        minute = time_str[2:4]
        second = time_str[4:9]

        time = str(hour+":"+minute+":"+second)

        "Calculate Latitude and Longitude"
        "Measurements in degrees"
        lat_deg = float(line[2][0:2])
        lon_deg = float(line[4][0:3])

        "Measurements in minutes"
        lat_minute = float(line[2][2:-1])
        lon_minute = float(line[4][3:-1])

        latitude = lat_deg + lat_minute/60
        longitude = lon_deg + lon_minute/60

        "Calculate total Altitude"
        alt1 = float(line[9])
        alt2 = float(line[11])
        altitude = alt1 + alt2

        time_list.append(time)
        lat_list.append(latitude)
        lon_list.append(longitude)
        alt_list.append(altitude)

    return time_list,lat_list,lon_list,alt_list


"Converts LLA (Lat, Long, Alt) to ECEF (Earth Centered, Earth Fixed) coordinates"
def LLA2ECEF(time,lat,long,alt):

    var1 = 6378137.0
    var2 = 0.00335281066474748 #Inverse flattening
    length_data = len(long)
    ecef = list()
    e_squared = (2.0 - var2) * var2

    for i in range(0,length_data):

        N = var1 / np.sqrt(1-e_squared*(np.sin(lat[i]))**2)
        x = (N + lat[i])*np.cos(lat[i])*np.cos(long[i])
        y = (N + lat[i])*np.cos(lat[i])*np.sin(long[i])
        z = (N*(1-e_squared) + alt[i])*np.sin(lat[i])

        txyz_coord = {
                'time':time[i],
                'x':x,
                'y':y,
                'z':z
                }

        ecef.append(txyz_coord)

    return ecef


"Converts LLA (Lat, Long, Alt) to ENU (East, North, Up) coordinates"
def LLA2ENU(time,lat,long,alt):

    var1 = 6378137.0
    var2 = 0.00335281066474748 #Inverse flattening
    length_data = len(long)
    enu = list()
    e_squared = (2.0 - var2) * var2

    a = var1;
    b = 6356752.3142;
    f = 1 - (b/a);
    e2 = (2-f)*f;
    ep  = math.sqrt((a**2-b**2)/(b**2))

    for i in range(0,length_data):

        N = var1 / np.sqrt(1-e_squared*(np.sin(lat[i]))**2)
        x = (N + lat[i])*np.cos(lat[i])*np.cos(long[i])
        y = (N + lat[i])*np.cos(lat[i])*np.sin(long[i])
        z = (N*(1-e_squared) + alt[i])*np.sin(lat[i])

        p = math.sqrt(x**2+y**2)
        th = math.atan2(a*z,b*p)
        lon = math.atan2(y,x)
        lat = math.atan2((z+(ep**2)*b*math.sin(th)**3),(p-e2*a*math.cos(th)**3))

        e = -math.sin(lat)*x + math.cos(lat)*y
        n = -math.cos(lat)*math.sin(lon)*x - math.sin(lat)*math.cos(lon)*y + math.cos(lon)*z
        u = math.cos(lat)*math.cos(lon)*x + math.sin(lat)*math.cos(lon)*y + math.sin(lon)*z

        tenu_coord = {
                'time':time[i],
                'e':e,
                'n':n,
                'u':u
                }

        enu.append(tenu_coord)

    return enu


"Converts ECEF coordinates to LLA coordinates"
def ECEF2LLA(master):

    a = 6378137.0;
    b = 6356752.3142;
    f = 1 - (b/a);
    e2 = (2-f)*f;
    ep  = math.sqrt((a**2-b**2)/(b**2))

    x = master['x']
    y = master['y']
    z = master['z']

    p = math.sqrt(x**2+y**2)
    th = math.atan2(a*z,b*p)
    lon = math.atan2(y,x)
    lat = math.atan2((z+(ep**2)*b*math.sin(th)**3),(p-e2*a*math.cos(th)**3))

    lat_master = lat*180/math.pi
    lon_master = lon*180/math.pi

    return lat_master,lon_master


"Converts ECEF coordinates to ENU coordinates in the prediction and correction"
def ECEF2ENU(input_data):

    a = 6378137.0;
    b = 6356752.3142;
    f = 1 - (b/a);
    e2 = (2-f)*f;
    ep  = math.sqrt((a**2-b**2)/(b**2))

    prediction_x = input_data['ekf_info']['prediction_state']['x']
    prediction_y = input_data['ekf_info']['prediction_state']['y']
    prediction_z = input_data['ekf_info']['prediction_state']['z']

    pred_p = math.sqrt(prediction_x**2+prediction_y**2)
    pred_th = math.atan2(a*prediction_z,b*pred_p)
    pred_lon = math.atan2(prediction_y,prediction_x)
    pred_lat = math.atan2((prediction_z+(ep**2)*b*math.sin(pred_th)**3),(pred_p-e2*a*math.cos(pred_th)**3))

    prediction_E = -math.sin(pred_lat)*prediction_x + math.cos(pred_lat)*prediction_y
    prediction_N = -math.cos(pred_lat)*math.sin(pred_lon)*prediction_x - math.sin(pred_lat)*math.cos(pred_lon)*prediction_y + math.cos(pred_lon)*prediction_z
    prediction_U = math.cos(pred_lat)*math.cos(pred_lon)*prediction_x + math.sin(pred_lat)*math.cos(pred_lon)*prediction_y + math.sin(pred_lon)*prediction_z

    correction_x = input_data['ekf_info']['correction_state']['x']
    correction_y = input_data['ekf_info']['correction_state']['y']
    correction_z = input_data['ekf_info']['correction_state']['z']

    corr_p = math.sqrt(prediction_x**2+prediction_y**2)
    corr_th = math.atan2(a*prediction_z,b*corr_p)
    corr_lon = math.atan2(prediction_y,prediction_x)
    corr_lat = math.atan2((prediction_z+(ep**2)*b*math.sin(corr_th)**3),(corr_p-e2*a*math.cos(corr_th)**3))

    correction_E = -math.sin(corr_lat)*correction_x + math.cos(corr_lat)*correction_y
    correction_N = -math.cos(corr_lat)*math.sin(corr_lon)*correction_x - math.sin(corr_lat)*math.cos(corr_lon)*correction_y + math.cos(corr_lon)*correction_z
    correction_U = math.cos(corr_lat)*math.cos(corr_lon)*correction_x + math.sin(corr_lat)*math.cos(corr_lon)*correction_y + math.sin(corr_lon)*correction_z

    innovation_x = input_data['ekf_info']['innovation']['x']
    innovation_y = input_data['ekf_info']['innovation']['y']
    innovation_z = input_data['ekf_info']['innovation']['z']

    inno_p = math.sqrt(innovation_x**2+innovation_y**2)
    inno_th = math.atan2(a*innovation_z,b*inno_p)
    inno_lon = math.atan2(innovation_y,innovation_x)
    inno_lat = math.atan2((innovation_z+(ep**2)*b*math.sin(inno_th)**3),(inno_p-e2*a*math.cos(inno_th)**3))

    innovation_E = -math.sin(inno_lat)*innovation_x + math.cos(inno_lat)*innovation_y
    innovation_N = -math.cos(inno_lat)*math.sin(inno_lon)*innovation_x - math.sin(inno_lat)*math.cos(inno_lon)*innovation_y + math.cos(inno_lon)*innovation_z
    innovation_U = math.cos(inno_lat)*math.cos(inno_lon)*innovation_x + math.sin(inno_lat)*math.cos(inno_lon)*innovation_y + math.sin(inno_lon)*innovation_z


    return prediction_E,prediction_N,prediction_U,correction_E,correction_N,correction_U,innovation_E,innovation_N,innovation_U


"Converts ECEF coordinates to ENU coordinates in the master's position"
def ECEF2ENUMaster(master):

    a = 6378137.0;
    b = 6356752.3142;
    f = 1 - (b/a);
    e2 = (2-f)*f;
    ep  = math.sqrt((a**2-b**2)/(b**2))

    x = master['x']
    y = master['y']
    z = master['z']

    p = math.sqrt(x**2+y**2)
    th = math.atan2(a*z,b*p)
    lon = math.atan2(y,x)
    lat = math.atan2((z+(ep**2)*b*math.sin(th)**3),(p-e2*a*math.cos(th)**3))

    e_master = -math.sin(lat)*x + math.cos(lat)*y
    n_master = -math.cos(lat)*math.sin(lon)*x - math.sin(lat)*math.cos(lon)*y + math.cos(lon)*z
    u_master = math.cos(lat)*math.cos(lon)*x + math.sin(lat)*math.cos(lon)*y + math.sin(lon)*z

    tenu_coord = {
        'e':e_master,
        'n':n_master,
        'u':u_master
            }

    return tenu_coord


"Calculate rotational matrix to transform ENU coordinates to ECEF"
def calculateR(lat,lon):

    sinphi = math.sin(lat)
    sinlambda = math.sin(lon)
    cosphi = math.cos(lat)
    coslambda = math.cos(lon)

    R = [[-sinlambda,-sinphi*coslambda,cosphi*coslambda],[coslambda,-sinphi*sinlambda,cosphi*sinlambda],[0,cosphi,sinphi]]

    return R


"Finds the closest time in the truth location time vector to the EKF_INFO reference time"
def findNearestLocation1(reference_time,time_vector,offset):

    diff_prev = 10000000
    idx = 0
    reference_time = reference_time.split(':')

    hour_ref = float(reference_time[0])
    minute_ref = float(reference_time[1])
    second_ref = float(reference_time[2])
    total_seconds_ref = hour_ref*3600 + minute_ref*60 + second_ref
    errors_direct = 0

    for i in range(0,len(time_vector)):

        if len(reference_time) == 4:

            time_vec = time_vector[i].split(':')
            hour_vec = float(time_vec[0])
            minute_vec = float(time_vec[1])
            second_vec = float(str(time_vec[2]+"."+time_vec[3]))
            total_seconds_vec = hour_vec*3600 + minute_vec*60 + second_vec + offset

            diff = abs(total_seconds_ref-total_seconds_vec)

            if(diff < diff_prev):

                diff_prev = diff
                idx = i
                save_time_vec = time_vec

        else:

            errors_direct = 1
            time_vec = time_vector[i].split(':')
            hour_vec = float(time_vec[0])
            minute_vec = float(time_vec[1])
            second_vec = float(str(time_vec[2]))
            total_seconds_vec = hour_vec*3600 + minute_vec*60 + second_vec + offset

            diff = abs(total_seconds_ref-total_seconds_vec)

            if(diff < diff_prev):

                diff_prev = diff
                idx = i

    return idx,errors_direct


"Finds the closest time in the truth location time vector to the EKF_INFO reference time"
def findNearestLocation1Interpol(reference_time,time_vector,offset):

    idx_inf = 0
    idx_sup = 0
    total_seconds_ref = timeSeconds(reference_time)

    for i in range(0,len(time_vector)-1):

        total_seconds_inf = timeSeconds(time_vector[i]) + offset
        total_seconds_sup = timeSeconds(time_vector[i+1]) + offset

        if total_seconds_ref >= total_seconds_inf and total_seconds_ref <= total_seconds_sup:

            idx_inf = i
            idx_sup = i+1
            break

    return idx_inf,idx_sup

"Finds the closest time in the truth location time vector to the EKF_INFO reference time"
def findNearestLocation2(reference_time,time_vector,offset):

    diff_prev = 10000000
    idx = 0
    reference_time = reference_time.split(':')

    #Time separated in components. We need hours and minutes and seconds to be the same
    hour_ref = float(reference_time[0])
    minute_ref = float(reference_time[1])
    second_ref = float(reference_time[2])
    total_seconds_ref = hour_ref*3600 + minute_ref*60 + second_ref

    for i in range(0,len(time_vector)):

        time_vec = time_vector[i].split(':')
        hour_vec = float(time_vec[0])
        minute_vec = float(time_vec[1])
        second_vec = float(time_vec[2])
        total_seconds_vec = hour_vec*3600 + minute_vec*60 + second_vec + offset

        diff = abs(total_seconds_ref-total_seconds_vec)

        if(diff < diff_prev):

            diff_prev = diff
            idx = i
            save_time_vec = time_vec

    return idx,save_time_vec


"Finds the closest time in the truth location time vector to the EKF_INFO reference time"
def findNearestLocation2Interpol(reference_time,time_vector,offset):

    idx_inf = 0
    idx_sup = 0
    total_seconds_ref = timeSeconds(reference_time)

    for i in range(0,len(time_vector)-1):

        total_seconds_inf = timeSeconds(time_vector[i]) + offset
        total_seconds_sup = timeSeconds(time_vector[i+1]) + offset

        if total_seconds_ref >= total_seconds_inf and total_seconds_ref <= total_seconds_sup:

            idx_inf = i
            idx_sup = i+1

    return idx_inf,idx_sup

"Gets the final ECEF location for ENU coordinates case"
def getTrueLocationENU2ECEF(time,data_row_area_info_master,e,n,u,R):

    ecef = list()

    ECEF_rover = np.dot(R,[[e],[n],[u]]) #data_row_area_info_master"

    true_xyz ={
        'time':time,
        'x' : ECEF_rover[0][0] + data_row_area_info_master['x'],
        'y' : ECEF_rover[1][0] + data_row_area_info_master['y'],
        'z' : ECEF_rover[2][0] + data_row_area_info_master['z']
        }

    ecef.append(true_xyz)

    return ecef


"Gets the final ENU location for ENU coordinates case"
def getTrueLocationENU2ENU(time,data_row_area_info_master,e,n,u):

    enu = list()

    true_enu ={
        'time':time,
        'e' : e + data_row_area_info_master['e'],
        'n' : n + data_row_area_info_master['n'],
        'u' : u + data_row_area_info_master['u']
        }

    enu.append(true_enu)

    return enu


"Gets the final ENU location for ENU coordinates case"
def getTrueLocationENU2ENUInterpol(reference_time,tru_time,e,n,u,idx_inf,idx_sup):

    enu = list()

    ENU_rover_time_inf = timeSeconds(tru_time[idx_inf])
    ENU_rover_time_sup = timeSeconds(tru_time[idx_sup])

    ENU_rover_inf_e = e[idx_inf]
    ENU_rover_inf_n = n[idx_inf]
    ENU_rover_inf_u = u[idx_inf]

    ENU_rover_sup_e = e[idx_sup]
    ENU_rover_sup_n = n[idx_sup]
    ENU_rover_sup_u = u[idx_sup]

    delta_time = ENU_rover_time_sup - ENU_rover_time_inf
    ref_time = timeSeconds(reference_time)

    if delta_time == 0:

        e_interpol = np.average([ENU_rover_inf_e,ENU_rover_sup_e])
        n_interpol = np.average([ENU_rover_inf_n,ENU_rover_sup_n])
        u_interpol = np.average([ENU_rover_inf_u,ENU_rover_sup_u])

    else:

        delta_e = ENU_rover_sup_e - ENU_rover_inf_e
        delta_n = ENU_rover_sup_n - ENU_rover_inf_n
        delta_u = ENU_rover_sup_u - ENU_rover_inf_u

        e_interpol = ENU_rover_inf_e + (ref_time-ENU_rover_time_inf)*(delta_e/delta_time)
        n_interpol = ENU_rover_inf_n + (ref_time-ENU_rover_time_inf)*(delta_n/delta_time)
        u_interpol = ENU_rover_inf_u + (ref_time-ENU_rover_time_inf)*(delta_u/delta_time)

    true_enu ={
        'time':reference_time,
        'e' : e_interpol,
        'n' : n_interpol,
        'u' : u_interpol
        }

    enu.append(true_enu)

    return enu


"Gets the final ECEF location for ENU coordinates case"
def getTrueLocationLLA2ECEF(rover_list,idx,master):

    ECEF_rover = rover_list[idx]

    true_xyz ={
        'time':ECEF_rover['time'],
        'x' : ECEF_rover['x'] + master['x'],
        'y' : ECEF_rover['y'] + master['y'],
        'z' : ECEF_rover['z'] + master['z']
        }

    return true_xyz


"Gets the final ENU location for ENU coordinates case"
def getTrueLocationLLA2ENU(rover_list,idx,master):

    ENU_rover = rover_list[idx]

    true_enu ={
        'time':ENU_rover['time'],
        'e' : ENU_rover['e'] + master['e'],
        'n' : ENU_rover['n'] + master['n'],
        'u' : ENU_rover['u'] + master['u']
        }

    return true_enu


"Gets the interpolated final ENU location for ENU coordinates case"

def getTrueLocationLLA2ENUInterpol(rover_list,idx_inf,idx_sup,data_time,master):

    ENU_rover_time_inf = timeSeconds(rover_list[idx_inf]['time'])
    ENU_rover_time_sup = timeSeconds(rover_list[idx_sup]['time'])

    ENU_rover_inf = rover_list[idx_inf]
    ENU_rover_inf_e = ENU_rover_inf['e']
    ENU_rover_inf_n = ENU_rover_inf['n']
    ENU_rover_inf_u = ENU_rover_inf['u']

    ENU_rover_sup = rover_list[idx_sup]
    ENU_rover_sup_e = ENU_rover_sup['e']
    ENU_rover_sup_n = ENU_rover_sup['n']
    ENU_rover_sup_u = ENU_rover_sup['u']

    delta_time = ENU_rover_time_sup - ENU_rover_time_inf
    ref_time = timeSeconds(data_time)

    delta_e = ENU_rover_sup_e - ENU_rover_inf_e
    delta_n = ENU_rover_sup_n - ENU_rover_inf_n
    delta_u = ENU_rover_sup_u - ENU_rover_inf_u

    e_interpol = ENU_rover_inf_e + master['e'] + (ref_time-ENU_rover_time_inf)*(delta_e/delta_time)
    n_interpol = ENU_rover_inf_n + master['e'] + (ref_time-ENU_rover_time_inf)*(delta_n/delta_time)
    u_interpol = ENU_rover_inf_u + master['e'] + (ref_time-ENU_rover_time_inf)*(delta_u/delta_time)

    true_enu ={
        'time':data_time,
        'e' : e_interpol,
        'n' : n_interpol,
        'u' : u_interpol
        }

    return true_enu


"Function that transforms a time vector or time into total seconds"
def timeSeconds(time_vector):

    if len(time_vector) == 4:

        time_vec = time_vector.split(':')
        hour_vector = float(time_vec[0])
        minute_vector = float(time_vec[1])
        second_vector = float(str(time_vec[2]+"."+time_vec[3]))
        total_seconds_vector = hour_vector*3600 + minute_vector*60 + second_vector

    else:

        time_vec = time_vector.split(':')
        hour_vector = float(time_vec[0])
        minute_vector = float(time_vec[1])
        second_vector = float(time_vec[2])
        total_seconds_vector = hour_vector*3600 + minute_vector*60 + second_vector

    return total_seconds_vector



"Calculates location error between prediction and true location"
def calculateLocationError(ECEF_true,ECEF_pred):

    ecef = ECEF_true[0]
    error_xyz = math.sqrt((ecef['x']-ECEF_pred['x'])**2+(ecef['y']-ECEF_pred['y'])**2+(ecef['z']-ECEF_pred['z'])**2)

    return error_xyz


"Calculates location error between prediction and true location"
def calculateLocationErrorENU(ENU_true,ENU_pred):

    enu = ENU_true[0]
    error_enu = math.sqrt((enu['e']-ENU_pred['e'])**2+(enu['n']-ENU_pred['n'])**2+(enu['u']-ENU_pred['u'])**2)

    return error_enu


"Calculates location error between prediction and true location"
def calculateLocationErrorENUInterpol(ENU_true):

    error_enu = math.sqrt(ENU_true[0]['e']**2+ENU_true[0]['n']**2+ENU_true[0]['u']**2)

    return error_enu


"Change adress and sky_clearance_file names for proper file names corresponding to sky clearance files"
"Add more elif conditionals corresponding to all files wanted to be read"
def getInputData(data,string):

    "Import sky obstruction from file"

    if string == 'file1':

        sky_input = readSkyFile('address/sky_clearance_file1.json')

    elif string == 'file2':

        sky_input = readSkyFile('address/sky_clearance_file2.json')

    elif string == 'file3':

        sky_input = readSkyFile('address/sky_clearance_file3.json')

    elif string == 'file4':

        sky_input = readSkyFile('address/sky_clearance_file4.json')


    "List to temporarily store information from the satellites information vector"
    satellites_input = list()

    "Final list containing all information required by the machine learning algorithms"
    full_input = list()

    for k in range(0,len(data)):

        data_row = data[k]
        data_row_id = data_row['id']

        if data_row_id == "ekf_info":

            "data_row_content is a list with rows and columns for each satellite and its corresponding info respectively"
            data_row_ekf = data_row['ekf_info']
            data_row_ekf_time = data_row['time']

            if data_row_ekf_time == "00:00:00.000":

                0

            else:

                data_row_area = data[k-1]
                data_row_area_id = data_row_area['id']

                if (data_row_area_id == "ekf_info") or (data_row_area_id == "satellite_info"):

                    data_row_area = data[k-2]
                    data_row_area_id = data_row_area['id']

                if (data_row_area_id == "ekf_info") or (data_row_area_id == "satellite_info"):

                    data_row_area = data[k-3]
                    data_row_area_id = data_row_area['id']

                data_row_area_info = data_row_area['area_info']

                data_row_area_used_satellites = data_row_area_info['usable_sat_list']
                data_row_area_lsq_residuals = data_row_area_info['lsq_residuals']

                data_row_area_used_satellites_number = len(data_row_area_used_satellites)
                data_row_area_lsq_residuals_number = len(data_row_area_lsq_residuals)

                satellites_used_list,satellites_used_all_info = makeListOfSatellites(data_row_area_used_satellites,data_row_area_used_satellites_number,data_row_area_lsq_residuals,data_row_area_lsq_residuals_number)

                "Adds information from the satellites of interest only. This consecutive verification is due to an error found in one of the files. May be reduced."
                data_row2 = data[k-2]
                data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-3]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-4]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-5]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-6]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-7]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-8]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-9]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-10]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-11]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-12]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-13]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-14]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-15]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-16]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-17]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-18]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-19]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-20]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-21]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-22]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-23]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-24]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-25]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-26]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-27]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-29]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-30]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-31]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-32]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-33]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-34]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-35]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-36]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-37]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-38]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-39]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-40]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-41]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-42]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-43]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-44]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-45]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-46]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-47]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-48]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-49]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-50]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-51]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-52]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-53]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-54]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-55]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-56]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-57]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-58]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-59]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-60]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-61]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-62]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-63]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-64]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-65]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-66]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-67]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-68]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-69]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-70]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-71]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-72]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-73]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-74]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-75]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-76]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-77]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-78]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-79]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-80]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-81]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-82]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-83]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-84]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-85]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-86]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-87]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-88]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-89]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-90]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-91]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-92]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-93]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-94]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-95]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-96]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-97]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-98]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-99]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-100]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-101]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-102]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-103]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-104]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-105]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-106]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-107]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-108]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-109]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-110]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-111]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-112]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-113]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-114]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-115]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-116]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-117]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-118]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-119]
                    data_row2_id = data_row2['id']

                if (data_row2_id == "area_info") or (data_row2_id == "ekf_info"):

                    data_row2 = data[k-120]
                    data_row2_id = data_row2['id']

                data_row2_sat = data_row2['satellite_info']
                data_row2_sat_number = len(data_row2_sat)

                for m in range(0,data_row2_sat_number):

                    data_row2_sat_individual = data_row2_sat[m]
                    data_row2_sat_individual_id = data_row2_sat_individual['satellite_id']
                    data_row2_sat_individual_const = data_row2_sat_individual['constellation']
                    data_row2_sat_individual_id_const = str(data_row2_sat_individual_id)+str(data_row2_sat_individual_const)

                    if checkSatelliteUsed(satellites_used_list,data_row2_sat_individual_id_const):

                        satellite_id_const_info = getUsedSatellite(satellites_used_all_info,data_row2_sat_individual_id_const)

                        sat_sample = {

                                'satellite_id': data_row2_sat_individual_id,
                                'constellation': data_row2_sat_individual['constellation'],
                                'tracking_type': data_row2_sat_individual['tracking_type'],
                                'elevation': data_row2_sat_individual['elevation'],
                                'azimuth': data_row2_sat_individual['azimuth'],
                                'cno': data_row2_sat_individual['cno'],
                                'lsq_residual': satellite_id_const_info['lsq_residual'],
                                'multipath': satellite_id_const_info['multipath'],
                                'raim': satellite_id_const_info['raim'],
                                'cycle_slip': satellite_id_const_info['cycle_slip'],
                                'amb_type': satellite_id_const_info['amb_type']

                                }

                        satellites_input.append(sat_sample)

                sky = findNearestSky(sky_input,data_row_ekf_time)

                full_sample = {

                        'time': data_row_ekf_time, #Using the time stamp from the 'ekf_info'
                        'nr_used_measurements': data_row_area_info['nr_used_measurements'],
                        'pdop': data_row_area_info['pdop'],
                        'ndop': data_row_area_info['ndop'],
                        'used_satellites': satellites_input,
                        'ekf_info': data_row_ekf,
                        'sky':sky

                        }

                full_input.append(full_sample)

                satellites_input = []

                full_sample = []

    return full_input


"Gets errors from Reference Station or target outputs"
def getOutputData(data,data_tru):

    if data_tru[0] == "e":

        "tru_coor1 = East"
        "tru_coor2 = North"
        "tru_coor3 = Up"
        tru_time,tru_coor1,tru_coor2,tru_coor3 = parseTrimble(data_tru)

    elif data_tru[0] == "E":

        "tru_coor1 = East"
        "tru_coor2 = North"
        "tru_coor3 = Up"
        tru_time,tru_coor1,tru_coor2,tru_coor3 = parseTrimble2(data_tru)

    else:

        "tru_coor1 = Latitude"
        "tru_coor2 = Longitude"
        "tru_coor3 = Altitude"
        tru_time,tru_coor1,tru_coor2,tru_coor3 = parseNMEA(data_tru)

    "Final output list containing all positioning errors for every case"
    full_output = list()

    for k in range(0,len(data)):

        data_row = data[k]
        data_row_id = data_row['id']

        if data_row_id == "ekf_info":

            "data_row_content is a list with rows and columns for each satellite and its corresponding info respectively"
            data_row_ekf_time = data_row['time']

            if data_row_ekf_time == "00:00:00.000":

                0

            else:

                data_row_area = data[k-1]
                data_row_area_id = data_row_area['id']

                if (data_row_area_id == "ekf_info") or (data_row_area_id == "satellite_info"):

                    data_row_area = data[k-2]
                    data_row_area_id = data_row_area['id']

                if (data_row_area_id == "ekf_info") or (data_row_area_id == "satellite_info"):

                    data_row_area = data[k-3]
                    data_row_area_id = data_row_area['id']

                data_row_area_info = data_row_area['area_info']
                data_row_area_info_master = data_row_area_info['master_position']
                data_row_area_info_master_enu = ECEF2ENUMaster(data_row_area_info_master)
                pred_e,pred_n,pred_u,corr_e,corr_n,corr_u,inno_e,inno_n,inno_u = ECEF2ENU(data_row)

                data_row_ekf_pred_enu = {
                        'e':pred_e,
                        'n':pred_n,
                        'u':pred_u
                        }

                if data_tru[1] == '$':

                    index,tru_time_pos = findNearestLocation2(data_row_ekf_time,tru_time,0)
                    index_inf,index_sup = findNearestLocation2Interpol(data_row_ekf_time,tru_time,0)
                    ENU_rover_full_list = LLA2ENU(tru_time,tru_coor1,tru_coor2,tru_coor3)
                    ENU_rover_true_interpol = getTrueLocationLLA2ENUInterpol(ENU_rover_full_list,index_inf,index_sup,data_row_ekf_time,data_row_area_info_master_enu)
                    error_enu_interpol = calculateLocationErrorENU(ENU_rover_true_interpol,data_row_ekf_pred_enu)

                else:

                    index,errors_direct = findNearestLocation1(data_row_ekf_time,tru_time,0)
                    index_inf,index_sup = findNearestLocation1Interpol(data_row_ekf_time,tru_time,0)
                    ENU_rover_true_interpol = getTrueLocationENU2ENUInterpol(data_row_ekf_time,tru_time,tru_coor1,tru_coor2,tru_coor3,index_inf,index_sup)
                    error_enu_interpol = calculateLocationErrorENUInterpol(ENU_rover_true_interpol)

                full_sample_out = {

                        'time': data_row_ekf_time,
                        'error': error_enu_interpol
                        }

                full_output.append(full_sample_out)

                full_sample_out = []

    return full_output


"Change all the names and addresses for the corresponding files that contain the input features and target outputs"
data_input = {
        'inputFeaturesFile1':'address/inputFile1.json',
        'inputFeaturesFile2':'address/inputFile2.json',
        'outputFeaturesFile1':'address/outputFile1.json'
        'outputFeaturesFile2':'address/outputFile2.json'
        }


"Shuffles the read data and creates the partition for training and testing"
def partition (input_data,output_data,test_fraction):

    input_train = list()
    input_test = list()
    output_train = list()
    output_test = list()

    combined = list(zip(input_data, output_data))
    random.shuffle(combined)
    input_data[:], output_data[:] = zip(*combined)

    len_list = len(input_data)
    index_partition = round(test_fraction*len_list)
    input_test = input_data[0:index_partition]
    input_train = input_data[index_partition:]
    output_test = output_data[0:index_partition]
    output_train = output_data[index_partition:]

    return input_train,input_test,output_train,output_test


"Eliminates repeated information in case it exists"
def eliminateRepeats(data):

    seen = list()
    new_data = list()
    repeated = list()
    counter = 0
    for i in data:
        if i not in seen:
            seen.append(i)
            new_data.append(i)
        else:
            counter+=1
            repeated.append(i)

    return new_data,counter,repeated


"Function to obtain difference between prediction and correction in ENU coordinates of the EKF process"
def errorPredictionCorrection(input_data):

    for i in input_data:

        prediction_E,prediction_N,prediction_U,correction_E,correction_N,correction_U,innovation_E,innovation_N,innovation_U = ECEF2ENU(i)

        prediction_state_ENU = {
                'e':prediction_E,
                'n':prediction_N,
                'u':prediction_U
                }

        correction_state_ENU = {
                'e':correction_E,
                'n':correction_N,
                'u':correction_U
                }

        difference_ENU = {
                'e':abs(prediction_E - correction_E),
                'n':abs(prediction_N - correction_N),
                'u':abs(prediction_U - correction_U)
                }

        innovation_ENU = {
                'e':innovation_E,
                'n':innovation_N,
                'u':innovation_U
                }

        i['ekf_info']['prediction_state_ENU'] = prediction_state_ENU
        i['ekf_info']['correction_state_ENU'] = correction_state_ENU
        i['ekf_info']['difference_ENU'] = difference_ENU
        i['ekf_info']['innovation_ENU'] = innovation_ENU

    return input_data


"Obtains the confidence interval to filter out outliers (beyond 98% interval)"
def obtain95ConfidenceInterval(errors):

    std_dev = np.std(errors)
    avg = np.average(errors)
    min_interval = avg - 2*std_dev
    max_interval = avg + 2*std_dev

    return min_interval,max_interval


"Get indices and outliers from output_data, considering also the ones from PDOP and NDOP"
def findOutliers(dataset,errors,min_interval,max_interval):

    outliers = list()
    [outliers.append(i) for i in range(0,len(dataset)) if dataset[i]['pdop'] > 1000 or dataset[i]['pdop'] > 1000 or dataset[i]['ekf_info']['prediction_covariance']['x'] > 100 or dataset[i]['ekf_info']['prediction_covariance']['y'] > 100 or dataset[i]['ekf_info']['prediction_covariance']['z'] > 100]
    [outliers.append(i) for i in range(0,len(errors)) if errors[i] < min_interval or errors[i] > max_interval]
    outliers_list = list(set(outliers))

    return outliers_list


"Eliminate outliers from input and output data, considering also the ones from PDOP and NDOP"
def eliminateOutliers(index,i_data,o_data):

    for ind in sorted(index, reverse = True):

        del i_data[ind]
        del o_data[ind]

    return i_data,o_data


"Attaches Error information to the input data vector to make getErrors method faster"
def attachErrorsToInput(input_data,output):

    for i in range(0,len(input_data)):

        input_data[i]['error'] = output[i]['error']

    return input_data


"Read the input files"
def read(string_input,string_truth,fraction):

    input_data = list()
    output_data = list()
    input_data_iter = list()
    output_data_iter = list()
    errors = list()
    errors_test = list()
    filter_errors_train = list()


    if isinstance(string_input,list):

        for i in range(0,len(string_input)):

            estimate = data_input[string_input[i]]
            truth = data_input[string_truth[i]]

            "Get input data (GNSS data details)"
            with open(estimate) as est:
                data = json.load(est)

            input_data_iter = getInputData(data,string_input[i])
            input_data_iter,counter_input,seen_input = eliminateRepeats(input_data_iter)

            "Get output data (Position Errors)"
            f = open(truth,"r")
            data_tru = f.read()
            output_data_iter = getOutputData(data,data_tru)
            f.close()

            output_data_iter,counter_output,seen_output = eliminateRepeats(output_data_iter)

            input_data.extend(input_data_iter[:])
            output_data.extend(output_data_iter[:])

            input_data_iter = []
            output_data_iter = []

            print("Read: "+str(i)+" "+"file: "+str(string_input[i]))

    else:

        estimate = data_input[string_input]
        truth = data_input[string_truth]
        errors = list()

        "Get input data (GNSS data details)"
        with open(estimate) as est:
            data = json.load(est)

        input_data = getInputData(data,string_input)
        input_data,counter_input,seen_input = eliminateRepeats(input_data)

        "Get output data (Position Errors)"
        f = open(truth,"r")
        data_tru = f.read()
        output_data = getOutputData(data,data_tru)
        f.close()

        output_data,counter_output,seen_output = eliminateRepeats(output_data)

    print("Done reading!")

    "Transforms the EKF prediction and correction data into its difference in plane coordinates"
    input_data = errorPredictionCorrection(input_data)

    "Separates the data into training and testing data randomly"
    input_train,input_test,output_train,output_test = partition(input_data,output_data,fraction)

    "Automatically calculates errors only from the output vector"
    [errors.append(i['error']) for i in output_train]
    [errors_test.append(i['error']) for i in output_test]

    "Cleans out the information of the input and output data set"
    min_interval,max_interval = obtain95ConfidenceInterval(errors)

    "Obtain the outliers and the amount of them to be filtered out"
    outliers_index = findOutliers(input_train,errors,min_interval,max_interval)
    outliers_test_index = findOutliers(input_test,errors_test,min_interval,max_interval)

    "Obtain the vector of input and output data with already filtered values"
    filter_input_train,filter_output_train = eliminateOutliers(outliers_index,input_train[:],output_train[:])
    filter_input_test,filter_output_test = eliminateOutliers(outliers_test_index,input_test[:],output_test[:])

    "Attaches errors to the input data vector"
    start_errors = time.time()
    filter_input_train = attachErrorsToInput(filter_input_train,filter_output_train)
    filter_input_test = attachErrorsToInput(filter_input_test,filter_output_test)
    end_errors = time.time()
    diff_errors = end_errors - start_errors
    print("Finished tree building execution time (s): "+str(diff_errors))

    "Obtain the list of filtered errors for training"
    [filter_errors_train.append(i['error']) for i in filter_output_train]

    "Returns input and output data in manageable format"
    return filter_input_train,filter_output_train,filter_errors_train,filter_input_test,filter_output_test
