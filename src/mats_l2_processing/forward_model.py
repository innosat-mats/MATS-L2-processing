# %%
import numpy as np
import pandas as pd
from mats_utils.geolocation.coordinates import col_heights, findheight
from mats_l1_processing.pointing import pix_deg
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline
import scipy.sparse as sp
from skyfield import api as sfapi
from skyfield.framelib import itrs
import pickle
from mats_l2_processing.grids import cart2sph
from fast_histogram import histogramdd

# %%

# %%
def generate_timescale():
    global timescale 
    timescale = sfapi.load.timescale()
    return

# %%

def prepare_profile(ch,col=None,rows=None):
    """Extract data and tanalt for image

    Detailed description

    Args:
        single column
        range of rows

    Returns: 
        profile, tanalts
    """
    image = np.stack(ch.ImageCalibrated)
    if col == None:
        col = int(ch['NCOL']/2)
    if rows.any() == None:
        rows = np.arrange(0,ch['NROW'])

    cs = col_heights(ch, col, 10, spline=True)
    tanalt = np.array(cs(rows))
    profile = np.array(image[rows, col])*1e15
    return profile,tanalt

def eci_to_ecef_transform(date):
    return R.from_matrix(itrs.rotation_at(timescale.from_datetime(date)))

def select_data(df, num_profiles, start_index = 0):
    if num_profiles + start_index > len(df):
        raise ValueError('Selected profiles out of bounds')
    df = df.loc[start_index:start_index+num_profiles-1]
    df = df.reset_index(drop=True)

def find_top_of_atmosphere(top_altitude,localR,ecipos,ecivec):
    sat_radial_pos = np.linalg.norm(ecipos)
    los_zenith_angle = np.arccos(np.dot(ecipos,ecivec)/sat_radial_pos)
    #solving quadratic equation to find distance start and end of atmosphere 
    b = 2*sat_radial_pos*np.cos(los_zenith_angle)
    root=np.sqrt(b**2+4*((top_altitude+localR)**2 - sat_radial_pos**2))
    distance_top_1 =(-b-root)/2
    distance_top_2 =(-b+root)/2
    
    return [distance_top_1,distance_top_2]

def generate_steps(stepsize,top_altitude,localR,ecipos,ecivec):
    distance_top_of_atmosphere = find_top_of_atmosphere(top_altitude,localR,ecipos,ecivec)
    steps = np.arange(distance_top_of_atmosphere[0], distance_top_of_atmosphere[1], stepsize)
    return steps

def generate_local_transform(df):
    """Calculate the transform from ecef to the local coordinate system.

    Detailed description

    Args:
        df (pandas dataframe): data.

    Returns: 
        ecef_to_local
    """
    first = 0
    mid = int((len(df)-1)/2)
    last = len(df)-1

    
    eci_to_ecef = eci_to_ecef_transform(df['EXPDate'][mid])

    posecef_first = eci_to_ecef.apply(df.afsTangentPointECI[first]).astype('float32')
    posecef_mid = eci_to_ecef.apply(df.afsTangentPointECI[mid]).astype('float32')
    posecef_last = eci_to_ecef.apply(df.afsTangentPointECI[last]).astype('float32')
    
    observation_normal = np.cross(posecef_first,posecef_last)
    observation_normal = observation_normal/np.linalg.norm(observation_normal) #normalize vector

    posecef_mid_unit = posecef_mid/np.linalg.norm(posecef_mid) #unit vector for central position
    ecef_to_local = R.align_vectors([[1,0,0],[0,1,0]],[posecef_mid_unit,observation_normal])[0]

    return ecef_to_local

def get_los_in_local_grid(df_row,icol,irow,stepsize,top_altitude, ecef_to_local, localR=None):

    rot_sat_channel = R.from_quat(df_row['qprime']) #Rotation between satellite pointing and channel pointing
    satpos_eci = df_row['afsGnssStateJ2000'][0:3] 
    date = df_row['EXPDate']
    current_ts = timescale.from_datetime(date)
    q = df_row['afsAttitudeState'] #Satellite pointing in ECI
    rot_sat = R.from_quat(np.roll(q, -1)) #Rotation matrix for q
    
    x, y = pix_deg(df_row, icol, irow) #Angles for single pixel
    rotation_channel_pix = R.from_euler('xyz', [0, y, x], degrees=True).apply([1, 0, 0]) #Rot matrix between pixel pointing and channels pointing
    los_eci = np.array(rot_sat.apply(rot_sat_channel.apply(rotation_channel_pix)))
    if localR == None:
        localR = np.linalg.norm(sfapi.wgs84.latlon(df_row.TPlat, df_row.TPlon, elevation_m=0).at(current_ts).position.m)        

    s_steps = generate_steps(stepsize,top_altitude=top_altitude,localR=localR,ecipos=satpos_eci,ecivec=los_eci)

    eci_to_ecef=R.from_matrix(itrs.rotation_at(current_ts))
    pos=np.expand_dims(satpos_eci, axis=0).T+s_steps*np.expand_dims(los_eci, axis=0).T
    posecef=(eci_to_ecef.apply(pos.T).astype('float32'))
    poslocal = ecef_to_local.apply(posecef) #convert to local
    poslocal_sph = cart2sph(poslocal)   #x: height, y: acrosstrac (angle), z: along track (angle)
    poslocal_sph = np.array(poslocal_sph).T

    return poslocal_sph

def generate_grid(df, ecef_to_local):

    first = 0
    mid = int((len(df)-1)/2)
    last = len(df)-1

    stepsize = 100 #should be global?
    top_alt = 120e3  #should be global?


    satpos_eci = df['afsGnssStateJ2000'][mid][0:3]
    mid_date = df['EXPDate'][mid]
    current_ts = timescale.from_datetime(mid_date)
    localR = np.linalg.norm(sfapi.wgs84.latlon(df.TPlat[mid], df.TPlon[mid], elevation_m=0).at(current_ts).position.m)
    
    #change to do only used columns and rows
    left_col = 0
    mid_col = int(df["NCOL"][0]/2) -1
    right_col =df["NCOL"][0] - 1
    irow = 0
    
    poslocal_sph = []
    poslocal_sph.append(get_los_in_local_grid(df.loc[first],left_col,irow,stepsize,top_alt,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[first],right_col,irow,stepsize,top_alt,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[first],mid_col,irow,stepsize,top_alt,ecef_to_local))

    poslocal_sph.append(get_los_in_local_grid(df.loc[mid],left_col,irow,stepsize,top_alt,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[mid],right_col,irow,stepsize,top_alt,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[mid],mid_col,irow,stepsize,top_alt,ecef_to_local))

    poslocal_sph.append(get_los_in_local_grid(df.loc[last],left_col,irow,stepsize,top_alt,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[last],right_col,irow,stepsize,top_alt,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[last],mid_col,irow,stepsize,top_alt,ecef_to_local))

    poslocal_sph = np.vstack(poslocal_sph)
    max_alt = poslocal_sph[:,0].max()
    min_alt = poslocal_sph[:,0].min()
    max_lat = poslocal_sph[:,2].max()
    min_lat = poslocal_sph[:,2].min()
    max_lon = poslocal_sph[:,1].max()
    min_lon = poslocal_sph[:,1].min()

    nalt = 50
    nlon = 10
    nlat = 25

    # altitude_grid = np.arange(localR+50e3,localR+130e3,2e3)
    # altitude_grid[0] = altitude_grid[0]-30e3
    # altitude_grid[-1] = altitude_grid[-1]+30e3
    # acrosstrack_grid = np.arange(-0.1,0.1,0.01)
    # #acrosstrack_grid = np.linspace(posecef_i_sph[:,1].min(),posecef_i_sph[:,1].max(),1)
    # alongtrack_grid = np.linspace(poslocal_sph[:,2].min()-0.5,poslocal_sph[:,2].max()+0.5,50)
    # alongtrack_grid[0] = alongtrack_grid[0]-0.5
    # alongtrack_grid[-1] = alongtrack_grid[-1]+0.5

    altitude_grid = np.linspace(min_alt-5e3,localR+top_alt+5e3,nalt)
    acrosstrack_grid = np.linspace(min_lon,max_lon,nlon)
    alongtrack_grid = np.linspace(min_lat,max_lat,nlat)

    return altitude_grid,acrosstrack_grid,alongtrack_grid

def calc_jacobian(df,columns,rows,edges=None):
    # -*- coding: utf-8 -*-
    """Calculate Jacobian.

    Detailed description

    Args:
        df (pandas dataframe): data.
        coluns (array): columns to use.
        rows: rows to use

    Returns:
        y, 
        K, 
        altitude_grid, (change to edges)
        alongtrack_grid,
        acrosstrack_grid, 
        ecef_to_local
    """
    stepsize = 100
    top_alt = 120e3
    profiles = []
    generate_timescale() #generates a global timescale object

    ecef_to_local = generate_local_transform(df)
    if edges == None:
        altitude_grid,acrosstrack_grid,alongtrack_grid = generate_grid(df, ecef_to_local)
        edges = [altitude_grid,acrosstrack_grid,alongtrack_grid]

    #Add check that all images are same format
    #change to edges
    K = sp.lil_array((len(rows)*len(columns)*len(df),(len(altitude_grid)-1)*(len(alongtrack_grid)-1)*(len(acrosstrack_grid)-1)))
    k_row = 0
    for i in range(len(df)):
        print("image number: " + str(i))
        current_ts = timescale.from_datetime(df['EXPDate'][i])
        localR = np.linalg.norm(sfapi.wgs84.latlon(df.loc[i].TPlat, df.loc[i].TPlon, elevation_m=0).at(current_ts).position.m)        
        for column in columns:
            profile,tanalts = prepare_profile(df.iloc[i],column,rows)
            profiles.append(profile)
            for irow in rows:
                posecef_i_sph = get_los_in_local_grid(df.loc[i],column,irow,stepsize,top_alt,ecef_to_local,localR=localR)
                hist = histogramdd(posecef_i_sph[::1,:],range=[[edges[0][0],edges[0][-1]],[edges[1][0],edges[1][-1]],
                    [edges[2][0],edges[2][-1]]], bins=[len(edges[0])-1,len(edges[1])-1,len(edges[2])-1])
                k = hist.reshape(-1)
                K[k_row,:] = k
                k_row = k_row+1

    y = np.array(profiles)
 
    return  y, K, altitude_grid, alongtrack_grid,acrosstrack_grid, ecef_to_local



# dftop = pd.read_pickle('/home/olemar/Projects/Universitetet/MATS/MATS-analysis/Donal/retrievals/verdec2d.pickle')
#%%
# starttime=datetime(2023,3,31,21,0)
# stoptime=datetime(2023,3,31,22,35)
# dftop=read_MATS_data(starttime,stoptime,level="1b",version="0.4")

# with open('verdec2d.pickle', 'wb') as handle:
#     pickle.dump(dftop, handle)
# #%%
# # df=df[df['channel']!='NADIR']
# df = dftop[dftop['channel'] == 'IR2'].dropna().reset_index(drop=True)#[0:10]


# #select part of orbit
# offset = 300
# num_profiles = 10 #use 50 profiles for inversion
# df = df.loc[offset:offset+num_profiles-1]
# df = df.reset_index(drop=True)
# columns = np.arange(1,df["NCOL"][0],5)
# rows = np.arange(1,df["NROW"][0],2)

# y, ks, altitude_grid, alongtrack_grid,acrosstrack_grid, ecef_to_local = calc_jacobian(df,columns,rows)
# filename = "jacobian_3.pkl"

# with open(filename, "wb") as file:
#     pickle.dump((y, ks, altitude_grid, alongtrack_grid,acrosstrack_grid, ecef_to_local), file)

# with open(filename, "rb") as file:
#     [y, ks, altitude_grid, alongtrack_grid,acrosstrack_grid, ecef_to_local] = pickle.load(file)
