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

# %%

# %%
# %%


def prepare_profile(ch,col=None,row=None):
    image = np.stack(ch.ImageCalibrated)
    if col == None:
        col = int(ch['NCOL']/2)
    if row.any() == None:
        row = np.arrange(0,ch['NROW'])

#    cs = col_heights(ch, col, 10, spline=True)
#    heights = np.array(cs(range(ch['NROW']-10)))
    profile = np.array(image[row, col])*1e15
    return profile

def eci_to_ecef_transform(timescale,date):
    return R.from_matrix(itrs.rotation_at(timescale.from_datetime(date)))

def cart2sph(pos):
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    longitude = np.arctan2(y, x)
    latitude = np.arcsin(z / radius)

    return np.array([radius,longitude,latitude]).T


def select_data(df, num_profiles, start_index = 0):
    if num_profiles + start_index > len(df):
        raise ValueError('Selected profiles out of bounds')
    df = df.loc[start_index:start_index+num_profiles-1]
    df = df.reset_index(drop=True)

def find_top_of_atmosphere(top_altitude,date,ecipos,ecivec):
    s_120_2=findheight(date,ecipos,ecivec,top_altitude,bracket=(30e5,40e5)).x
    return s_120_2

def generate_steps(stepsize,top_altitude=None,date=None,ecipos=None,ecivec=None):
    #distance_top_of_atmosphere = find_top_of_atmosphere(top_altitude,date,ecipos,ecivec)
    distance_top_of_atmosphere = np.array([1800e3,1800e3 + 2e6-4e5])
    steps = np.arange(distance_top_of_atmosphere[0], distance_top_of_atmosphere[1], stepsize)
    return steps

def generate_local_transform(df,timescale):
    first = 0
    mid = int((len(df)-1)/2)
    last = len(df)-1

    
    eci_to_ecef = eci_to_ecef_transform(timescale,df['EXPDate'][mid])

    posecef_first = eci_to_ecef.apply(df.afsTangentPointECI[first]).astype('float32')
    posecef_mid = eci_to_ecef.apply(df.afsTangentPointECI[mid]).astype('float32')
    posecef_last = eci_to_ecef.apply(df.afsTangentPointECI[last]).astype('float32')
    
    observation_normal = np.cross(posecef_first,posecef_last)
    observation_normal = observation_normal/np.linalg.norm(observation_normal)

    posecef_mid_unit = posecef_mid/np.linalg.norm(posecef_mid)
    ecef_to_local = R.align_vectors([[1,0,0],[0,1,0]],[posecef_mid_unit,observation_normal])[0]

    return ecef_to_local

def get_los_in_local_grid(df_row,icol,irow,stepsize,topalt,timescale, ecef_to_local):

    qp = R.from_quat(df_row['qprime'])
    satpos_eci = df_row['afsGnssStateJ2000'][0:3]
    date = df_row['EXPDate']
    current_ts = timescale.from_datetime(date)
    q = df_row['afsAttitudeState']
    quat = R.from_quat(np.roll(q, -1))
    
    x, y = pix_deg(df_row, icol, irow)
    los_sat = R.from_euler('xyz', [0, y, x], degrees=True).apply([1, 0, 0])
    los_eci = np.array(quat.apply(qp.apply(los_sat)))
    s_steps = generate_steps(stepsize)

    eci_to_ecef=R.from_matrix(itrs.rotation_at(current_ts))
    pos=np.expand_dims(satpos_eci, axis=0).T+s_steps*np.expand_dims(los_eci, axis=0).T
    posecef=(eci_to_ecef.apply(pos.T).astype('float32'))
    poslocal = ecef_to_local.apply(posecef) #convert to local
    poslocal_sph = cart2sph(poslocal)   #x: height, y: acrosstrac (angle), z: along track (angle)

    return poslocal_sph

def generate_grid(df,timescale, ecef_to_local):

    first = 0
    mid = int((len(df)-1)/2)
    last = len(df)-1

    stepsize = 100
    top_alt = 120e3


    satpos_eci = df['afsGnssStateJ2000'][mid][0:3]
    date = df['EXPDate'][mid]
    current_ts = timescale.from_datetime(date)
    localR = np.linalg.norm(sfapi.wgs84.latlon(df.TPlat[mid], df.TPlon[mid], elevation_m=0).at(current_ts).position.m)
    
    left_col = 0
    mid_col = int(df["NCOL"][0]/2) -1
    right_col =df["NCOL"][0] - 1
    irow = 0
    
    poslocal_sph = []
    poslocal_sph.append(get_los_in_local_grid(df.loc[first],left_col,irow,stepsize,top_alt,timescale,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[first],right_col,irow,stepsize,top_alt,timescale,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[first],mid_col,irow,stepsize,top_alt,timescale,ecef_to_local))

    poslocal_sph.append(get_los_in_local_grid(df.loc[mid],left_col,irow,stepsize,top_alt,timescale,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[mid],right_col,irow,stepsize,top_alt,timescale,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[mid],mid_col,irow,stepsize,top_alt,timescale,ecef_to_local))

    poslocal_sph.append(get_los_in_local_grid(df.loc[last],left_col,irow,stepsize,top_alt,timescale,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[last],right_col,irow,stepsize,top_alt,timescale,ecef_to_local))
    poslocal_sph.append(get_los_in_local_grid(df.loc[last],mid_col,irow,stepsize,top_alt,timescale,ecef_to_local))

    poslocal_sph = np.vstack(poslocal_sph)
    max_alt = poslocal_sph[:,0].max()
    min_alt = poslocal_sph[:,0].min()
    max_lat = poslocal_sph[:,2].max()
    min_lat = poslocal_sph[:,2].min()
    max_lon = poslocal_sph[:,1].max()
    min_lon = poslocal_sph[:,1].min()

    dz = 1e3
    nlon = 10
    nlat = 25

    altitude_grid = np.arange(localR+50e3,localR+130e3,2e3)
    altitude_grid[0] = altitude_grid[0]-30e3
    altitude_grid[-1] = altitude_grid[-1]+30e3
    acrosstrack_grid = np.arange(-0.1,0.1,0.01)
    #acrosstrack_grid = np.linspace(posecef_i_sph[:,1].min(),posecef_i_sph[:,1].max(),1)
    alongtrack_grid = np.linspace(poslocal_sph[:,2].min()-0.5,poslocal_sph[:,2].max()+0.5,50)
    alongtrack_grid[0] = alongtrack_grid[0]-0.5
    alongtrack_grid[-1] = alongtrack_grid[-1]+0.5

    # altitude_grid = np.arange(min_alt-5e3,localR+top_alt+5e3,dz)
    # acrosstrack_grid = np.linspace(min_lon,max_lon,nlon)
    # alongtrack_grid = np.linspace(min_lat,max_lat,nlat)

    return altitude_grid,acrosstrack_grid,alongtrack_grid

def get_tangent_geometry(df_row,spline_pixels,column):
    #calulate tangent geometries for center column limited rows and make a spline for it
    x, yv = pix_deg(df_row, column, spline_pixels)
    ecivec = np.zeros((3, len(yv)))
    for irow, y in enumerate(yv):
        los = R.from_euler('xyz', [0, y, x], degrees=True).apply([1, 0, 0])
        ecivec[:, irow] = np.array(quat.apply(qp.apply(los)))
    eci_spline=CubicSpline(spline_pixels,ecivec.T)

    return eci_spline#calculate jacobian for all measurements

def calc_jacobian(df,columns,rows):
    #spline_pixels = np.linspace(0, df['NROW'].iloc[0], 5)
    #s_steps = 500
    stepsize = 100
    top_alt = 120e3
    profiles = []

    timescale = sfapi.load.timescale()
    ecef_to_local = generate_local_transform(df,timescale)
    altitude_grid,acrosstrack_grid,alongtrack_grid = generate_grid(df,timescale, ecef_to_local)
    edges = [altitude_grid,acrosstrack_grid,alongtrack_grid]

    #Add check that all images are same format
    ks = sp.lil_matrix((len(rows)*len(columns)*len(df),(len(altitude_grid)-1)*(len(alongtrack_grid)-1)*(len(acrosstrack_grid)-1)))
    k_row = 0
    for i in range(len(df)):
        print("image number: " + str(i))
        for column in columns:
            p = prepare_profile(df.iloc[i],column,rows)
            profiles.append(p)
            for irow in rows:
                posecef_i_sph = get_los_in_local_grid(df.loc[i],column,irow,stepsize,top_alt,timescale,ecef_to_local)
                hist, _ = np.histogramdd(posecef_i_sph[::1,:],bins=edges)
                k = hist.reshape(-1)
                ks[k_row,:] = k
                k_row = k_row+1

    y = np.array(profiles)
 
    return  y, ks, altitude_grid, alongtrack_grid,acrosstrack_grid, ecef_to_local



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
