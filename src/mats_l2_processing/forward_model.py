# %%
import numpy as np
import pandas as pd
from mats_utils.geolocation.coordinates import col_heights, findheight
from mats_l1_processing.pointing import pix_deg
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import CubicSpline, interp1d
import scipy.sparse as sp
from skyfield import api as sfapi
from skyfield.framelib import itrs
import pickle
from mats_l2_processing.grids import cart2sph
from fast_histogram import histogramdd
from bisect import bisect_left
import boost_histogram as bh
import time
from itertools import chain, repeat
from multiprocessing import Pool

# %%


def load_abstable():
    global abstable_tan_height
    global abstable_distance
    factors_path = "/home/olemar/Projects/Universitetet/MATS/MATS-L2-processing/data/splinedlogfactorsIR2.npy"
    # factors_path = "/home/lk/mats-analysis/MATS-L2-processing/data/splinedlogfactorsIR2.npy"
    abstable_tan_height, abstable_distance = np.load(factors_path, allow_pickle=True)
    return


def generate_timescale():
    global timescale
    timescale = sfapi.load.timescale()
    return


def generate_stepsize():
    global stepsize
    # stepsize = 100
    stepsize = 10
    return


def generate_top_alt():
    global top_alt
    top_alt = 120e3
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
    if col is None:
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

def find_top_of_atmosphere(top_altitude,localR,satpos,losvec):
    sat_radial_pos = np.linalg.norm(satpos)
    los_zenith_angle = np.arccos(np.dot(satpos,losvec)/sat_radial_pos)
    #solving quadratic equation to find distance start and end of atmosphere 
    b = 2*sat_radial_pos*np.cos(los_zenith_angle)
    root=np.sqrt(b**2+4*((top_altitude+localR)**2 - sat_radial_pos**2))
    distance_top_1 =(-b-root)/2
    distance_top_2 =(-b+root)/2

    return [distance_top_1,distance_top_2]

def generate_steps(stepsize,top_altitude,localR,satpos,losvec):
    distance_top_of_atmosphere = find_top_of_atmosphere(top_altitude,localR,satpos,losvec)
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
    observation_normal = observation_normal/np.linalg.norm(observation_normal)  # normalize vector

    posecef_mid_unit = posecef_mid/np.linalg.norm(posecef_mid)  # unit vector for central position
    ecef_to_local = R.align_vectors([[1,0,0],[0,1,0]],[posecef_mid_unit,observation_normal])[0]

    return ecef_to_local

def get_los_ecef(image,icol,irow,rot_sat_channel,rot_sat_eci,eci_to_ecef):
    x, y = pix_deg(image, icol, irow) #Angles for single pixel
    rotation_channel_pix = R.from_euler('xyz', [0, y, x], degrees=True).apply([1, 0, 0]) #Rot matrix between pixel pointing and channels pointing
    los_eci = np.array(rot_sat_eci.apply(rot_sat_channel.apply(rotation_channel_pix)))
    los_ecef = eci_to_ecef.apply(los_eci) 
    return los_ecef

def get_steps_in_local_grid(image,ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False):

    if localR is None:
        date = image['EXPDate']
        current_ts = timescale.from_datetime(date)
        localR = np.linalg.norm(sfapi.wgs84.latlon(image.TPlat, image.TPlon, elevation_m=0).at(current_ts).position.m)        

    s_steps = generate_steps(stepsize,top_altitude=top_alt,localR=localR,satpos=satpos_ecef,losvec=los_ecef)

    posecef=(np.expand_dims(satpos_ecef, axis=0).T+s_steps*np.expand_dims(los_ecef, axis=0).T).astype('float32')
    poslocal = ecef_to_local.apply(posecef.T) #convert to local (for middle alongtrack measurement)
    poslocal_sph = cart2sph(poslocal)   
    poslocal_sph=np.array(poslocal_sph).T
    if do_abs:
        weights = get_weights(poslocal_sph,s_steps,localR)
    else:
        weights = np.ones((poslocal_sph.shape[0]))

    return poslocal_sph,weights

def get_weights(posecef_sph,s_steps,localR):
    #Calculate the weights 
    minarg=posecef_sph[:,0].argmin() #find tangent point
    distances=(s_steps-s_steps[minarg])/1000 #distance to tangent point (km)
    target_tanalt = (posecef_sph[minarg,0]-localR)/1000 #value of tangent altitude
    lowertan=bisect_left(abstable_tan_height,target_tanalt)-1 #index in table of value below tangent altitude
    uppertan=lowertan+1 #index in table of value above tangent altitude
    absfactor_distance_below=abstable_distance[lowertan](distances) #extract values below
    absfactor_distance_above=abstable_distance[uppertan](distances) #extract values above
    #make interpolater that generates optical depth for each distance as a function of tanalt
    interpolated_abs_factor=interp1d([abstable_tan_height[lowertan],abstable_tan_height[uppertan]],np.array([absfactor_distance_below,absfactor_distance_above]).T)
    #get transmissivity for generated optical depth vector
    weights=np.exp(interpolated_abs_factor(target_tanalt))
    return weights


def generate_grid(df, columns, rows, ecef_to_local, grid_proto=None):
    lims = grid_limits(df, columns, rows, ecef_to_local)
    print(lims)
    if grid_proto is None:
        grid_proto = [lims[i][2] for i in range(3)]
    result = []
    for i in range(3):
        if (type(grid_proto[i]) is int) or (type(grid_proto[i]) is float):
            grid_len = int(grid_proto[i])
            assert grid_len > 0, "Malformed grid_spec parameter!"
            result.append(np.linspace(lims[i][0], lims[i][1], grid_len))
        elif isinstance(grid_proto[i], np.ndarray):
            assert len(grid_proto[i].shape) == 1
            # for j, spec in enumerate(grid_spec[i]):
            #    assert type(spec) is tuple, "Malformed grid_spec parameter!"
            #    assert len(spec) == 3, "Malformed grid_spec parameter!"
            #    if j == 0:
            #        axis_grid = np.linspace(*spec)
            #    else:
            #        axis_grid = np.concatenate((axis_grid, np.linspace(*spec)))
            result.append(grid_proto[i][np.logical_and(grid_proto[i] > lims[i][0], grid_proto[i] < lims[i][1])])
        else:
            raise ValueError(f"Malformed grid_spec parameter: {type(grid_proto[i])}")
    return result


def grid_limits(df, columns, rows, ecef_to_local):

    first = 0
    mid = int((len(df)-1)/2)
    last = len(df)-1

    mid_date = df['EXPDate'][mid]
    current_ts = timescale.from_datetime(mid_date)
    localR = np.linalg.norm(sfapi.wgs84.latlon(df.TPlat[mid], df.TPlon[mid], elevation_m=0).at(current_ts).position.m)

    rot_sat_channel = R.from_quat(df.loc[mid]['qprime']) #Rotation between satellite pointing and channel pointing

    q = df.loc[mid]['afsAttitudeState'] #Satellite pointing in ECI
    rot_sat_eci = R.from_quat(np.roll(q, -1)) #Rotation matrix for q (should go straight to ecef?)

    eci_to_ecef=R.from_matrix(itrs.rotation_at(current_ts))
    satpos_eci = df.loc[mid]['afsGnssStateJ2000'][0:3] 
    satpos_ecef=eci_to_ecef.apply(satpos_eci)


    #change to do only used columns and rows
    if len(columns)==1:
        mid_col = columns[0]
    else:
        left_col = columns[0]
        mid_col = int(df["NCOL"][0]/2) -1
        right_col =df["NCOL"][0] - 1

    irow = rows[0]
    poslocal_sph = []       

    #Which localR to use in get_step_local_grid?

    los_ecef = get_los_ecef(df.loc[first],mid_col,irow,rot_sat_channel,rot_sat_eci,eci_to_ecef)
    poslocal_sph.append(get_steps_in_local_grid(df.loc[first],ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False)[0])
    los_ecef = get_los_ecef(df.loc[mid],mid_col,irow,rot_sat_channel,rot_sat_eci,eci_to_ecef)
    poslocal_sph.append(get_steps_in_local_grid(df.loc[mid],ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False)[0])
    los_ecef = get_los_ecef(df.loc[last],mid_col,irow,rot_sat_channel,rot_sat_eci,eci_to_ecef)
    poslocal_sph.append(get_steps_in_local_grid(df.loc[last],ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False)[0])


    if len(columns)>1:
        los_ecef = get_los_ecef(df.loc[first],left_col,irow,rot_sat_channel,rot_sat_eci,eci_to_ecef)
        poslocal_sph.append(get_steps_in_local_grid(df.loc[first],ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False)[0])
        los_ecef = get_los_ecef(df.loc[mid],left_col,irow,rot_sat_channel,rot_sat_eci,eci_to_ecef)
        poslocal_sph.append(get_steps_in_local_grid(df.loc[mid],ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False)[0])
        los_ecef = get_los_ecef(df.loc[last],left_col,irow,rot_sat_channel,rot_sat_eci,eci_to_ecef)
        poslocal_sph.append(get_steps_in_local_grid(df.loc[last],ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False)[0])

        los_ecef = get_los_ecef(df.loc[first],right_col,irow,rot_sat_channel,rot_sat_eci,eci_to_ecef)
        poslocal_sph.append(get_steps_in_local_grid(df.loc[first],ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False)[0])
        los_ecef = get_los_ecef(df.loc[mid],right_col,irow,rot_sat_channel,rot_sat_eci,eci_to_ecef)
        poslocal_sph.append(get_steps_in_local_grid(df.loc[mid],ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False)[0])
        los_ecef = get_los_ecef(df.loc[last],right_col,irow,rot_sat_channel,rot_sat_eci,eci_to_ecef)
        poslocal_sph.append(get_steps_in_local_grid(df.loc[last],ecef_to_local, satpos_ecef, los_ecef, localR=None, do_abs=False)[0])

    poslocal_sph = np.vstack(poslocal_sph)
    max_rad = poslocal_sph[:,0].max()
    min_rad = poslocal_sph[:,0].min()
    max_along = poslocal_sph[:,2].max()
    min_along = poslocal_sph[:,2].min()
    max_across = poslocal_sph[:,1].max()
    min_across = poslocal_sph[:,1].min()
    if len(columns) < 2:
        max_across = max_across + 0.2
        min_across = min_across - 0.2

    nalt = int(len(rows/2))+2
    nacross = int(len(columns/2))+2
    nalong = int(len(df)/2)+2

    if (nacross-2)<2:
        nacross=2
    if (nalong-2)<2:
        nalong=2

    # altitude_grid = np.linspace(min_rad-10e3,localR+top_alt+10e3,nalt)
    # acrosstrack_grid = np.linspace(min_across-0.1,max_across+0.1,nacross)
    # alongtrack_grid = np.linspace(min_along-0.2,max_along+0.2,nalong)

    return (min_rad - 10e3, localR + top_alt + 10e3, nalt), (min_across - 0.1, max_across + 0.1, nacross), \
        (min_along - 0.6, max_along + 0.6, nalong)


def calc_jacobian(df, columns, rows, edges=None, grid_proto=None, processes=4):
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
    generate_timescale()  # generates a global timescale object
    generate_stepsize()
    generate_top_alt()
    load_abstable()
    ecef_to_local = generate_local_transform(df)

    if edges is None:
        altitude_grid, acrosstrack_grid, alongtrack_grid = generate_grid(df, columns, rows, ecef_to_local, grid_proto)
        edges = [altitude_grid, acrosstrack_grid, alongtrack_grid]

    is_regular_grid = True
    for axis_edges in edges:
        widths = np.diff(axis_edges)
        if not all(np.abs(widths - np.mean(widths)) < 0.001 * np.abs(np.mean(widths))):
            is_regular_grid = False
            break
    do_abs = (df['channel'][0] == "IR2")
    if do_abs:
        print("Warning: absorbtion disabled for selected channel (not implemented yet).")
    per_image_args = [(i, df.loc[i], df.iloc[i], df['EXPDate'][i]) for i in range(len(df))]
    common_args = (edges, is_regular_grid, do_abs, columns, rows,
                   ecef_to_local, altitude_grid, alongtrack_grid, acrosstrack_grid)
    time0 = time.time()
    with Pool(processes=processes) as pool:
        results = pool.starmap(image_jacobian, zip(per_image_args, repeat(common_args)))
    time1 = time.time()
    print("Assembling sparse Jacobian matrix...")
    K = sp.vstack([k_part for k_part, _, _ in results])
    y = np.array(list(chain.from_iterable([profiles for _, profiles, _ in results])))
    tan_alts = np.array(list(chain.from_iterable([profiles for _, _, profiles in results])))

    time2 = time.time()
    print(f"Jacobian contribution from {len(df)} images calculated in {time1 - time0:.1f} s.")
    print(f"Results assembled to a sparse matrix in {time2 - time1:.1f} s.")
    return y, K, altitude_grid, alongtrack_grid, acrosstrack_grid, ecef_to_local, tan_alts


def image_jacobian(per_image_arg, common_args):
    i, loc, iloc, expDate = per_image_arg
    edges, grid_is_regular, do_abs, columns, rows, \
        ecef_to_local, altitude_grid, alongtrack_grid, acrosstrack_grid = common_args

    print(f"Processing of image {i} started.")
    tic = time.time()

    image_profiles = []
    image_tanalts = []
    image_K = sp.lil_array((len(rows) * len(columns),
                           (len(altitude_grid) - 1) * (len(alongtrack_grid) - 1) * (len(acrosstrack_grid) - 1)))
    image_k_row = 0

    rot_sat_channel = R.from_quat(loc['qprime'])  # Rotation between satellite pointing and channel pointing
    q = loc['afsAttitudeState']  # Satellite pointing in ECI
    rot_sat_eci = R.from_quat(np.roll(q, -1))  # Rotation matrix for q (should go straight to ecef?)

    current_ts = timescale.from_datetime(expDate)
    localR = np.linalg.norm(sfapi.wgs84.latlon(loc.TPlat, loc.TPlon, elevation_m=0).at(current_ts).position.m)
    print(localR)
    eci_to_ecef = R.from_matrix(itrs.rotation_at(current_ts))
    satpos_eci = loc['afsGnssStateJ2000'][0:3]
    satpos_ecef = eci_to_ecef.apply(satpos_eci)
    for column in columns:
        s_profile, s_tanalt = prepare_profile(iloc, column, rows)
        image_profiles.append(s_profile)
        image_tanalts.append(s_tanalt)
        for irow in rows:
            los_ecef = get_los_ecef(loc, column, irow, rot_sat_channel, rot_sat_eci, eci_to_ecef)
            posecef_i_sph, weights = get_steps_in_local_grid(loc, ecef_to_local, satpos_ecef, los_ecef, localR,
                                                             do_abs=do_abs)
            if grid_is_regular:
                hist = histogramdd(posecef_i_sph[::1, :], weights=weights,
                    range=[[edges[0][0], edges[0][-1]], [edges[1][0], edges[1][-1]], [edges[2][0], edges[2][-1]]],
                    bins=[len(edges[0]) - 1, len(edges[1]) - 1, len(edges[2]) - 1])
            else:
                hist, _ = np.histogramdd(posecef_i_sph[::1, :], bins=edges, weights=weights)
            image_K[image_k_row, :] = hist.reshape(-1)
            image_k_row += 1

    toc = time.time()
    print(f"Image {i} processed in {toc-tic:.1f} s.")
    return image_K, image_profiles, image_tanalts

# %%
