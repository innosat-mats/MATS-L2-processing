import numpy as np

def geoid_radius(latitude):
    '''
    Function from GEOS5 class.
    GEOID_RADIUS calculates the radius of the geoid at the given latitude
    [Re] = geoid_radius(latitude) calculates the radius of the geoid (km)
    at the given latitude (degrees).
    ----------------------------------------------------------------
            Craig Haley 11-06-04
    ---------------------------------------------------------------
    '''
    DEGREE = np.pi / 180.0
    EQRAD = 6378.14 * 1000
    FLAT = 1.0 / 298.257
    Rmax = EQRAD
    Rmin = Rmax * (1.0 - FLAT)
    Re = np.sqrt(1./(np.cos(latitude*DEGREE)**2/Rmax**2
                + np.sin(latitude*DEGREE)**2/Rmin**2)) 
    return Re

def sph2cart(r, phi, theta):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def cart2sph(pos):
    x = pos[:,0]
    y = pos[:,1]
    z = pos[:,2]
    radius = np.sqrt(x**2 + y**2 + z**2)
    longitude = np.arctan2(y, x)
    latitude = np.arcsin(z / radius)

    return np.array([radius,longitude,latitude]).T


def localgrid_to_lat_lon_alt_3D(ecef_to_local,altitude_grid,acrosstrack_grid,alongtrack_grid):
    arrayshape = (len(altitude_grid),len(acrosstrack_grid),len(alongtrack_grid))
    non_uniform_ecef_grid_x = np.zeros(arrayshape)
    non_uniform_ecef_grid_y = np.zeros(arrayshape)
    non_uniform_ecef_grid_z = np.zeros(arrayshape)

    non_uniform_ecef_grid_altitude = np.zeros(arrayshape)
    non_uniform_ecef_grid_r = np.zeros(arrayshape)
    non_uniform_ecef_grid_lat = np.zeros(arrayshape)
    non_uniform_ecef_grid_lon = np.zeros(arrayshape)

    for i in range(arrayshape[0]):
        for j in range(arrayshape[1]):
            for k in range(arrayshape[2]):
                non_uniform_ecef_grid_x[i,j,k],non_uniform_ecef_grid_y[i,j,k],non_uniform_ecef_grid_z[i,j,k]=ecef_to_local.inv().apply(sph2cart(altitude_grid[i],acrosstrack_grid[j],alongtrack_grid[k]))
                non_uniform_ecef_grid_r[i,j,k],non_uniform_ecef_grid_lon[i,j,k],non_uniform_ecef_grid_lat[i,j,k] = cart2sph(non_uniform_ecef_grid_x[i,j,k],non_uniform_ecef_grid_y[i,j,k],non_uniform_ecef_grid_z[i,j,k])
                non_uniform_ecef_grid_altitude[i,j,k] = non_uniform_ecef_grid_r[i,j,k]-geoid_radius(non_uniform_ecef_grid_lat[i,j,k])

    return non_uniform_ecef_grid_altitude,non_uniform_ecef_grid_lat,non_uniform_ecef_grid_lon,non_uniform_ecef_grid_r
