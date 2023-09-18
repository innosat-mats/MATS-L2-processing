#%%
import pickle
from mats_l2_processing.inverse_model import do_inversion,generate_xa_from_gaussian
import numpy as np
from matplotlib import pyplot as plt
from mats_l2_processing.grids import localgrid_to_lat_lon_alt_3D, center_grid, geoid_radius
from scipy.interpolate import griddata

filename = "/home/olemar/Projects/Universitetet/MATS/MATS-L2-processing/jacobian_3d.pkl"

with open(filename, "rb") as file:
    [y, ks, altitude_grid_edges, alongtrack_grid_edges,acrosstrack_grid_edges, ecef_to_local] = pickle.load(file)

# %%
radius_grid = center_grid(altitude_grid_edges)
alongtrack_grid = center_grid(alongtrack_grid_edges)
acrosstrack_grid = center_grid(acrosstrack_grid_edges)

non_uniform_ecef_grid_altitude,non_uniform_ecef_grid_lon,non_uniform_ecef_grid_lat,non_uniform_ecef_grid_r = localgrid_to_lat_lon_alt_3D(radius_grid,acrosstrack_grid,alongtrack_grid,ecef_to_local)

xa = generate_xa_from_gaussian(non_uniform_ecef_grid_altitude)*2e12+1e11
y = y.flatten()
x_hat = do_inversion(ks,y,xa=xa)
#%%
plt.plot(x_hat)
plt.plot(xa,'r--')
plt.show()
#%%
plt.plot(ks.dot(x_hat)[:],label = 'fitted')
plt.plot(y[:].T,'r--',label = 'measured')
plt.plot(ks.dot(xa)[:],label = 'aprioi')
plt.legend()
plt.title('y space metrics')
plt.show()

#%%
plt.plot(ks.dot(x_hat)[:],label = 'fitted')
plt.plot(y[:].T,'r',label = 'measured')
plt.plot(ks.dot(xa)[:],'g--',label = 'aprioi')
plt.legend()
plt.title('y space metrics zoom')
plt.xlim([10000,11000])
plt.ylim([-1e15,2.5e16])
plt.show()

#%%
plt.plot(ks.dot(x_hat)-y)
plt.title('residuals')
plt.show()

#%%
plt.plot(ks.dot(x_hat)-y)
plt.title('residuals zoom')
plt.xlim([10000,11000])
plt.ylim([-0.5*1e15,0.5*1e15])
plt.show()

#%%
plt.plot(ks.dot(x_hat)-y)
plt.title('residuals zoom')
plt.xlim([10000,11000])
plt.ylim([-2e15,2e15])
plt.show()

#%%
x_hat_reshape1 = np.array(x_hat).reshape(len(radius_grid)
,len(acrosstrack_grid)
,len(alongtrack_grid))

xa_reshape1 = np.array(xa).reshape(len(radius_grid)
,len(acrosstrack_grid)
,len(alongtrack_grid))

#%%
plt.pcolor(np.sum(x_hat_reshape1[:,:,:],axis=0))
plt.show()
# # %%
plt.pcolor(np.sum(x_hat_reshape1[:,:,:],axis=1))
plt.show()

#%%
plt.pcolor(np.sum(x_hat_reshape1[:,:,:],axis=2))
plt.show()

#%%
plt.pcolor(np.sum(xa_reshape1[:,:,:],axis=0))
plt.show()
# # %%
# plt.pcolor(np.sum(x_hat_reshape1[:,:,:],axis=1))
# plt.show()

#%%
plt.pcolor(np.sum(xa_reshape1[:,:,:],axis=2))
plt.show()

# %%
#crop edges

x_hat_reshape1_cropped = x_hat_reshape1[10:-20,:,20:-20]

# %%
plt.pcolor(np.sum(x_hat_reshape1_cropped[:,:,:],axis=1))
plt.clim([0,5e12])
plt.colorbar()
plt.title('2D retrieval')
plt.show()

# #%%
# alt_grid = np.arange(70e3, 110e3, 1e3)
# along_grid = np.arange(-0.5, 0.5, 0.005)
# across_grid = np.arange(-0.05, 0.05, 0.0025)

# #%%
# _, acrosstrack_grid_expanded, alongtrack_grid_expanded = np.meshgrid(radius_grid, acrosstrack_grid, alongtrack_grid, indexing='ij')

# #%%
# data_points = np.array([non_uniform_ecef_grid_altitude.flatten(),acrosstrack_grid_expanded.flatten(),alongtrack_grid_expanded.flatten()]).T  # Shape: (N, 3)
# #data_points = np.array([alt.flatten(),lon.flatten(),lat.flatten()]).T  # Shape: (N, 3)
# values = x_hat_reshape1.flatten()       # Shape: (N,)

# #%%
# x_grid, y_grid, z_grid = np.meshgrid(alt_grid, across_grid,along_grid, indexing='ij')

# #%%
# interpgrid = np.array([x_grid.flatten(),y_grid.flatten(),z_grid.flatten()]).T
# # Interpolate using linear interpolation
# #%%
# interpolated_values = griddata(data_points, values, interpgrid, method='linear')

# #%%
# interpolated_values = interpolated_values.reshape(len(alt_grid),len(across_grid),len(along_grid))



#%%
alt_grid = np.arange(70e3, 120e3, 0.5e3)
along_grid = np.arange(-0.5, 0.5, 0.005)

#%%
_, alongtrack_grid_expanded = np.meshgrid(radius_grid, alongtrack_grid, indexing='ij')

#%%
data_points = np.array([non_uniform_ecef_grid_altitude.flatten(),alongtrack_grid_expanded.flatten()]).T  # Shape: (N, 3)
#data_points = np.array([alt.flatten(),lon.flatten(),lat.flatten()]).T  # Shape: (N, 3)
values = x_hat_reshape1.flatten()       # Shape: (N,)

#%%

#%%
x_grid, z_grid = np.meshgrid(alt_grid, along_grid, indexing='ij')

#%%
interpgrid = np.array([x_grid.flatten(),z_grid.flatten()]).T
# Interpolate using linear interpolation
#%%
interpolated_values = griddata(data_points, values, interpgrid, method='nearest',rescale=True)
#%%
interpolated_values = interpolated_values.reshape(len(alt_grid),len(along_grid))


# %%
plt.scatter(data_points[::1,1],data_points[::1,0],10,values[::1])

# %%
plt.pcolor(along_grid,alt_grid,interpolated_values)
plt.clim([-1e12,1e13])
plt.xlim([-0.3,0.3])
plt.colorbar()
plt.title('Data interpolated onto regular grid')
plt.show()
# %%
