#%%
import pickle
from mats_l2_processing.inverse_model import do_inversion
import numpy as np
from matplotlib import pyplot as plt

filename = "/home/olemar/Projects/Universitetet/MATS/MATS-L2-processing/jacobian_3.pkl"

with open(filename, "rb") as file:
    [y, ks, altitude_grid_edges, alongtrack_grid_edges,acrosstrack_grid_edges, ecef_to_local] = pickle.load(file)

altitude_grid_ref = np.load("/home/olemar/Projects/Universitetet/MATS/MATS-L2-processing/edges0.npy")
alongtrack_grid_ref = np.load("/home/olemar/Projects/Universitetet/MATS/MATS-L2-processing/edges1.npy")
acrosstrack_grid_ref = np.load("/home/olemar/Projects/Universitetet/MATS/MATS-L2-processing/edges2.npy")
y = y.reshape(-1)
y = np.matrix(y)

x_hat = do_inversion(ks,y)

plt.plot(x_hat)
plt.show()


plt.plot(ks.dot(x_hat)[::10])
plt.plot(y[:,::10].T,'r--')
plt.show()

#%%
def center_grid(grid):
    return (grid[:-1]+grid[1:])/2

# %%
altitude_grid = center_grid(altitude_grid_edges)
alongtrack_grid = center_grid(alongtrack_grid_edges)
acrosstrack_grid = center_grid(acrosstrack_grid_edges)

x_hat_reshape1 = np.array(x_hat).reshape(len(altitude_grid)
,len(acrosstrack_grid)
,len(alongtrack_grid))

#%%
plt.pcolor(np.sum(x_hat_reshape1[:,:,:],axis=0))
plt.show()
# %%
plt.pcolor(np.sum(x_hat_reshape1[:,:,:],axis=1))
plt.show()

#%%
plt.pcolor(np.sum(x_hat_reshape1[:,:,:],axis=2))
plt.show()