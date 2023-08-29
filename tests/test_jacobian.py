#calculate jacobian for all measurements
import numpy as np
import pandas as pd
from mats_l2_processing.forward_model import calc_jacobian
import pickle


dftop = pd.read_pickle('/home/olemar/Projects/Universitetet/MATS/MATS-analysis/Donal/retrievals/verdec2d.pickle')
#%%
# starttime=datetime(2023,3,31,21,0)
# stoptime=datetime(2023,3,31,22,35)
# dftop=read_MATS_data(starttime,stoptime,level="1b",version="0.4")

# with open('verdec2d.pickle', 'wb') as handle:
#     pickle.dump(dftop, handle)
#%%
# df=df[df['channel']!='NADIR']
df = dftop[dftop['channel'] == 'IR2'].dropna().reset_index(drop=True)#[0:10]


#select part of orbit
offset = 300
num_profiles = 100 #use 50 profiles for inversion
df = df.loc[offset:offset+num_profiles-1]
df = df.reset_index(drop=True)
columns = np.arange(0,df["NCOL"][0],2)
rows = np.arange(0,df["NROW"][0]-10,1)

y, ks, altitude_grid, alongtrack_grid,acrosstrack_grid, ecef_to_local = calc_jacobian(df,columns,rows)
filename = "jacobian_3.pkl"

with open(filename, "wb") as file:
    pickle.dump((y, ks, altitude_grid, alongtrack_grid,acrosstrack_grid, ecef_to_local), file)

with open(filename, "rb") as file:
    [y, ks, altitude_grid, alongtrack_grid,acrosstrack_grid, ecef_to_local] = pickle.load(file)

