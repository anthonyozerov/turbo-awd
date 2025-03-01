import sys
import os
import numpy as np

from scipy.io import savemat
from turboawd.aposteriori import load_online_data
from py2d.filter import filter2D

# read experiment dir from sys.argv
if len(sys.argv) < 2:
    print("Usage: python filter_online.py <experiment_dir>")
    sys.exit(1)
experiment_dir = sys.argv[1]
# remove trailing slash

if experiment_dir.endswith('/'):
    experiment_dir = experiment_dir[:-1]
assert os.path.exists(experiment_dir)

omegas, _ = load_online_data(experiment_dir+'/data')
assert omegas.shape[1:] == (1024, 1024)

nx = 1024
Lx = 2 * np.pi
NCoarse = 128
Delta = 2*Lx/NCoarse

filtered_dir = experiment_dir+'-filtered'

# make directory to save filtered omegas in
os.makedirs(filtered_dir, exist_ok=True)
os.makedirs(filtered_dir+'/data', exist_ok=True)

# filter and save omegas
for i in range(omegas.shape[0]):
    omega_filtered = filter2D(omegas[i], filterType='gaussian', coarseGrainType='spectral', Delta=Delta, Ngrid=np.ones(2, dtype=int)*NCoarse)
    savemat(os.path.join(filtered_dir+'/data', f'{i+1}.mat'), {'Omega': omega_filtered})
