# %%
from py2d.gradient_model import TauGM4, PiOmegaGM2, PiOmegaGM4
from py2d.initialize import initialize_wavenumbers_rfft2
import numpy as np
import h5py
from scipy.io import loadmat, savemat
from tqdm import tqdm
import sys
import os

# get args from command line
path = sys.argv[1]
if len(sys.argv) < 3:
    which = ''
    normalized = True
else:
    which = f'_{sys.argv[2]}'
    normalized = False

# %%
nx = 128
Lx = 2 * np.pi

N_LES = 128*np.ones(2, dtype=int)
Delta = 2*Lx/N_LES[0]

dealias = False

# Initialize wavenumbers for derivative calculation
Kx, Ky, Ksq, _, invKsq = initialize_wavenumbers_rfft2(nx, nx, Lx, Lx, INDEXING='ij')

# %%
if not normalized:
    normalization = loadmat(f'{path}/Normalization_coefficients{which}.mat')
with h5py.File(f'{path}/FDNS Psi W{which}.mat', 'r') as f:
    psi_data = np.array(f['Psi'], np.float32)
    omega_data = np.array(f['W'], np.float32)
    if not normalized:
        psi_data = psi_data * normalization['SDEV_IP'][0][0] + normalization['MEAN_IP'][0][0]
        omega_data = omega_data * normalization['SDEV_IW'][0][0] + normalization['MEAN_IW'][0][0]
with h5py.File(f'{path}/FDNS U V{which}.mat', 'r') as f:
    u_data = np.array(f['U'], np.float32)
    v_data = np.array(f['V'], np.float32)
    if not normalized:
        u_data = u_data * normalization['SDEV_IU'][0][0] + normalization['MEAN_IU'][0][0]
        v_data = v_data * normalization['SDEV_IV'][0][0] + normalization['MEAN_IV'][0][0]

# %%
N = u_data.shape[2]
Tau11GM4_data, Tau12GM4_data, Tau22GM4_data = np.zeros((nx,nx,N)), np.zeros((nx,nx,N)), np.zeros((nx,nx,N))
GM2_data, GM4_data = np.zeros((nx,nx,N)), np.zeros((nx,nx,N))

filterType = 'gaussian'
for i in tqdm(range(N)):
    Tau11GM4, Tau12GM4, Tau22GM4 = TauGM4(u_data[:,:,i], v_data[:,:,i], Kx, Ky, Delta=Delta)
    Tau11GM4_data[:,:,i] = Tau11GM4
    Tau12GM4_data[:,:,i] = Tau12GM4
    Tau22GM4_data[:,:,i] = Tau22GM4
    GM2_data[:,:,i] = PiOmegaGM2(omega_data[:,:,i], u_data[:,:,i], v_data[:,:,i], Kx, Ky, Delta, filterType=filterType, spectral=False, dealias=dealias)
    GM4_data[:,:,i] = PiOmegaGM4(omega_data[:,:,i], u_data[:,:,i], v_data[:,:,i], Kx, Ky, Delta, filterType=filterType, spectral=False, dealias=dealias)

# %%
with h5py.File(f'{path}/FDNS_big{which}.mat', 'w') as f:
    f['psi'] = psi_data
    f['omega'] = omega_data
    f['u'] = u_data
    f['v'] = v_data
    f['tau11GM4'] = Tau11GM4_data
    f['tau12GM4'] = Tau12GM4_data
    f['tau22GM4'] = Tau22GM4_data
    f['gm2'] = GM2_data
    f['gm4'] = GM4_data

for i in range(10):
    index = np.random.randint(0, N)
    savemat(f'{path}/fdns_ic_{i}{which}.mat', {'Omega': omega_data[:,:,index].astype(np.float64)})
