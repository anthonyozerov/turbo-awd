# %%
from py2d.gradient_model import TauGM4, PiOmegaGM2, PiOmegaGM4
from py2d.initialize import initialize_wavenumbers_rfft2
import numpy as np
import h5py
from scipy.io import loadmat
from tqdm import tqdm

# %%
nx = 128
Lx = 2 * np.pi

N_LES = 128*np.ones(2, dtype=int)
Delta = 2*Lx/N_LES[0]

dealias = False

# Initialize wavenumbers for derivative calculation
Kx, Ky, Ksq, _, invKsq = initialize_wavenumbers_rfft2(nx, nx, Lx, Lx, INDEXING='ij')

# %%
normalization = loadmat('Normalization_coefficients_train.mat')
with h5py.File('FDNS Psi W_train.mat', 'r') as f:
    psi_data = np.array(f['Psi'], np.float32) * normalization['SDEV_IP'][0][0] + normalization['MEAN_IP'][0][0]
    omega_data = np.array(f['W'], np.float32) * normalization['SDEV_IW'][0][0] + normalization['MEAN_IW'][0][0]
with h5py.File('FDNS U V_train.mat', 'r') as f:
    u_data = np.array(f['U'], np.float32) * normalization['SDEV_IU'][0][0] + normalization['MEAN_IU'][0][0]
    v_data = np.array(f['V'], np.float32) * normalization['SDEV_IV'][0][0] + normalization['MEAN_IV'][0][0]


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
with h5py.File('FDNS_big_train.mat', 'w') as f:
    f['psi'] = psi_data
    f['omega'] = omega_data
    f['u'] = u_data
    f['v'] = v_data
    f['tau11GM4'] = Tau11GM4_data
    f['tau12GM4'] = Tau12GM4_data
    f['tau22GM4'] = Tau22GM4_data
    f['gm2'] = GM2_data
    f['gm4'] = GM4_data