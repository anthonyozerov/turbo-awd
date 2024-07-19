import sys
import os
import gc
import yaml
import numpy as np
import h5py
import torch

from torch import nn

from torch.utils.data import DataLoader, TensorDataset

from awave2.dwt.dwt1d import DWT1d, DWT1dConfig
from awave2.dwt.loss import DWTLossConfig
from awave2.ddw.loss import DDWLossConfig
from awave2.dwt.utils import get_dwt_padding

import pywt

# get the config file path from the command line
config_path = sys.argv[1]
assert os.path.exists(config_path), f"Invalid config path: {config_path}"
config = yaml.safe_load(open(config_path, "r"))

which = config["which"]
print(which)

torch.cuda.empty_cache()
gc.collect()

# Get the data parameters from the config file
# to change which data is used, change the train_dir, train_file, and key
train_dir = config["data"]["train_dir"]
abs_path = os.path.abspath(train_dir)
assert os.path.exists(train_dir), f"Invalid train_dir: {abs_path}"

train_file = config["data"]["train_file"]
train_path = f"{train_dir}/{train_file}"
abs_path = os.path.abspath(train_path)
assert os.path.exists(train_path), f"File: {abs_path} not found"

key = config["data"]["key"]
# Load the data and put it in the right shape
with h5py.File(train_path, "r") as f:
    assert key in f, f"Key: {key} not found in file: {train_file}"
    data = np.array(f[key], np.float32)

Nlon = data.shape[0]
Nlat = data.shape[1]

train_N = config["data"]["train_N"]
train_mat = np.zeros([train_N, Nlon, Nlat, 1])

# data is [Nlon, Nlat, N]
train_mat[:, :, :, 0] = np.moveaxis(data[:, :, :train_N], -1, 0)
# train_mat is [train_N, Nlon, Nlat, 1]
train_mat = np.moveaxis(train_mat, -1, 1)
# now train_mat is [train_N, 1, Nlon, Nlat]
assert train_mat.shape == (train_N, 1, Nlon, Nlat)

train_mat = torch.from_numpy(train_mat).float()
trainset = TensorDataset(train_mat)

# This is the dataloader we will use to optimize
# Its batch size and number of workers can be set in the config file
trainloader = DataLoader(trainset, **config["dataloader"])
batch = next(iter(trainloader))

# Set up the DWT1d object which we will use.
# It is configured in the config file
# The initial wavelet given in the config doesn't matter, as the object is
# just a placeholder where we will put our own wavelet parameters
dwt_config = DWT1dConfig(dim_size=batch[0].shape[-1], dim=-1, **config["dwt"])
dwt = DWT1d(dwt_config)

# Set up the loss module.
dwt_loss_config = DWTLossConfig()
ddw_loss_config = DDWLossConfig(wt_loss_config=DWTLossConfig())
ddw_loss = ddw_loss_config.get_loss_module()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# we are optimizing for h0 and h1. These should be numpy arrays which
# are the lowpass filters for the wavelet transform.
# h0 is deconstruction, h1 is reconstruction.
# note that the highpass filters are not needed as they can be obtained
# from the lowpass filters.


# z is a list with one element which has shape [batch_size, 1, Nlon, Nlat]
# (z can be obtained from a dataloader)
def forward(h0, h1, batch):
    # the next few lines just work with h0 and h1 to make them the right shape and
    # class to assign to the DWT1d object.
    _h0 = np.array(h0[::-1]).ravel()
    _h1 = np.array(h1).ravel()

    t = torch.get_default_dtype()

    dwt.h0 = nn.Parameter(torch.tensor(_h0, device=device, dtype=t).reshape((1, 1, -1)))
    dwt.h1 = nn.Parameter(torch.tensor(_h1, device=device, dtype=t).reshape((1, 1, -1)))
    dwt.pad = get_dwt_padding(
        N=dwt.dim_size, L=dwt.h0.numel(), J=dwt.num_detail_levels, mode=dwt.pywt_mode
    )

    # tell pytorch to not do gradient computations, because we don't need them!
    with torch.no_grad():
        # compute the wavelet transform
        z = batch[0]
        z_coeffs = dwt(z)
        # compute the inverse wavelet transform
        z_recon = dwt.inverse(z_coeffs)
        # compute the loss
        loss_dict = ddw_loss(z, [z_coeffs], z_recon, wts=[dwt])
        # print(f'recon: {loss_dict["recon"]}, coeff: {loss_dict["coeff"]}')

    return loss_dict["loss"]


# here is an example where, instead of optimizing for h0, h1,
# we will just run the forward transform a bunch of times with different
# biorthogonal wavelets obtained from PyWavelets
data = next(iter(trainloader))

wavelets = pywt.wavelist("bior")
for w in wavelets:
    h0, g0, h1, g1 = pywt.Wavelet(w).filter_bank
    print(w, forward(h0, h1, data))

# I imagine the optimization will be more like:
# init_h0, _, init_h1, _ = pywt.wavelet("bior1.5").filter_bank
# for i in range(num_iterations):
#     data = next(iter(trainloader))
#     for j in range(ensemble_size):
#         # somehow pick h0 and h1
#         # save the loss from forward(h0, h1, data)
#     do something

# if the optimization is over orthogonal wavelets, then just set
# h1 as the flipped version of h0. (if h0=[1,2,3], then h1=[3,2,1])
# In this case, learn_dual should be set to false in the config file.

# it might be good to start experimentation with bior1.3, as it has
# only 10 parameters (5 for h0, 5 for h1). bior3.1 has only 6 parameters,
# and Adam is able to optimize it pretty nicely.
