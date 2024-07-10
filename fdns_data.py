from functools import partial
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import random_split, TensorDataset, DataLoader
import h5py

# General variables
trainN = 1500
valN = 50
# lead=1
num_epochs = 500
# pool_size = 2
# drop_prob=0.0
conv_activation = "relu"
Nlat = 128
Nlon = 128
n_channels = 2
n_channels_out = 1
NT = trainN  # Numer of snapshots per file
numDataset = 1  # number of dataset / 2
new = 0  # 1 - new CNN; 0 - read from pre-trained CNN


def load_data(data_dir='fdns-data'):
    input_normalized = np.zeros([NT, Nlon, Nlat, n_channels], np.float32)
    output_normalized = np.zeros([NT, Nlon, Nlat, n_channels_out], np.float32)
    input_normalized_val = np.zeros([valN, Nlon, Nlat, n_channels], np.float32)
    output_normalized_val = np.zeros([valN, Nlon, Nlat, n_channels_out], np.float32)

    # Training data input
    Filename = f"{data_dir}/FDNS Psi W_train.mat"
    with h5py.File(Filename, "r") as f:
        Psi = np.array(f["Psi"], np.float32)

        input_normalized[:, :, :, 0] = np.moveaxis(Psi[:, :, :trainN], -1, 0)
        del Psi
        f.close()

    Filename = f"{data_dir}/FDNS Psi W_train.mat"
    with h5py.File(Filename, "r") as f:
        w = np.array(f["W"], np.float32)

        input_normalized[:, :, :, 1] = np.moveaxis(w[:, :, :trainN], -1, 0)
        del w
        f.close()

    # Validation data input
    Filename = f"{data_dir}/FDNS Psi W_val.mat"
    with h5py.File(Filename, "r") as f:
        Psi = np.array(f["Psi"], np.float32)

        input_normalized_val[:, :, :, 0] = np.moveaxis(Psi[:, :, -valN:], -1, 0)
        del Psi
        f.close()

    Filename = f"{data_dir}/FDNS Psi W_val.mat"
    with h5py.File(Filename, "r") as f:
        w = np.array(f["W"], np.float32)

        input_normalized_val[:, :, :, 1] = np.moveaxis(w[:, :, -valN:], -1, 0)
        del w
        f.close()

    input_normalized = np.moveaxis(input_normalized, -1, 1)
    input_normalized_val = np.moveaxis(input_normalized_val, -1, 1)

    # Training data output
    Filename = f"{data_dir}/FDNS PI_train.mat"
    with h5py.File(Filename, "r") as f:
        PI = np.array(f["PI"], np.float32)
        output_normalized[:, :, :, 0] = np.moveaxis(PI[:, :, :trainN], -1, 0)
        del PI
        f.close()

    # Validation data output
    Filename = f"{data_dir}/FDNS PI_val.mat"
    with h5py.File(Filename, "r") as f:
        PI = np.array(f["PI"], np.float32)
        output_normalized_val[:, :, :, 0] = np.moveaxis(PI[:, :, -valN:], -1, 0)
        del PI
        f.close()

    output_normalized = np.moveaxis(output_normalized, -1, 1)
    output_normalized_val = np.moveaxis(output_normalized_val, -1, 1)
    print("Size of input:")
    print(input_normalized.shape)
    print("Size of output:")
    print(output_normalized.shape)

    input_normalized_torch = torch.from_numpy(input_normalized).float()  # .cuda()
    output_normalized_torch = torch.from_numpy(output_normalized).float()  # .cuda()

    input_normalized_val_torch = torch.from_numpy(
        input_normalized_val
    ).float()  # .cuda()
    output_normalized_val_torch = torch.from_numpy(
        output_normalized_val
    ).float()  # .cuda()


    train_dataset = TensorDataset(
        input_normalized_torch, output_normalized_torch
    )  # create your training datset
    val_dataset = TensorDataset(
        input_normalized_val_torch, output_normalized_val_torch
    )  # create your val datset

    del input_normalized_torch
    del output_normalized_torch
    del input_normalized
    del output_normalized
    del input_normalized_val
    del output_normalized_val

    return (
        train_dataset,
        val_dataset,
    )
