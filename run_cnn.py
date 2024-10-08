import sys
import os
import gc
import yaml
import numpy as np
import h5py
import torch
import lightning as L

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import DataLoader, TensorDataset

from awave2.dwt.dwt1d import DWT1d, DWT1dConfig
from awave2.dwt.loss import DWTLossConfig
from awave2.ddw.loss import DDWLossConfig

from cnn import CNN
from net import Net
from unet import UNet

config_path = sys.argv[1]
name = config_path.split('/')[-1].split('.')[0]

assert os.path.exists(config_path), f"Invalid config path: {config_path}"
config = yaml.safe_load(open(config_path, "r"))

torch.cuda.empty_cache()
gc.collect()

train_dir = config["data"]["train_dir"]
train_input_file = config["data"]["train_input_file"]
train_input_path = f"{train_dir}/{train_input_file}"

keys = config["data"]["keys"]
n_channels = len(keys)

with h5py.File(train_input_path, "r") as f:
    data = [np.array(f[key], np.float32) for key in keys] #np.array(f[key], np.float32)
    # normalize
    data = [(d-np.mean(d))/np.std(d) for d in data]
    Nlon = data[0].shape[0]
    Nlat = data[0].shape[1]
    N = data[0].shape[2]
data = np.array(data)
# data is [channels, Nlon, Nlat, N]
data = np.moveaxis(data, -1, 0)
# data is [N, channels, Nlon, Nlat]
print(data.shape)
assert data.shape == (N, n_channels, Nlon, Nlat)

train_N = config["data"]["train_N"]

train_input_mat = data[:train_N, :, :, :]
val_input_mat = data[train_N:, :, :, :]

assert train_input_mat.shape == (train_N, n_channels, Nlon, Nlat)
assert val_input_mat.shape == (N - train_N, n_channels, Nlon, Nlat)

train_output_file = config["data"]["train_output_file"]
train_output_path = f"{train_dir}/{train_output_file}"
with h5py.File(train_output_path, "r") as f:
    data = np.array(f['PI'], np.float32)

if 'residual' in config:
    from scipy.io import loadmat
    normalization = loadmat(f"{config['data']['train_dir']}/Normalization_coefficients_train.mat")
    data = data * normalization['SDEV_IPI'][0][0] + normalization['MEAN_IPI'][0][0]
    with h5py.File(train_input_path, "r") as f:
        data -= np.array(f[config['residual']], np.float32)
    data = data / normalization['SDEV_IPI'][0][0]

# data is [Nlon, Nlat, N]
data = np.moveaxis(data, -1, 0)
data = np.expand_dims(data, axis=1)
print(data.shape)
assert data.shape == (N, 1, Nlon, Nlat)

train_output_mat = data[:train_N, :, :, :]
val_output_mat = data[train_N:, :, :, :]

assert train_output_mat.shape == (train_N, 1, Nlon, Nlat)
assert val_output_mat.shape == (N - train_N, 1, Nlon, Nlat)

train_input_mat = torch.from_numpy(train_input_mat).float()
train_output_mat = torch.from_numpy(train_output_mat).float()

val_input_mat = torch.from_numpy(val_input_mat).float()
val_output_mat = torch.from_numpy(val_output_mat).float()

trainset = TensorDataset(train_input_mat, train_output_mat)
valset = TensorDataset(val_input_mat, val_output_mat)

trainloader = DataLoader(trainset, **config["dataloader_train"])
batch = next(iter(trainloader))

valloader = DataLoader(valset, **config["dataloader_val"])

if 'cnn' in config:
    network_module = Net(n_channels=n_channels, n_channels_out=1, **config['cnn'])
elif 'unet' in config:
    network_module = UNet(n_channels_in=n_channels, n_channels_out=1, **config['unet'])
cnn = CNN(cnn=network_module, optimizer_config=config["optimizer"], verbose=True)

config["checkpoint"]["filename"] = name + '-{epoch:02d}'
checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

# if slurm
if "SLURM_JOB_ID" in os.environ:
    config['slurm_job_id'] = os.environ['SLURM_JOB_ID']

config["wandb"]["name"] = name
wandb_logger = WandbLogger(config=config, **config["wandb"])

trainer = L.Trainer(
    logger=wandb_logger, callbacks=[checkpoint_callback], **config["trainer"]
)

trainer.fit(model=cnn, train_dataloaders=trainloader, val_dataloaders=valloader)

gc.collect()
torch.cuda.empty_cache()
