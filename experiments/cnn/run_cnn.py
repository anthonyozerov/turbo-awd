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

from turboawd.cnn import CNN
from turboawd.net import Net

# load configuration
config_path = sys.argv[1]
assert os.path.exists(config_path), f"Invalid config path: {config_path}"
config_meta = yaml.safe_load(open(config_path, "r"))

architecture = config_meta["architecture"]
channels = config_meta["channels"]
data = config_meta["data"]
training = config_meta["training"]

config_architecture = yaml.safe_load(open(f"configs/architecture/{architecture}.yaml", "r"))
config_channels = yaml.safe_load(open(f"configs/channels/{channels}.yaml", "r"))
config_data = yaml.safe_load(open(f"configs/data/{data}.yaml", "r"))
config_training = yaml.safe_load(open(f"configs/training/{training}.yaml", "r"))

config = {**config_architecture, **config_channels, **config_data, **config_training}
name = '-'.join([data,architecture,channels,training])

#name = '-'.join(config_path.split('/')[1:]).split('.')[0]
print(name)

torch.cuda.empty_cache()
gc.collect()

# load data
train_dir = config["data"]["train_dir"]
train_input_file = config["data"]["train_input_file"]
train_input_path = f"{train_dir}/{train_input_file}"

# keys says which channels will be used
keys = config["input_channels"]
n_channels = len(keys)

# load input data
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

# load output data
train_output_file = config["data"]["train_output_file"]
train_output_path = f"{train_dir}/{train_output_file}"
with h5py.File(train_output_path, "r") as f:
    data = np.array(f['PI'], np.float32)

# make output data be residual if needed
if 'residual' in config:
    from scipy.io import loadmat
    if 'norm_file' in config['data']:
        norm_path = config['data']['train_dir'] + '/' + config['data']['norm_file']
    else:
        norm_path = config['data']['train_dir'] + '/Normalization_coefficients_train.mat'
    normalization = loadmat(norm_path)

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

# make inputs and outputs TensorDataset

train_input_mat = torch.from_numpy(train_input_mat).float()
train_output_mat = torch.from_numpy(train_output_mat).float()

val_input_mat = torch.from_numpy(val_input_mat).float()
val_output_mat = torch.from_numpy(val_output_mat).float()

trainset = TensorDataset(train_input_mat, train_output_mat)
valset = TensorDataset(val_input_mat, val_output_mat)

# define dataloaders
trainloader = DataLoader(trainset, **config["dataloader_train"])
valloader = DataLoader(valset, **config["dataloader_val"])

# sample batch (used to get shape)
batch = next(iter(trainloader))

# initialize CNN architecture
if 'cnn' in config:
    network_module = Net(n_channels=n_channels, n_channels_out=1, **config['cnn'])

# initialize CNN object
cnn = CNN(cnn=network_module, optimizer_config=config["optimizer"], verbose=True)

# define checkpoint callback
config["checkpoint"]["filename"] = name + '-{epoch:02d}'
checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

# if slurm, save job id to config
if "SLURM_JOB_ID" in os.environ:
    config['slurm_job_id'] = os.environ['SLURM_JOB_ID']

# set up weights and biases logger
config["wandb"]["name"] = name
wandb_logger = WandbLogger(config=config, **config["wandb"])

# set up and fit trainer
trainer = L.Trainer(
    logger=wandb_logger, callbacks=[checkpoint_callback], **config["trainer"]
)
trainer.fit(model=cnn, train_dataloaders=trainloader, val_dataloaders=valloader)

# save model as torchscript
cnn.to_torchscript(f"{config['checkpoint']['dirpath']}/{name}-ts.pt")

gc.collect()
torch.cuda.empty_cache()