import gc
import numpy as np
import h5py
import os
import sys
import yaml

import lightning as L
import torch

from torch.utils.data import DataLoader, TensorDataset

from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from cnnoutwave import CNNOutWave

from net import Net
from ddw import DDW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config_path = sys.argv[1]

assert os.path.exists(config_path), f"Invalid config path: {config_path}"
config = yaml.safe_load(open(config_path, "r"))

torch.cuda.empty_cache()
gc.collect()

print("loading data")

train_input_dir = config["data"]["train_input_dir"]
train_input_file = config["data"]["train_input_file"]
train_input_path = f"{train_input_dir}/{train_input_file}"

train_output_dir = config["data"]["train_output_dir"]
train_output_file = config["data"]["train_output_file"]
train_output_path = f"{train_output_dir}/{train_output_file}"

train_N = config["data"]["train_N"]

with h5py.File(train_input_path, "r") as f:
    psi = np.array(f["Psi"], np.float32)
    w = np.array(f["W"], np.float32)

Nlon = psi.shape[0]
Nlat = psi.shape[1]

input_mat = np.zeros([train_N, Nlon, Nlat, 2])

# data is [Nlon, Nlat, N]
input_mat[:, :, :, 0] = np.moveaxis(psi[:, :, :train_N], -1, 0)
input_mat[:, :, :, 1] = np.moveaxis(w[:, :, :train_N], -1, 0)
# train_mat is [train_N, Nlon, Nlat, 1]
input_mat = np.moveaxis(input_mat, -1, 1)
# now train_mat is [train_N, 1, Nlon, Nlat]
assert input_mat.shape == (train_N, 2, Nlon, Nlat)

input_mat = torch.from_numpy(input_mat).float()

with h5py.File(train_output_path, "r") as f:
    pi = np.array(f["PI"], np.float32)

output_mat = np.zeros([train_N, Nlon, Nlat, 1])
output_mat[:, :, :, 0] = np.moveaxis(pi[:, :, :train_N], -1, 0)
output_mat = np.moveaxis(output_mat, -1, 1)
assert output_mat.shape == (train_N, 1, Nlon, Nlat)

output_mat = torch.from_numpy(output_mat).float()

trainset = TensorDataset(input_mat, output_mat)

trainloader = DataLoader(trainset, **config["dataloader"])
batch = next(iter(trainloader))

cnn_path = config["cnn"]["cnn_path"]
cnn = torch.load(cnn_path, map_location=device)
batch = next(iter(trainloader))

# use most recent ddw in directory (by timestamp)
ddw_dir = config["ddw"]["ddw_dir"]
ddw_files = os.listdir(ddw_dir)
ddw_files = [f for f in ddw_files if f.endswith(".ckpt")]
ddw_files = sorted(ddw_files, key=lambda f: os.path.getmtime(f"{ddw_dir}/{f}"))
ddw_file = ddw_files[-1]
print(ddw_file)

ddw = DDW.load_from_checkpoint(f"{ddw_dir}/{ddw_file}")

dwt = ddw.dwts[0].eval()

with torch.no_grad():
    torch.cuda.empty_cache()
gc.collect()

cnnoutwave = CNNOutWave(
    cnn=cnn, dwt=dwt, optimizer_config=config["optimizer"], verbose=True
)
checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

wandb_logger = WandbLogger(config=config, **config["wandb"])

trainer = L.Trainer(
    logger=wandb_logger, callbacks=[checkpoint_callback], **config["trainer"]
)

trainer.fit(model=cnnoutwave, train_dataloaders=trainloader)

gc.collect()
torch.cuda.empty_cache()
