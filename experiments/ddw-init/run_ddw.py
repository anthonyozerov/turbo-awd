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

from ddw import DDW

config_path = sys.argv[1]

assert os.path.exists(config_path), f"Invalid config path: {config_path}"
config = yaml.safe_load(open(config_path, "r"))

which = config["which"]
print(which)

torch.cuda.empty_cache()
gc.collect()

train_dir = config["data"]["train_dir"]
train_file = config["data"]["train_file"]
train_path = f"{train_dir}/{train_file}"
key = config["data"]["key"]

with h5py.File(train_path, "r") as f:
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

trainloader = DataLoader(trainset, **config["dataloader"])
batch = next(iter(trainloader))

dwt_config = DWT1dConfig(dim_size=batch[0].shape[-1], dim=-1, **config["dwt"])

dwt_loss_config = DWTLossConfig()
ddw_loss_config = DDWLossConfig(wt_loss_config=DWTLossConfig())
ddw_loss = ddw_loss_config.get_loss_module()

dwt = DWT1d(dwt_config)
dwts = [dwt]

ddw = DDW(loss=ddw_loss, dwts=dwts, optimizer_config=config["optimizer"], verbose=True)

checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

wandb_logger = WandbLogger(config=config, **config["wandb"])

trainer = L.Trainer(
    logger=wandb_logger, callbacks=[checkpoint_callback], **config["trainer"]
)

trainer.fit(model=ddw, train_dataloaders=trainloader)

gc.collect()
torch.cuda.empty_cache()
