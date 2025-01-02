import sys
import os
import gc
import yaml
import numpy as np
import h5py
import torch
import lightning as L
import onnxruntime as rt

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from torch.utils.data import DataLoader, TensorDataset

from awave2.dwt.dwt1d import DWT1d, DWT1dConfig
from awave2.dwt.loss import DWTLossConfig
from awave2.ddw.loss import DDWLossConfig

from turboawd.cnn import CNN
from turboawd.net import Net
from turboawd.utils import load_cnn_config, load_data, normalize

########################## LOAD CONFIGS ##############################
print('Loading configs')
# load configuration
# the main config file specifies the names of the other config files
config_path = sys.argv[1]

config, name = load_cnn_config(config_path)

print(name)

# if slurm, save job id to config
if "SLURM_JOB_ID" in os.environ:
    config["slurm_job_id"] = os.environ["SLURM_JOB_ID"]

torch.cuda.empty_cache()
gc.collect()

########################## LOAD DATA ##############################
print('Loading data')
# load data
train_dir = config["data"]["train_dir"]
train_input_file = config["data"]["train_input_file"]
train_input_path = f"{train_dir}/{train_input_file}"

# keys says which channels will be used
keys = config["input_channels"]
n_channels = len(keys)
train_N = config["data"]["train_N"]

input_train = load_data(train_input_path, keys, centerscale=True, before=train_N, tensor=True)
input_val = load_data(train_input_path, keys, centerscale=True, after=train_N, tensor=True)
assert input_train.shape == (train_N, n_channels, 128, 128)

train_output_path = config["data"]["train_dir"] + "/" + config["data"]["train_output_file"]
if "residual" in config:
    norm_path = config["data"]["train_dir"] + "/" + config["data"]["norm_file"]
    output = load_data(train_output_path, ["PI"], norm_path=norm_path,
                       norm_keys=["IPI"], denorm=True)
    residual = load_data(train_input_path, [config["residual"]])
    output -= residual
    output = normalize(output, norm_path, ["IPI"], sd_only=True)

else:
    output = load_data(train_output_path, ["PI"])

output_train = output[:train_N]
output_val = output[train_N:]

assert output_train.shape == (train_N, 1, 128, 128)

output_train = torch.from_numpy(output_train).float()
output_val = torch.from_numpy(output_val).float()

trainset = TensorDataset(input_train, output_train)
valset = TensorDataset(input_val, output_val)

# define dataloaders
trainloader = DataLoader(trainset, **config["dataloader_train"])
valloader = DataLoader(valset, **config["dataloader_val"])

# sample batch (used to get shape)
batch = next(iter(trainloader))

########################## INITIALIZE CNN ##############################
print('Initializing CNN')
# initialize CNN architecture
network_module = Net(
    n_channels=n_channels, n_channels_out=1, **config["cnn-architecture"]
)

# initialize CNN object
cnn = CNN(cnn=network_module, optimizer_config=config["optimizer"], verbose=True)

# test saving to ONNX
print("Testing saving ONNX...")
input_sample = torch.randn(batch[0].shape)
savepath = f"{config['checkpoint']['dirpath']}/{name}.onnx"
cnn.to_onnx(savepath, input_sample, export_params=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
# test running ONNX
print("Testing loading ONNX...")
sess = rt.InferenceSession(savepath)
rt_inputs = {sess.get_inputs()[0].name: input_sample.detach().numpy()}
rt_outs = sess.run(None, rt_inputs)
assert rt_outs[0].shape == batch[1].shape
# test with smaller batch
smallbatch_size=batch[0].shape[0]//2
rt_inputs = {sess.get_inputs()[0].name: input_sample.detach().numpy()[:smallbatch_size]}
rt_outs = sess.run(None, rt_inputs)
assert rt_outs[0].shape == (smallbatch_size, 1, 128, 128)
# end sesssion
del sess

########################## TRAIN CNN ##############################
print('Setting up training')

# define checkpoint callback
config["checkpoint"]["filename"] = name + "-{epoch:02d}"
checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

# set up weights and biases logger
config["wandb"]["name"] = name
wandb_logger = WandbLogger(config=config, **config["wandb"])

# set up and fit trainer
trainer = L.Trainer(
    logger=wandb_logger, callbacks=[checkpoint_callback], **config["trainer"]
)
trainer.fit(model=cnn, train_dataloaders=trainloader, val_dataloaders=valloader)

# save model as ONNX
cnn.to_onnx(savepath, input_sample, export_params=True)

gc.collect()
torch.cuda.empty_cache()
