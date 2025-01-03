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

from turboawd.utils import load_data

from turboawd.ddw import DDW

########################## LOAD CONFIG ##############################
print("Loading config")

config_path = sys.argv[1]

assert os.path.exists(config_path), f"Invalid config path: {config_path}"
config = yaml.safe_load(open(config_path, "r"))
name = config['identifier']

which = config["which"]
print(which)

torch.cuda.empty_cache()
gc.collect()

########################## LOAD DATA ##############################
print("Loading data")

train_dir = config["data"]["train_dir"]
train_file = config["data"]["train_file"]
train_path = f"{train_dir}/{train_file}"
key = config["data"]["key"]

train_N = config["data"]["train_N"]

input_train = load_data(train_path, [key], before=train_N, tensor=True)
assert input_train.shape == (train_N, 1, 128, 128)
input_val = load_data(train_path, [key], after=train_N, tensor=True)

trainset = TensorDataset(input_train)
valset = TensorDataset(input_val)

trainloader = DataLoader(trainset, **config["dataloader_train"])
valloader = DataLoader(valset, **config["dataloader_val"])

batch = next(iter(trainloader))

########################## INIT DDW ##############################
dwt_config = DWT1dConfig(dim_size=batch[0].shape[-1], dim=-1, **config["dwt"])

dwt_loss_config = DWTLossConfig()
ddw_loss_config = DDWLossConfig(wt_loss_config=DWTLossConfig())
ddw_loss = ddw_loss_config.get_loss_module()

dwt = DWT1d(dwt_config)
dwts = [dwt]

ddw = DDW(loss=ddw_loss, dwts=dwts, optimizer_config=config["optimizer"], verbose=True)

# test saving to ONNX
print("Testing saving ONNX...")
input_sample = torch.randn(batch[0].shape)
os.makedirs(config["checkpoint"]["dirpath"], exist_ok=True)
savepath = f"{config['checkpoint']['dirpath']}/{name}.onnx"
ddw.to_onnx(savepath, input_sample, export_params=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
# test running ONNX
print("Testing loading ONNX...")
sess = rt.InferenceSession(savepath)
rt_inputs = {sess.get_inputs()[0].name: input_sample.detach().numpy()}
rt_outs = sess.run(None, rt_inputs)

# test with smaller batch
smallbatch_size=batch[0].shape[0]//2
rt_inputs = {sess.get_inputs()[0].name: input_sample.detach().numpy()[:smallbatch_size]}
rt_outs = sess.run(None, rt_inputs)
# end session
del sess

########################## TRAIN DDW ##############################
checkpoint_callback = ModelCheckpoint(**config["checkpoint"])

wandb_logger = WandbLogger(config=config, **config["wandb"])

trainer = L.Trainer(
    logger=wandb_logger, callbacks=[checkpoint_callback], **config["trainer"]
)

trainer.fit(model=ddw, train_dataloaders=trainloader, val_dataloaders=valloader)

ddw.to_onnx(savepath, input_sample, export_params=True,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})

gc.collect()
torch.cuda.empty_cache()
