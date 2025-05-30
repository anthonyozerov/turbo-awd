import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import onnxruntime as rt
import h5py
import numpy as np
import random
import glob

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader, TensorDataset


from turboawd.utils import load_online_config, load_data

with open("../aposteriori/epoch_sd_results/epoch_sd.yaml", "r") as f:
    epoch_sd_results = yaml.safe_load(f)

# get cnn names

online_config_names = list(epoch_sd_results.keys())
paths = [f"../online/configs/{name}.yaml" for name in online_config_names]
online_configs = [
    yaml.safe_load(open(path, "r")) for path in paths
]
for c in online_configs:
    print(c)

apriori_path = '../apriori/results/A/results-4.h5'
with h5py.File(apriori_path, 'r') as f:
    print(f.keys())
    psiomega = np.array(f['psiomega'])

# now we make our dataset
# every x is an image with 3 channels:
# psi, omega, and the CNN's output
# and y is the corresponding sd for that CNN in the online experiment

xs_train = []
ys_train = []

xs_test = []
ys_test = []

n_models = sum([len(epoch_sd_results[name]) for name in online_config_names])
n_train = np.floor(n_models * 0.7)

train_idx = np.random.choice(n_models, int(n_train), replace=False)

print('creating dataset')
idx = 0
for i in range(len(online_configs)):

    config = online_configs[i]
    name = online_config_names[i]
    cnn_name = config['cnn']
    print(f"Processing {name}...")

    epochs = epoch_sd_results[name].keys()

    for epoch in epochs:

        # print(f"  Epoch {epoch}...")
        # load the data
        sd = epoch_sd_results[name][epoch]

        with h5py.File(apriori_path, 'r') as f:
            cnn_output = np.array(f[cnn_name][str(epoch)])
        # print(cnn_output.shape)
        
        for j in range(len(cnn_output)):
            x = np.stack([psiomega[j][0], psiomega[j][1], cnn_output[j]], axis=0)
            assert x.shape == (3, 128, 128), f"Shape mismatch: {x.shape} != (3, 128, 128)"

            if idx in train_idx:
                xs_train.append(x)
                ys_train.append(sd)
            else:
                xs_test.append(x)
                ys_test.append(sd)

        idx += 1

# make a torch dataset
x_train = np.array(xs_train)
y_train = np.array(ys_train)

x_test = np.array(xs_test)
y_test = np.array(ys_test)

x_train = torch.from_numpy(x_train).float()
y_train = torch.from_numpy(y_train).float()

x_test = torch.from_numpy(x_test).float()
y_test = torch.from_numpy(y_test).float()

trainset = TensorDataset(x_train, y_train)
valset = TensorDataset(x_test, y_test)

# Add custom rotated dataset for training
class RotatedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        inp, out = self.dataset[idx]
        # Choose a random rotation: 90, 180, or 270 degrees (k rotations)
        k = random.choice([0, 1, 2, 3])
        # Assume images have shape (channels, H, W)
        inp = torch.rot90(inp, k, dims=(1, 2))
        return inp, out

trainset = RotatedDataset(trainset)
# create dataloaders, with random rotation for x_train
trainloader = DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2
)
valloader = DataLoader(
    valset, batch_size=128, shuffle=False, num_workers=2
)
print('dataset created')

# sample batch (used to get shape)
batch = next(iter(trainloader))

############################# INITIALIZE CNN ##############################

# input is 128x128 with 3 channels
# output should be scalar
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, padding=2, padding_mode='circular')
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=2, padding_mode='circular')
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, padding=2, padding_mode='circular')
        self.conv4 = nn.Conv2d(32, 32, kernel_size=5, padding=2, padding_mode='circular')
        self.conv5 = nn.Conv2d(32, 32, kernel_size=5, padding=2, padding_mode='circular')
        self.fc1 = nn.Linear(32 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # now we have 32 channels of 64x64
        x = self.pool(F.relu(self.conv2(x)))
        # now we have 32 channels of 32x32
        x = self.pool(F.relu(self.conv3(x)))
        # now we have 32 channels of 16x16
        x = self.pool(F.relu(self.conv4(x)))
        # now we have 32 channels of 8x8
        x = self.pool(F.relu(self.conv5(x)))
        # now we have 32 channels of 4x4
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

import lightning as L
from lightning.pytorch import Trainer

class CNN(L.LightningModule):
    def __init__(self, net, verbose=False):
        super().__init__()
        self.save_hyperparameters()

        self.net = net

        self.verbose = verbose

    def forward(self, x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        y_cnn = self.forward(x)

        # l1 loss
        l1_loss = F.l1_loss(y_cnn.squeeze(), y)
        # mse loss
        mse_loss = F.mse_loss(y_cnn.squeeze(), y)
        
        loss = {"l1": l1_loss, "mse": mse_loss}

        self.log_dict(loss, on_epoch=True)

        return loss["l1"]
    
    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_cnn = self.forward(x)

        # l1 loss
        l1_loss = F.l1_loss(y_cnn.squeeze(), y)
        # mse loss
        mse_loss = F.mse_loss(y_cnn.squeeze(), y)

        loss = {"l1_val": l1_loss, "mse_val": mse_loss}

        self.log_dict(loss, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=2e-5)
        return optimizer

# Initialize the CNN model
cnn = CNN(net, verbose=True)

# test saving to ONNX
print("Testing saving ONNX...")
ckpt_dir = 'checkpoints'
name = "cd_cnn"
os.makedirs(ckpt_dir, exist_ok=True)

input_sample = torch.randn(batch[0].shape)
savepath = f"{ckpt_dir}/{name}.onnx"
savepath_test = f"{ckpt_dir}/{name}_test.onnx"
cnn.to_onnx(
    savepath_test,
    input_sample,
    export_params=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)
# test running ONNX
print("Testing loading ONNX...")
sess = rt.InferenceSession(savepath_test)
rt_inputs = {sess.get_inputs()[0].name: input_sample.detach().numpy()}
rt_outs = sess.run(None, rt_inputs)
assert rt_outs[0].squeeze().shape == batch[1].shape, f"Output shape mismatch: {rt_outs[0].shape} != {batch[1].shape}"
# test with smaller batch
smallbatch_size = batch[0].shape[0] // 2
rt_inputs = {sess.get_inputs()[0].name: input_sample.detach().numpy()[:smallbatch_size]}
rt_outs = sess.run(None, rt_inputs)
assert rt_outs[0].squeeze().shape == (smallbatch_size,)
# end session
del sess
os.remove(savepath_test)

######### TRAIN CNN ##############

# Initialize the Wandb logger
wandb_logger = WandbLogger(
    project="turboawd",
    name="cd_model",
    log_model=True,
    save_dir=".",
)

print("Setting up training")

# define checkpoint callback
checkpoint_callback = ModelCheckpoint(filename=name + "-{epoch:02d}", save_top_k=1, monitor="l1_val", mode="min", dirpath=ckpt_dir)

# search for an existing checkpoint to resume training
resume_checkpoint = None
checkpoint_pattern = os.path.join(ckpt_dir, name + "-*.ckpt")
ckpt_files = sorted(glob.glob(checkpoint_pattern))
if ckpt_files:
    resume_checkpoint = ckpt_files[-1]
    print(f"Resuming training from {resume_checkpoint}")

# set up weights and biases logger
wandb_logger = WandbLogger(name=name, project="awd-dev", group='cd-cnn')

# set up and fit trainer
trainer = L.Trainer(
    logger=wandb_logger,
    callbacks=[checkpoint_callback],
    max_epochs=1000,
)
trainer.fit(
    model=cnn,
    train_dataloaders=trainloader,
    val_dataloaders=valloader,
    ckpt_path=resume_checkpoint,
)

# save model as ONNX
cnn.to_onnx(
    savepath,
    input_sample,
    export_params=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
)