import wandb
import gc

from copy import deepcopy

import lightning as L
import torch

from torch.utils.data import DataLoader

from awave2.awd.attributer import SaliencyAttributer
from awave2.awd.loss import AWDLossConfig
from awave2.dwt.dwt1d import DWT1d, DWT1dConfig
from awave2.dwt.loss import DWTLossConfig
from pytorch_lightning.loggers import WandbLogger

from cnnwave import CNNWave
from fdns_data import load_data

torch.cuda.empty_cache()

gc.collect()

cnn = torch.load("trained-cnn/CNN.pt")

cnn.eval()


trainset, testset, input_normalized_val_torch, output_normalized_val_torch = load_data(
    None
)

trainloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)
valloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)  # valN
batch = next(iter(trainloader))

dwt_config = DWT1dConfig(
    dim_size=batch[0].shape[-1],
    dim=-1,
    init_wavelet="bior3.1",
    padding_mode="zero",
    learn_dual=True,  # True for biorthogonal wavelets
)

dwt0 = DWT1d(dwt_config)
dwt1 = deepcopy(dwt0)

attributer = None  # SaliencyAttributer()

awd_loss_config = AWDLossConfig(wt_loss_config=DWTLossConfig())
awd_loss_config.recon_loss_config.weight = 1
awd_loss = awd_loss_config.get_loss_module()


cnnwave = CNNWave(
    cnn, dwts=[dwt0, dwt1], loss=awd_loss, attributer=attributer, verbose=True
)

# import os
# path = 'awd-dev/sa49qwqc/checkpoints/'
# fname = os.listdir(path)[0]
# cnnwave = CNNWave.load_from_checkpoint(path+fname,
#                                       cnn=cnn, dwt=dwt, attributer=attributer, loss=awd_loss)

print(cnnwave.dwt0.h0)
print(cnnwave.dwt0.h1)

print(cnnwave.dwt1.h0)
print(cnnwave.dwt1.h1)


# wandb_logger = WandbLogger(project="awd-dev")

with torch.no_grad():
    torch.cuda.empty_cache()

gc.collect()
trainer = L.Trainer(
    limit_train_batches=10,
    max_epochs=300,  # accumulate_grad_batches=4,
    # logger=wandb_logger,
    log_every_n_steps=1,
)
trainer.fit(model=cnnwave, train_dataloaders=trainloader, val_dataloaders=valloader)
wandb.finish()

gc.collect()
torch.cuda.empty_cache()
