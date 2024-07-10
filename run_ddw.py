import sys

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from torch.utils.data import DataLoader
from awave2.dwt.dwt1d import DWT1d, DWT1dConfig
from awave2.dwt.loss import DWTLossConfig
from awave2.ddw.loss import DDWLossConfig

from fdns_data import load_data

import gc

from lightning.pytorch.loggers import WandbLogger

from ddw import DDW


which = sys.argv[1]

if which not in ["input", "output"]:
    print("Invalid argument")
    sys.exit(1)

print(which)

torch.cuda.empty_cache()
gc.collect()

trainset, testset = load_data()

trainloader = DataLoader(trainset, batch_size=8, shuffle=True, num_workers=2)
valloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)  # valN
batch = next(iter(trainloader))

dwt_config = DWT1dConfig(
    dim_size=batch[0].shape[-1],
    dim=-1,
    init_wavelet="bior3.1",
    padding_mode="zero",
    learn_dual=True,  # True for biorthogonal wavelets
)

dwt_loss_config = DWTLossConfig()
ddw_loss_config = DDWLossConfig(wt_loss_config=DWTLossConfig())
ddw_loss = ddw_loss_config.get_loss_module()

if which == "input":
    dwt0 = DWT1d(dwt_config)
    dwt1 = DWT1d(dwt_config)
    dwts = [dwt0, dwt1]
else:
    dwt = DWT1d(dwt_config)
    dwts = [dwt]

ddw = DDW(loss=ddw_loss, dwts=dwts, which=which, verbose=True)

checkpoint_callback = ModelCheckpoint(
    monitor="loss",
    dirpath=f"{which}",
    filename="ddw-{epoch:02d}",
    save_top_k=3,
    mode="min",
)

wandb_logger = WandbLogger(name=f"{which}", project="awd-dev")
trainer = L.Trainer(
    max_epochs=1000,  # accumulate_grad_batches=4,
    logger=wandb_logger,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
)

trainer.fit(model=ddw, train_dataloaders=trainloader, val_dataloaders=valloader)

gc.collect()
torch.cuda.empty_cache()
