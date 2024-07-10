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
from copy import deepcopy

from lightning.pytorch.loggers import WandbLogger


which = sys.argv[1]

if which not in ["input", "output"]:
    print("Invalid argument")
    sys.exit(1)

print(which)

torch.cuda.empty_cache()
gc.collect()

trainset, testset = load_data()

trainloader = DataLoader(trainset, batch_size=100, shuffle=True, num_workers=1)
valloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)  # valN
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
    dwt1 = deepcopy(dwt0)
    dwts = [dwt0, dwt1]
else:
    dwt = DWT1d(dwt_config)
    dwts = [dwt]


class DDW(L.LightningModule):
    def __init__(self, loss, dwts, verbose=False):
        super().__init__()
        self.loss = loss
        self.dwts = dwts
        self.verbose = verbose
        self.n_dwts = len(dwts)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if which == "input":
            z = x
        else:
            z = y
        if len(z.shape) == 3:
            z = z.unsqueeze(1)
        z_coeffs = [
            self.dwts[i](z[:, i, :, :].unsqueeze(1)) for i in range(self.n_dwts)
        ]
        z_recons = [self.dwts[i].inverse(z_coeffs[i]) for i in range(self.n_dwts)]
        z_recon = torch.cat(z_recons, dim=1)

        loss = self.loss(z, z_coeffs, z_recon, wts=self.dwts)

        return loss["loss"]

    def configure_optimizers(self):
        params = sum([list(wt.parameters()) for wt in self.dwts], [])
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return optimizer


ddw = DDW(loss=ddw_loss, dwts=dwts, verbose=True)

checkpoint_callback = ModelCheckpoint(
    monitor="loss",
    dirpath=f"{which}",
    filename="sample-mnist-{epoch:02d}",
    save_top_k=3,
    mode="min",
)

wandb_logger = WandbLogger(name=f'{which}',
                           project="awd-dev")
trainer = L.Trainer(
    limit_train_batches=100,
    max_epochs=100,  # accumulate_grad_batches=4,
    logger=wandb_logger,
    log_every_n_steps=1,
    callbacks=[checkpoint_callback],
)


trainer.fit(model=ddw, train_dataloaders=trainloader, val_dataloaders=valloader)

gc.collect()
torch.cuda.empty_cache()
