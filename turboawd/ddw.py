import lightning as L
import torch

from torch.nn import ModuleList


class DDW(L.LightningModule):
    def __init__(self, loss, dwts, optimizer_config, verbose=False):
        super().__init__()
        self.save_hyperparameters()

        self.loss = loss
        self.dwts = ModuleList(dwts)
        self.verbose = verbose
        self.n_dwts = len(dwts)
        self.optimizer_config = optimizer_config

    def training_step(self, batch, batch_idx):
        z = batch[0]
        z_coeffs = [
            self.dwts[i](z[:, i, :, :].unsqueeze(1)) for i in range(self.n_dwts)
        ]
        z_recons = [self.dwts[i].inverse(z_coeffs[i]) for i in range(self.n_dwts)]
        z_recon = torch.cat(z_recons, dim=1)

        loss = self.loss(z, z_coeffs, z_recon, wts=self.dwts)

        self.log_dict(loss, on_epoch=True)

        return loss["loss"]

    def configure_optimizers(self):
        params = sum([list(wt.parameters()) for wt in self.dwts], [])
        optimizer = torch.optim.Adam(params, **self.optimizer_config)
        return optimizer
