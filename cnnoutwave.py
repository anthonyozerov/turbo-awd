import lightning as L
import torch
from torch.nn import MSELoss


class CNNOutWave(L.LightningModule):
    def __init__(self, cnn, dwt, optimizer_config, verbose=False):
        super().__init__()
        self.cnn = cnn
        self.dwt = dwt
        self.loss = MSELoss()
        self.verbose = verbose
        self.optimizer_config = optimizer_config

    def training_step(self, batch, batch_idx):
        if self.verbose:
            print("training step")
        x, y = batch

        if self.verbose:
            print(x.shape)
            print("forward transform")

        y_cnn = self.cnn(x)

        y_cnn_coeffs = torch.cat(self.dwt(y_cnn), axis=-1)
        y_coeffs = torch.cat(self.dwt(y), axis=-1)

        if self.verbose:
            print("computing loss")
        loss = self.loss(y_cnn_coeffs, y_coeffs)

        self.log_dict({"MSEloss": loss})

        print(loss)
        return loss

    def configure_optimizers(self):
        params = list(self.cnn.parameters())
        optimizer = torch.optim.Adam(params, **self.optimizer_config)
        return optimizer
