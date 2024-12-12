import lightning as L
import torch
from torch.nn import MSELoss


class CNN(L.LightningModule):
    def __init__(self, cnn, optimizer_config, verbose=False):
        super().__init__()
        self.save_hyperparameters()

        self.cnn = cnn

        self.verbose = verbose
        self.optimizer_config = optimizer_config
        self.loss = MSELoss()

    def forward(self, x):
        return self.cnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_cnn = self.forward(x)
        # MSE loss
        mse_loss = self.loss(y_cnn, y)
        loss = {"loss": mse_loss}

        self.log_dict(loss, on_epoch=True)

        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_cnn = self.forward(x)
        # MSE loss
        mse_loss = self.loss(y_cnn, y)
        self.log("val_loss", mse_loss, on_epoch=True)

    def configure_optimizers(self):
        params = self.cnn.parameters()
        optimizer = torch.optim.Adam(params, **self.optimizer_config)
        return optimizer
