import lightning as L
import torch
from copy import deepcopy
from awave2.dwt.fdata import ForwardDataDWT


class CNNWave(L.LightningModule):
    def __init__(self, cnn, loss, attributer=None, dwt=None, dwts=None, verbose=False):
        super().__init__()
        self.cnn = cnn
        self.same_wt = dwt is not None

        assert (dwt is None) != (dwts is None), "either dwt or dwts must be provided"

        self.dwt = dwt
        self.dwts = dwts
        if not self.same_wt:
            self.n_dwts = len(dwts)

        self.attributer = attributer
        self.loss = loss
        self.verbose = verbose

    def training_step(self, batch, batch_idx):
        if self.verbose:
            print("training step")
        x, y = batch

        if self.verbose:
            print(x.shape)
            print("forward transform")

        if self.same_wt:
            x_coeffs = self.dwt(x)
        else:
            x_coeffs = [
                self.dwts[i](x[:, i, :, :].unsqueeze(1)) for i in range(self.n_dwts)
            ]

        if self.verbose:
            print("computing reconstruction")
        if self.same_wt:
            x_recon = self.dwt.inverse(x_coeffs)
        else:
            x_recons = [self.dwts[i].inverse(x_coeffs[i]) for i in range(self.n_dwts)]
            x_recon = torch.cat(x_recons, dim=1)

        if self.verbose:
            print("computing residual")
        x_resid = deepcopy(x) - x_recon.detach()
        y_recon = self.cnn(x_recon + x_resid)

        sample = 100

        # flatten to 2D (batch x output)
        y_flat = torch.flatten(y_recon, start_dim=1, end_dim=-1)
        # subsample outputs if requested
        n_outputs = y_flat.shape[1]
        if sample is not None:
            y_flat = y_flat[:, torch.randint(0, n_outputs, (sample,))]

        if self.attributer is not None:
            if self.verbose:
                print("constructing fdata, computing attrs")
            if self.same_wt:
                fdata = ForwardDataDWT(x_coeffs, y_flat)
                attrs = self.attributer(fdata)
            else:
                fdatas = [
                    ForwardDataDWT(x_coeffs[i], y_flat) for i in range(self.n_dwts)
                ]
                attrs = [self.attributer(fdata) for fdata in fdatas]
                attrs = sum(attrs, [])
        else:
            attrs = []

        if self.verbose:
            print("computing loss")
        if self.same_wt:
            loss = self.loss(x, x_recon, wt=self.dwt, attrs=attrs)
        else:
            loss = self.loss(x, x_recon, wts=self.dwts, attrs=attrs)
        self.log_dict(loss)

        return loss["loss"]

    def configure_optimizers(self):
        if self.same_wt:
            params = list(self.dwt.parameters())
        else:
            params = sum([list(wt.parameters()) for wt in self.dwts], [])
        optimizer = torch.optim.Adam(params, lr=1e-3)
        return optimizer
