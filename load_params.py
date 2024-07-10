import torch
import os

from torch.utils.data import DataLoader
from awave2.awd.attributer import SaliencyAttributer
from awave2.awd.loss import AWDLossConfig
from awave2.dwt.dwt1d import DWT1d, DWT1dConfig
from awave2.dwt.loss import DWTLossConfig
from net import Net
from cnnwave import CNNWave
from fdns_data import load_data

cnn = torch.load("trained-cnn/CNN.pt", map_location=torch.device("cpu"))
cnn.eval()


trainset, testset, input_normalized_val_torch, output_normalized_val_torch = load_data(
    None
)

trainloader = DataLoader(trainset, batch_size=10, shuffle=True, num_workers=1)
batch = next(iter(trainloader))

dwt_config = DWT1dConfig(
    dim_size=batch[0].shape[-1],
    dim=-1,
    init_wavelet="bior3.1",
    padding_mode="zero",
    learn_dual=True,
)

dwt = DWT1d(dwt_config)

attributer = SaliencyAttributer()

awd_loss_config = AWDLossConfig(wt_loss_config=DWTLossConfig())
awd_loss = awd_loss_config.get_loss_module()


print("loading")

path = "awd-dev/sa49qwqc/checkpoints/"
fname = os.listdir(path)[0]
print(path + fname)
cnnwave = CNNWave.load_from_checkpoint(
    path + fname, cnn=cnn, dwt=dwt, attributer=attributer, loss=awd_loss
)
print(cnnwave.dwt.h0, cnnwave.dwt.h1)
