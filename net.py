import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_channels=2, n_channels_out=1, l1=64, l2=5):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=n_channels, out_channels=l1, kernel_size=l2, padding="same"
        )  # Input layer
        self.conv2 = nn.Conv2d(
            in_channels=l1, out_channels=n_channels_out, kernel_size=l2, padding="same"
        )  # Output layer
        self.conv31 = nn.Conv2d(
            in_channels=l1, out_channels=l1, kernel_size=l2, padding="same"
        )  # Middle layers
        self.conv32 = nn.Conv2d(
            in_channels=l1, out_channels=l1, kernel_size=l2, padding="same"
        )  # Middle layers
        self.conv33 = nn.Conv2d(
            in_channels=l1, out_channels=l1, kernel_size=l2, padding="same"
        )  # Middle layers
        self.conv34 = nn.Conv2d(
            in_channels=l1, out_channels=l1, kernel_size=l2, padding="same"
        )  # Middle layers
        self.conv35 = nn.Conv2d(
            in_channels=l1, out_channels=l1, kernel_size=l2, padding="same"
        )  # Middle layers
        self.conv36 = nn.Conv2d(
            in_channels=l1, out_channels=l1, kernel_size=l2, padding="same"
        )  # Middle layers
        self.conv37 = nn.Conv2d(
            in_channels=l1, out_channels=l1, kernel_size=l2, padding="same"
        )  # Middle layers

    def forward(self, x):
        x0 = F.relu(self.conv1(x))
        x1 = F.relu(self.conv31(x0))
        x2 = F.relu(self.conv32(x1))
        x3 = F.relu(self.conv33(x2))
        x4 = F.relu(self.conv34(x3))
        x5 = F.relu(self.conv35(x4))
        x6 = F.relu(self.conv36(x5))
        x7 = F.relu(self.conv37(x6))

        xo = self.conv2(x7)
        return xo
