import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, n_channels=2, n_channels_out=1, l1=64, l2=5, n_hidden_layers=7):
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels=n_channels, out_channels=l1, kernel_size=l2, padding="same"
        )  # Input layer

        self.conv_hidden = []
        for i in range(n_hidden_layers):
            self.conv_hidden.append(
                nn.Conv2d(
                    in_channels=l1, out_channels=l1, kernel_size=l2, padding="same"
                )
            )
        self.conv_hidden = nn.ModuleList(self.conv_hidden)

        self.conv2 = nn.Conv2d(
            in_channels=l1, out_channels=n_channels_out, kernel_size=l2, padding="same"
        )  # Output layer

    def forward(self, x):
        x0 = F.relu(self.conv1(x))

        x_prev = x0
        for conv in self.conv_hidden:
            x_prev = F.relu(conv(x_prev))

        xo = self.conv2(x_prev)
        return xo
