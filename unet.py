import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, n_channels_in, n_channels_out):
        super(UNet, self).__init__()

        self.input = nn.Sequential(
            nn.Conv2d(n_channels_in, 8, kernel_size=3, padding='same', bias=False),
            nn.ReLU(inplace=True),
        )
        self.down1 = Down(8, 16)
        self.down2 = Down(16, 32)
        self.down3 = Down(32, 64)
        self.up1 = Up(64, 32)
        self.up2 = Up(32, 16)
        self.up3 = Up(16, n_channels_out)

    def forward(self, x):
        # print(x.shape)
        x0 = self.input(x)
        # print(x0.shape)
        x1 = self.down1(x0)
        # print(x1.shape)
        x2 = self.down2(x1)
        # print(x2.shape)
        x3 = self.down3(x2)
        # print(x3.shape)
        x4 = self.up1(x3, x2)
        # print(x4.shape)
        x5 = self.up2(x4, x1)
        # print(x5.shape)
        x6 = self.up3(x5, x0)
        # print(x6.shape)
        return x6


# this class will do a 2x downsample
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):

        return self.seq(x)

# this class will do a 2x upsample
class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, stride=2, kernel_size=2, bias=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        #print(x1.shape)
        #print(x2.shape)
        return self.conv(torch.cat([x1, x2], dim=1))