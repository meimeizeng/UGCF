import sys
sys.path.append('/path/to/attention')

import torch.nn as nn
import torch.nn.functional as F
import torch

class Down(nn.Module):
    def __init__(self):
        super(Down, self).__init__()
        self.max_pool_conv = nn.MaxPool2d(2)

    def forward(self, x):
        return self.max_pool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)

class UP(nn.Module):
    def __init__(self, in_channels, out_channels, bicubic=True):
        super(UP, self).__init__()
        self.bicubic = bicubic
        if bicubic:
            self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        if self.bicubic:
            out = self.conv(self.up(x))
        else:
            out = self.up(x)
        return out

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Conv, self).__init__()
        self.conv = torch.nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)


class DBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 n_feats,
                 kernel_size=3):
        super(DBlock, self).__init__()
        reduction = ms_channels // 2
        self.convF = nn.Conv2d(in_channels=n_feats, out_channels=ms_channels, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.convFT = nn.Conv2d(in_channels=ms_channels, out_channels=n_feats, kernel_size=3, stride=1, padding=1,
                                bias=False)
        channel_input1 = n_feats
        channel_output1 = n_feats
        self.conv1 = Conv(channel_input1, channel_output1)
        self.down1 = Down()
        channel_input2 = channel_output1
        channel_output2 = channel_output1 // 2
        self.conv2 = Conv(channel_input2, channel_output2)
        self.down2 = Down()
        channel_input3 = channel_output2
        channel_output3 = channel_input3
        self.conv3 = Conv(channel_input3, channel_output3)
        self.up1 = UP(channel_output3, channel_output3, bicubic=False)
        channel_input4 = channel_output3
        channel_output4 = channel_output1 // 2
        self.conv4 = Conv(channel_input4, channel_output4)
        self.up2 = UP(channel_output4, channel_output1, bicubic=False)
        channel_input5 = channel_output1
        channel_output5 = n_feats
        self.conv5 = Conv(channel_input5, channel_output5)
        self.conv6 = OutConv(channel_output5, n_feats)
        self.rho = nn.Parameter(
            nn.init.normal_(
                torch.empty(1).cuda(), mean=0.1, std=0.3
            ))

    def forward(self, HR, S, D):
        r = D - self.rho * self.convFT(self.convF(D) - (HR - S))
        x1 = self.conv1(r)
        x2 = self.down1(x1)
        x2 = self.conv2(x2)
        x3 = self.down2(x2)
        x3 = self.conv3(x3)
        x4 = self.up1(x3)
        x4 = self.conv4(x4 + x2)
        x5 = self.up2(x4)
        x5 = self.conv5(x5 + x1)
        x6 = self.conv6(x5)
        D = x6 + r
        return D




class UNet(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feats,
                 n_layer):
        super(UNet, self).__init__()
        self.D_blocks = DBlock(ms_channels, n_feats, 3)
        self.convert = nn.Conv2d(in_channels=ms_channels, out_channels=n_feats, kernel_size=3, stride=1, padding=1)

    def forward(self, ms, pan=None):
        if type(pan) == torch.Tensor:
            pass
        elif pan == None:
            raise Exception('User does not provide pan image!')
        N, ms_channels, h, w = ms.shape
        N, pan_channels, H, W = pan.shape
        HR = F.upsample(ms, [H, W])
        S = F.upsample(ms, [H, W])
        D = pan
        D = D.expand(-1, ms_channels, -1, -1)
        D = self.convert(D - HR)
        HR = self.D_blocks(HR, S, D)
        HR = self.D_blocks(HR, S, D)
        HR = self.D_blocks(HR, S, D)


        return HR
