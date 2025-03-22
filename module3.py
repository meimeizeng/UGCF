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


def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride == 1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
batch_size = 4
n_feats = 32
n_layer = 8 # 8

class REAM(nn.Module):
    def __init__(self,
                 in_channels,
                 mid_channels,
                 out_channels,
                 kernel_size=3):
        super(REAM, self).__init__()
        p = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size, padding=p, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size, padding=p, bias=False)
        self.relu = nn.PReLU()
        self.ca = CustomModule(out_channels)

    def forward(self, x):
        res = self.ca(self.conv2(self.relu(self.conv1(x))))
        res += x
        return res


class HBlock(nn.Module):
    def __init__(self,
                 ms_channels,
                 n_feats,
                 kernel_size=3):
        super(HBlock, self).__init__()
        self.convF = nn.Conv2d(in_channels=n_feats, out_channels=ms_channels, kernel_size=3, stride=1, padding=1,bias=False)
        self.eta = nn.Parameter(
            nn.init.normal_(
                torch.empty(1).cuda(), mean=0.1, std=0.3
            ))
        self.prox = REAM(ms_channels, n_feats, ms_channels, kernel_size)
    def forward(self, HR, S, D):
        E = HR - S - self.convF(D)
        R = HR - self.eta * E
        HR = self.prox(R)
        return HR


class CustomModule(nn.Module):
    def __init__(self, in_channels):
        super(CustomModule, self).__init__()
        self.in_channels = in_channels
        self.branch1_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.branch1_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        self.branch2_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.branch2_conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.ac=nn.Sigmoid()

        self.final_conv = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)

    def forward(self, x):
        # Branch 1
        branch1 = F.relu(self.branch1_conv1(x))
        branch1 = self.branch1_conv2(branch1)

        # Branch 2
        branch2 = F.relu(self.branch2_conv1(x))
        branch2 = self.branch2_conv2(branch2)

        # Concatenate along the channel dimension
        concat = torch.cat((branch1, branch2), dim=1)
        concat=self.ac(concat)
        # Reduce the concatenated features back to the original number of channels
        reduced = self.final_conv(concat)

        # Global average pooling
        gap = F.adaptive_avg_pool2d(reduced, (1, 1))

        # Element-wise multiplication with the original input
        output = x * gap

        return output


class ZEA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ZEA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, f):
        x = f
        c1_ = (self.conv1(f))
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        # c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear')
        c3 = F.interpolate(c3, size=[x.size(2), x.size(3)], mode='bicubic', align_corners=True)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)
        return x * m


class M3DN11(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feats,
                 n_layer):
        super(M3DN11, self).__init__()
        self.D_blocks = nn.ModuleList([DBlock(ms_channels, n_feats, 3) for i in range(n_layer)])
        self.H_blocks = nn.ModuleList([HBlock(ms_channels, n_feats, 3) for i in range(n_layer)])
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
        for i in range(len(self.D_blocks)):
            D = self.D_blocks[i](HR, S, D)
            HR = self.H_blocks[i](HR, S, D)

        return HR


if __name__ == "__main__":
    ms = torch.randn(2, 8, 32, 32).cuda()
    pan= torch.randn(2, 1, 64, 64).cuda()
    #输出和输入尺寸一样
    ms_channels,num=8,8
    H,W=64,64
    HR=F.upsample(ms, [H, W]).cuda()
    S = F.upsample(ms, [H, W]).cuda()
    D = torch.randn(2, 1, 64, 64)
    D = D.expand(-1, ms_channels, -1, -1).cuda()
    D1 = nn.Conv2d(in_channels=ms_channels, out_channels=n_feats, kernel_size=3, stride=1, padding=1).cuda()
    D = (D1(D - HR)).cuda()
    model1=DBlock(8,32,3).cuda()
    model2=HBlock(8,32,3).cuda()

   # model = CustomModule(in_channels=num)
    D=model1(HR,S,D)
    HR=model2(HR,S,D)
    model3=M3DN11(ms_channels=8,pan_channels=1,n_feats=32,n_layer=4).cuda()
    output_tensor=model3(ms,pan)
    ##和模块的输入不一样
    print(output_tensor.shape)


##会保持过程尺寸不变，REAM最后的layer需要和输入x的通道保持一致。