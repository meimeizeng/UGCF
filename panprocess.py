import torch.nn as nn
from senet import se_resnet
import torch



class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        # 这一行千万不要忘记
        super(DepthWiseConv, self).__init__()

        # 逐通道卷积
        self.depth_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=in_channel,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1,
                                    groups=in_channel)
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积

        # 逐点卷积
        self.point_conv = nn.Conv2d(in_channels=in_channel,
                                    out_channels=out_channel,
                                    kernel_size=1,
                                    stride=1,
                                    padding=0,
                                    groups=1)

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class panprocess(nn.Module):
    def __init__(self):
        super(panprocess, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1,
                               padding=1, bias=True)
        self.depth1 = DepthWiseConv(8, 1)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()




    def forward(self, xin):
        x1 = self.conv1(xin)
        x1 = self.relu(x1)
        x2 = self.depth1(x1)
        x2 = self.tanh(x2)

        x3 = self.conv1(x2)
        x3 = self.relu(x3)
        x4 = self.depth1(x3)
        x4 = self.tanh(x4)
        x5 = self.pool(x4).squeeze(-1).squeeze(-1)

        return x5


class spectralnet(nn.Module):
    def __init__(self):
        model = se_resnet.se_resnet18(num_classes=1)
        # 64*64,但是这样就不能随意切换尺寸了
        super(spectralnet, self).__init__()
        model.conv1=nn.Conv2d(8,64,kernel_size=3,padding=1)
        ######### 8 64 变成 8 8 
        self.enc1 = nn.Sequential(model)
    def forward(self, x1):
        enc1 = self.enc1(x1)
        return enc1

##上面这两个输出放到正则项里就vans


class SpatialAttentionBlock(nn.Module):
    def __init__(self,channel):
        super(SpatialAttentionBlock, self).__init__()
         # Maximum pooling
        self.featureMap_max = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.MaxPool2d(kernel_size=(5, 5), stride=(1, 1),padding=0)
        )
        # Average pooling
        self.featureMap_avg = nn.Sequential(
            nn.ReflectionPad2d(2),
            nn.AvgPool2d(kernel_size=(5, 5), stride=(1,1), padding=0)
        )

        # Deviation pooling
        # var = \sqrt(featureMap - featureMap_avg)^2

        # Dimensionality Reduction
        self.reduce_dim = nn.Sequential(
            nn.Conv2d(in_channels=channel * 4, out_channels=channel, kernel_size=(3,3), stride=(1, 1), padding=1,bias=False),
            nn.Conv2d(in_channels=channel,out_channels=channel,kernel_size=(1,1),stride=(1,1),bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):
        x_max = self.featureMap_max(x)
        x_avg = self.featureMap_avg(x)
        x_var = torch.sqrt(torch.pow(x - x_avg,2) + 1e-7)

        y = torch.cat([x_max,x_avg,x_var,x],dim=1)
        z = self.reduce_dim(y)
        return x * z
#channel是8

#注意力机制

class ChannelAttentionBlock(nn.Module):
    def __init__(self, channel, reduction=4):
        super(ChannelAttentionBlock, self).__init__()
        self.reduction = reduction
        self.dct_layer = nn.AdaptiveAvgPool2d(1)# DCTLayer(channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            #先放缩
            nn.ReLU(inplace=True),
            #激活
            nn.Linear(channel // reduction, channel, bias=False),
            #再放缩
            nn.Sigmoid()
            #再激活
        )

    def forward(self,x):
        n,c,h,w = x.size()
        y = self.dct_layer(x).squeeze(-1).squeeze(-1)
        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


#叠加法，小手段
