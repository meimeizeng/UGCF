import torch
import torch.nn as nn
import torch.nn.functional as F
import panprocess
# 从se_resnet.py中导入SELayer
from senet.se_resnet import SELayer

# 定义梯度反转层
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None

class GradientReversal(nn.Module):
    def __init__(self, lambda_=0.010):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalLayer.apply(x, self.lambda_)

class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.channel = panprocess.ChannelAttentionBlock(8)#
    def forward(self, x,y):
        batch_size, C, H, W = x.size()
        query = self.query(y).view(batch_size, C, -1)
        key = self.key(x).view(batch_size, C, -1)
        value = self.value(x).view(batch_size, C, -1)

        attention = torch.bmm(query.permute(0, 2, 1), key)
        attention = self.softmax(attention)
        attention1 = self.channel(x)
        out = torch.bmm(attention, value.permute(0, 2, 1))
        out1 = torch.bmm(attention1, value.permute(0, 2, 1))
        out = out+out1
        out = out.permute(0, 2, 1).view(batch_size, C, H, W)

        return out

class FFTConvAttentionModel(nn.Module):
    def __init__(self, in_channels):
        super(FFTConvAttentionModel, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv5x5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        self.avg_pool = nn.AvgPool2d(kernel_size=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.attention = Attention(in_channels)
        self.grad_reverse = GradientReversal()
        self.se_layer = SELayer(in_channels)  # 使用SELayer调整通道数

    def process_feature(self, real, imag):
        real_conv = self.conv3x3(real) + self.conv5x5(real)
        imag_conv = self.conv3x3(imag) + self.conv5x5(imag)

        real_conv = self.avg_pool(real_conv) + self.max_pool(real_conv)
        imag_conv = self.avg_pool(imag_conv) + self.max_pool(imag_conv)

        real_conv = torch.sigmoid(real_conv)
        imag_conv = torch.sigmoid(imag_conv)

        real_attended = self.attention(real_conv,imag_conv)
        imag_attended = self.attention(imag_conv,real_conv)

        real_reversed_grad = self.grad_reverse(real_attended)
        imag_reversed_grad = self.grad_reverse(imag_attended)

        complex_reversed_grad = torch.complex(real_reversed_grad, imag_reversed_grad)

        return complex_reversed_grad

    def forward(self, img1, img2):
        # 检查并调整输入维度
        if img1.shape[1] == 8:
            img1 = self.se_layer(img1)
        if img2.shape[1] == 8:
            img2 = self.se_layer(img2)

        fft_img1 = torch.fft.fft2(img1)
        fft_img2 = torch.fft.fft2(img2)

        real_img1, imag_img1 = fft_img1.real, fft_img1.imag
        real_img2, imag_img2 = fft_img2.real, fft_img2.imag

        processed_img1 = self.process_feature(real_img1, imag_img1)
        processed_img2 = self.process_feature(real_img2, imag_img2)

        ifft_result1 = torch.fft.irfft2(processed_img1)
        ifft_result2 = torch.fft.irfft2(processed_img2)

        return ifft_result1, ifft_result2
'''
# 示例用法
batch_size = 1
img1 = torch.randn(batch_size, 8, 256, 256)
img2 = torch.randn(batch_size, 1, 256,256)

# 将 img2 扩展到与 img1 相同的通道数
img2_expanded = img2.expand(batch_size, 8, 256, 256)

model = FFTConvAttentionModel(in_channels=8)

output1, output2 = model(img1, img2_expanded)

print("Output shape 1:", output1.shape)
print("Output shape 2:", output2.shape)
'''



