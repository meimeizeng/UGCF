import torch
import torch.nn as nn
import torch.nn.functional as F


class Mask(nn.Module):
    def __init__(self, mask_percentage):
        super(Mask, self).__init__()
        self.mask_percentage = mask_percentage

    def forward(self, x):
        if not self.training:
            return x
        mask = torch.rand_like(x) < self.mask_percentage
        x = x * mask.float()
        return x


class Encoder(nn.Module):
    def __init__(self, in_channels, latent_dim, mask_percentage):
        super(Encoder, self).__init__()
        self.mask = Mask(mask_percentage)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc_mean = nn.Linear(64 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(64 * 8 * 8, latent_dim)

    def forward(self, x, num_masks=1):
        z_samples = []
        for _ in range(num_masks):
            masked_x = self.mask(x)
            masked_x = F.relu(self.conv1(masked_x))
            masked_x = F.relu(self.conv2(masked_x))
            masked_x = self.adaptive_pool(masked_x)
            masked_x = masked_x.view(masked_x.size(0), -1)
            mean = self.fc_mean(masked_x)
            logvar = self.fc_logvar(masked_x)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mean + eps * std
            z_samples.append(z)

        z_samples = torch.stack(z_samples, dim=0)  # (num_masks, batch_size, latent_dim)
        return z_samples  # 返回多次掩码和采样的潜在变量 z

'''
# 示例用法
input_channels = 1  # 输入图像的通道数
latent_dim = 20  # 潜在空间的维度
batch_size = 8  # 批量大小
mask_percentage = 0.25  # 掩码百分比
num_masks = 20  # 掩码操作的次数

# 创建 Encoder 实例
encoder = Encoder(input_channels, latent_dim, mask_percentage)

# 生成输入图像
input_image = torch.randn(batch_size, input_channels, 2, 2)  # 假设输入图像大小为 64x64

# 多次调用编码器以获得多个 z 值
z_samples = encoder(input_image, num_masks=num_masks)  # z_samples shape: (num_masks, batch_size, latent_dim)

# 计算每个维度的方差
variance =torch.mean( torch.var(z_samples, dim=(0))  )# 在维度0和1上计算方差，得到每个维度的方差

# 打印输出的方差值
print("Variance of sampled latent variables (per dimension):", variance)
'''