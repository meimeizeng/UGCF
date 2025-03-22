import torch
import torch.nn as nn
import torch.nn.functional as F


class HREM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HREM, self).__init__()

        # 上分支（1x1卷积 → 1x1卷积 → ReLU）
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

        # 下分支（3x3卷积 → ReLU → 3x3卷积）
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )

        # 特征融合（逐元素相加）
        self.conv1x1_merge = nn.Conv2d(out_channels, out_channels, kernel_size=1)

        # 特征提取分支（3x3卷积 → 最大池化）
        self.feature_extract = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 注意力机制（ReLU → Sigmoid）
        self.attention = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )

        # 上采样 + 1x1卷积
        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        branch1_out = self.branch1(x)
        branch2_out = self.branch2(x)

        # 逐元素相加
        fusion = branch1_out + branch2_out
        fusion = self.conv1x1_merge(fusion)

        # 经过特征提取分支
        extracted_features = self.feature_extract(fusion)
        attention_map = self.attention(extracted_features)

        # 上采样并通过1x1卷积
        attention_map = self.upsample(attention_map)

        # 逐元素相乘
        enhanced_features = fusion * attention_map

        return enhanced_features


# 测试
if __name__ == "__main__":
    model = HREM(in_channels=64, out_channels=64)
    x = torch.randn(1, 64, 128, 128)  # 假设输入特征图尺寸为 128x128
    y = model(x)
    print(y.shape)  # 输出的形状
