import torch
import torch.nn as nn


class SobelFilter(nn.Module):
    def __init__(self):
        super(SobelFilter, self).__init__()
        self.conv_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)
        self.conv_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3, padding=1, bias=False)

        # Sobel filters for horizontal and vertical gradients
        self.conv_horizontal.weight.data = torch.FloatTensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]])
        self.conv_vertical.weight.data = torch.FloatTensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]])

    def forward(self, x):
        # x shape: (batch_size, input_channels, H, W)
        # Convert to grayscale if input_channels > 1
        if x.size(1) > 1:
            x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]  # RGB to grayscale
            x_gray = x_gray.unsqueeze(1)  # shape: (batch_size, 1, H, W)
        else:
            x_gray = x

        # Apply Sobel filters
        gradient_h = self.conv_horizontal(x_gray)
        gradient_v = self.conv_vertical(x_gray)

        # Compute magnitude of gradients
        gradient_mag = torch.sqrt(gradient_h ** 2 + gradient_v ** 2)

        return gradient_mag

'''
# 测试示例
sobel = SobelFilter()
# 创建一个虚拟输入，形状为 (1, 8, 64, 64)
dummy_input = torch.randn(9, 32, 8,8)
'''
# 运行模型
#output = sobel(dummy_input)

# 输出形状
#print("Output shape:", output.shape)
#无论输入多少通道，最终输出为1*1*64*64
class ImageEncoder(nn.Module):
    def __init__(self, input_channels):
        super(ImageEncoder, self).__init__()
        height=32
        width=32
        # Define convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        # Define pooling layer
        self.pool = nn.MaxPool2d(kernel_size=1, stride=2)###################kernel_size太大，图尺寸太小

        # Define activation function
        self.activation = nn.ReLU()

        # Fully connected layer for variance calculation
        self.fc_variance = nn.Linear(128 * (height // 2) * (width // 2), 1)

    def forward(self, x):
        # Apply convolutional layers
        x = self.conv1(x)
        x = self.activation(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.activation(x)
        x = self.pool(x)

        # Flatten the output for fully connected layer
        x = torch.flatten(x, 1)

        # Calculate variance in the feature space
        variance = torch.var(x, dim=1).mean()  # Calculate variance along each batch

        return variance
'''
encoder = ImageEncoder(input_channels=1)  # Assuming input_channels is the number of channels in Sobel output
##这里是sobel的输出

# Pass Sobel output through the encoder
variance = encoder(sobel(dummy_input))

# Print or use the variance
print("Variance in feature space:", variance)
'''