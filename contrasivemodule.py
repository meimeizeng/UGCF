import torch
import torch.nn as nn
import torchvision.models as models
import random
from regularizedvgg import vggfeature

class RandomAugmentation(nn.Module):
    def __init__(self, prob=0.5):
        super(RandomAugmentation, self).__init__()
        self.prob = prob

    def random_mask(self, x):
        mask = torch.rand_like(x) > 0.5
        return x * mask

    def random_rotate(self, x):
        k = random.randint(0, 3)  # Randomly choose 0, 1, 2, or 3
        return torch.rot90(x, k, [2, 3])

    def add_noise(self, x):
        noise = torch.randn_like(x) * 0.1
        return x + noise

    def forward(self, x):
        if random.random() < self.prob:
            transform = random.choice([self.random_mask, self.random_rotate, self.add_noise])
            x = transform(x)
        return x

class Aug(nn.Module):
    def __init__(self):
        super(Aug, self).__init__()

    def random_mask(self, x):
        mask = torch.rand_like(x)
        return x * mask






class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.features = nn.Sequential(*list(vgg.features.children())[:-1])

    def forward(self, x):
        return self.features(x)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.augmentation = RandomAugmentation()
        self.aug=Aug()
        self.feature_extractor = vggfeature()
        self.conv1= nn.Conv2d(64,64,kernel_size=1)
        self.sig=nn.Sigmoid()

    def forward(self, x1, x2, x3):
        x1_aug = self.sig(self.conv1(self.augmentation(x1)))
        x2=self.sig(self.conv1(self.aug(x2)))
        x3=self.sig(self.conv1(self.aug(x3)))
        f1,f2,f3 = self.feature_extractor(x1_aug,x2,x3)

        return f1, f2, f3


# Example usage
if __name__ == "__main__":
    batch_size = 4
    channels = 8  # Number of channels in the input
    height = 64
    width = 64

    x1 = torch.randn(batch_size, channels, height, width)
    x2 = torch.randn(batch_size, channels, height, width)
    x3 = torch.randn(batch_size, channels, height, width)

    model = Model()
    f1, f2, f3 = model(x1, x2, x3)

    print(f1.shape)  # Should be (batch_size, 512, height/32, width/32) for VGG16
    print(f2.shape)
    print(f3.shape)
    

