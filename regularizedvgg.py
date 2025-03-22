import torchvision
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

pthfile = ''#your pretrained VGG file
model1 = torchvision.models.vgg19()

model1.load_state_dict(torch.load(pthfile))


feature = list(model1.features.children())
features = (model1.features.children())
#还得改输入通道
#print(feature)#引用feature中的childern设置

class vggfeature(nn.Module):
    def __init__(self):
        super(vggfeature, self).__init__()

        model1.features[0]=nn.Conv2d(8,64,kernel_size=3,padding=1)
        #######################输入为4或者8
        self.enc1 = nn.Sequential(*features)
        self.pixel = nn.PixelShuffle(4)
        self.c1=nn.Conv2d(8, 1, kernel_size=1)
    def forward(self, x1,x2,x3):#改成3输入结构，都是7*7
        enc1 =self.enc1(x1)
        enc1 = self.pixel(enc1)

        
        enc2 =self.enc1(x2)
        enc2 = self.pixel(enc2)
        enc3=self.enc1(x3)
        enc3 = self.pixel(enc3)
        #print(enc1.shape)
        
        return enc1,enc2,enc3
        
        # 对于模型的每个权重，使其不进行反向传播，即固定参数

    for param in model1.parameters():
        param.requires_grad = False
    # 将分类器的最后层输出维度换成了num_cls，这一层需要重新学习
    for param in model1.features[0].parameters():
        param.requires_grad = True

class vggsmall(nn.Module):
    def __init__(self):
        super(vggsmall, self).__init__()

        f4[0]=nn.Conv2d(4,64,kernel_size=3,padding=1)
        self.enc1 = nn.Sequential(*f4)
        self.c1=nn.Conv2d(128,4,1)
        self.c2=f3
        self.pixel = nn.PixelShuffle(4)
    def forward(self, x1):#改成3输入结构，都是7*7
        e1 = self.enc1(x1)
        e1= self.c1(e1)
        e1= self.c2(e1)


        return e1
        # 对于模型的每个权重，使其不进行反向传播，即固定参数

    for param in f2.parameters():
        param.requires_grad = False
    # 将分类器的第一次层输出维度换成了num_cls，这一层需要重新学习
    for param in f2[0].parameters():
        param.requires_grad = True
