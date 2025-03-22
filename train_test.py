import torch
import torch.nn as nn
import sys

import numpy as np
import os
import time
import datetime
from torch.utils.data import DataLoader
sys.path.append("..")

from models.module3 import UNet
from contrasivemodule import Model
from capsulevae  import ImageEncoder,SobelFilter
from models import get_sat_param
from fftwithsenet import FFTConvAttentionModel
from metrics import get_metrics_reduced
from utils import PSH5Dataset,save_param, psnr_loss, ssim
from HREM import HREM
#超参数设置
input_channels = 8  # 输入图像的通道数
ms_channels=8

latent_dim = 5  # 潜在空间的维度
mask_percentage = 0.25  # 掩码百分比
num_masks = 5  # 掩码操作的次数
num_epochs = 600
lr = 1e-4
weight_decay = 0
batch_size = 16
n_feats = 4
n_layer = 4  # 8
model_str = 'M3DNew'
satellite_str = 'WorldView3'  # 'WorldView2'  #'Quickbird'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class UGCF(nn.Module):
    def __init__(self,
                 ms_channels,
                 pan_channels,
                 n_feats,
                 n_layer):
        super(UGCF, self).__init__()
        self.cmcm = FFTConvAttentionModel(ms_channels)
        self.encoder = ImageEncoder(input_channels=1)
        ##这里是sobel的输出
        self.sobel = SobelFilter()
        ##传进来是什么就是什么
        #self.c1 = Encoder(8, latent_dim, mask_percentage)#
        #self.c2 = Encoder(8, latent_dim, mask_percentage)#
        self.unet = UNet(8,#4
                             pan_channels,
                             n_feats,
                             n_layer)
        self.hrem= HREM(ms_channels,ms_channels)
        self.cfem= Model()
        self.conv1 = nn.Conv2d(32, 8, kernel_size=1)#
        self.conv2 = nn.Conv2d(8, 8, kernel_size=1)#

    def forward(self, x, y, gt,ms):  # 这里面x= lms，y=pan
        HR = self.unet(x, y)
        HR = self.hrem(HR)

        batch_size, num, H, W = HR.size()
        d1, d2 = self.cmcm(HR, y.expand(batch_size, num, H, W))

        x1,x2,x3 = self.cfem(HR,ms,gt)

        con1=torch.add(x1,x2)
        con1=torch.add(con1,x3)
        con2=torch.add(d1,d2)

        con1=self.conv1(con1)
        con2=self.conv2(con2)

        vae1 = self.c1(con1).abs().mean()
        vae2 = self.c2(con2).abs().mean()

        v1 = self.encoder(self.sobel(con1))
        v2 = self.encoder(self.sobel(con2))
        data1 = np.array(((vae1 + v1).item(), (vae2 + v2).item()))
        # standardized_data = standardize(data1)
        standardized_data = data1 / (abs(data1[0]) + abs(data1[1]))

        #融合图像，正样本，负样本

        return HR, d1, d2,x1,x2,x3,standardized_data



# . Get the parameters of your satellite
sat_param = get_sat_param(satellite_str)
if sat_param != None:
    ms_channels, pan_channels, scale = sat_param
else:
    print('You should specify `ms_channels`, `pan_channels` and `scale`! ')
    ms_channels = 8
    pan_channels = 1
    scale = 4

net = UGCF(8,
           pan_channels,
           n_feats,
           n_layer).cuda()


# . Get your optimizer, scheduler and loss function
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
loss_fn = nn.L1Loss().to(device)

# . Create your data loaders
prepare_data_flag = False  # set it to False, if you have prepared dataset
# train_path      = '%s_train_64.h5' % (satellite_str)
train_path = 'train_wv3.h5'
validation_path = 'valid_wv3.h5'
test_path = 'test_wv3_multiExm1.h5'

trainloader = DataLoader(PSH5Dataset(train_path),
                         batch_size=batch_size)  # [N,C,K,H,W]
validationloader = DataLoader(PSH5Dataset(validation_path),
                              batch_size=1)
testloader = DataLoader(PSH5Dataset(test_path),
                        batch_size=1)

loader = {'train': trainloader,
          'validation': validationloader}




save_path = os.path.join(
    'logs/%s' % (model_str))
file_name = 'best_net.pth'
full_path = os.path.join(save_path, '/best_net.pth')
file1_name = 'last_net.pth'
full1_path = os.path.join(save_path, '/last_net.pth')

# writer = SummaryWriter(save_path)
params = {'model': model_str,
          'satellite': satellite_str,
          'epoch': num_epochs,
          'lr': lr,
          'batch_size': batch_size,
          'n_feats': n_feats,
          'n_layer': n_layer}
# directory=os.path.join(save_path, 'param.json')
# os.mkdirs(directory,exist_ok=True)
save_param(params, save_path)

step = 0
best_psnr_val, psnr_val, ssim_val = 0., 0., 0.
# torch.backends.cudnn.benchmark = True
prev_time = time.time()




net.load_state_dict(torch.load('lastqbguide.pth')['net'])
optimizer.load_state_dict(torch.load('lastqbguide.pth')['optimizer'])


for epoch in range(num_epochs):

    epoch_loss_train = 0.
    epoch_loss_val = 0.
    total = 0


    for i, (ms, pan, gt) in enumerate(loader['train']):
        # 0. preprocess data
        ms, pan, gt, = ms.cuda(), pan.cuda(), gt.cuda()
        criterion2 = nn.MSELoss()
        # 1. update
        net.train()
        net.zero_grad()
        optimizer.zero_grad()
        n1=nn.Upsample(scale_factor=4, mode='bilinear')
        m1=n1(ms)
        imgf, d1, d2,x1,x2,x3,standardized_data = net(ms, pan,gt,m1)
        coef1 = 10e-2
        # d1是sr，d2是gt，d3是binary
        goodone = criterion2(x1, x2)
        badone = criterion2(x1, x3)

        c1,c2=standardized_data[1],standardized_data[0]

        loss1 = loss_fn(gt, imgf)
        loss2 = criterion2(d1, d2)
        loss3=coef1 * max(goodone - badone + 10e-5, 0)
        loss = loss1 + c1*loss2+c2*loss3

        ##这里加不加loss2是说loss2即跨模态部分是否起作用
        loss.backward()
        optimizer.step()

        total += ms.size(0)
        epoch_loss_train += loss.item()

        # 2. print
        # Determine approximate time left
        batches_done = epoch * len(loader['train']) + i
        batches_left = num_epochs * len(loader['train']) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [loss: %f] [PSNR/Best: %.4f/%.4f] ETA: %s"
            % (
                epoch,
                num_epochs,
                i,
                len(loader['train']),
                loss.item(),
                psnr_val,
                best_psnr_val,
                time_left,
            )
        )

        # 3. Log the scalar values
        # writer.add_scalar('loss', loss.item(), step)
        # writer.add_scalar('learning rate', optimizer.state_dict()['param_groups'][0]['lr'], step)
        step += 1

    epoch_loss_train /= total
    # 这里加了一个数防止除以0
    # writer.add_scalar('loss_train', epoch_loss_train, epoch)

    # validation
    current_psnr_val = psnr_val
    psnr_val = 0.
    ssim_val = 0.
    loss_val = 0.
    metrics = torch.zeros(5, validationloader.__len__())
    with torch.no_grad():
        net.eval()
        for i, (ms, pan, gt) in enumerate(loader['validation']):
            c1 = nn.MSELoss()
            ms = ms.cuda()
            pan = pan.cuda()
            gt = gt.cuda()
            n1=nn.Upsample(scale_factor=4, mode='bilinear')
            m1=n1(ms)
            imgf, d1, d2,x1,x2,x3,standardized_data = net(ms, pan,gt,m1)
            loss_val += loss_fn(gt, imgf) + 0.05 * c1(d1, d2)
            psnr_val += psnr_loss(imgf, gt, 1.)
            ssim_val += ssim(imgf, gt, 11, 'mean', 1.)
        psnr_val = float(psnr_val / loader['validation'].__len__())
        ssim_val = float(ssim_val / loader['validation'].__len__())
        loss_val = float(loss_val / loader['validation'].__len__())
        metrics[:, i] = torch.Tensor(get_metrics_reduced(imgf, gt))[:5]
        psnr_val, ssim_val, SCC, SAM, ERGAS = metrics.mean(dim=1)
    print(psnr_val, ssim_val, SCC, SAM, ERGAS)
    # writer.add_scalar('PSNR/val', psnr_val, epoch)
    # writer.add_scalar('SSIM/val', ssim_val, epoch)
    # print(ssim_val,psnr_val)

    psnr_val = 0.
    ssim_val = 0.
    metrics = torch.zeros(5, testloader.__len__())
    with torch.no_grad():
        net.eval()
        for i, (ms, pan, gt) in enumerate(testloader):
            ms = ms.cuda()
            pan = pan.cuda()
            gt = gt.cuda()
            n1=nn.Upsample(scale_factor=4, mode='bilinear')
            m1=n1(ms)
            #print()
            imgf, d1, d2,x1,x2,x3,standardized_data = net(ms, pan,gt,m1)
            metrics[:, i] = torch.Tensor(get_metrics_reduced(imgf, gt))[:5]
        psnr_val, ssim_val, SCC, SAM, ERGAS = metrics.mean(dim=1)
        print(psnr_val, ssim_val, SCC, SAM, ERGAS)
    scheduler.step(loss_val)

    # visualize the step parameter rho and eta

    # Save the best weight  and  early stopping
    if best_psnr_val < psnr_val:
        best_psnr_val = psnr_val
        # count_no = 0
        torch.save({'net': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch},
                   'wv3guide.pth')  # _use_new_zipfile_serialization=False

    # Save the current weight
    torch.save({'net': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch},
               'lastwv3guide.pth')

    # # early stopping
    # if acc_valid <= best_psnr_val:
    #     count_no += 1
    #     if count_no > count_max:
    #         break
    # else:
    #     count_no = 0
    #     best_acc = acc_v4alid




