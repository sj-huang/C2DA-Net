import sys
sys.path.append('../../Domain')
import itertools
import numpy as np
import torch
from data_load_train import read_train
from data_load_test import read_test
from util_acc_compute_Unet_GAN_128 import dice_loss, train, dice_batch_loss
from Unet2D_Unet_GAN_128 import *
from unet import UNet



'''
    下面四行是100*100读取，64批次读取，此时应该裁剪100*100大小的，但是不需要先扩大图像
    测试集仍然用原图
'''
train_data = read_train('../data/train')
test_data = read_test('../data/val')

cuda = True if torch.cuda.is_available() else False
criterion_GAN = torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
criterion_dice = dice_batch_loss()

discriminator = feature_dis()
discriminator_g1 = Discriminator_G()
discriminator_g2 = Discriminator_G()
Unet = UNet()
generator = GeneratorUNet()


# discriminator_g = torch.load("./save/D_G.pth")
# Unet = torch.load("./save/U.pth")
# generator = torch.load("./save/G.pth")

# discriminator.apply(weights_init_normal)
discriminator_g1.apply(weights_init_normal)
discriminator_g2.apply(weights_init_normal)


# 学习率全部缩十倍(最好的学习率是0.0001)
opt_lr = 0.0001
opt_b1 = 0.5
opt_b2 = 0.999
# Optimizers

optimizer_U = torch.optim.Adam(itertools.chain(generator.parameters(), Unet.parameters()), lr=opt_lr, betas=(opt_b1, opt_b2))
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt_lr, betas=(opt_b1, opt_b2))
optimizer_D_G1 = torch.optim.Adam(discriminator_g1.parameters(), lr=0.00001, betas=(opt_b1, opt_b2))
optimizer_D_G2 = torch.optim.Adam(discriminator_g2.parameters(), lr=0.00001, betas=(opt_b1, opt_b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.00001, betas=(opt_b1, opt_b2))

print("***************************开始训练***************************")
# train(Unet, generator, discriminator,discriminator_g, train_data, test_data, 5000, optimizer_U, optimizer_G, optimizer_D, optimizer_D_G, criterion_dice, criterion_pixelwise, criterion_GAN)
train(Unet, generator, discriminator,discriminator_g1,discriminator_g2, train_data, test_data, 5000, optimizer_U, optimizer_G, optimizer_D, optimizer_D_G1,optimizer_D_G2, criterion_dice, criterion_pixelwise, criterion_GAN)
# train(Unet, generator, discriminator_g, train_data, test_data, 5000, optimizer_U, optimizer_G, optimizer_D_G, criterion_dice, criterion_pixelwise, criterion_GAN)