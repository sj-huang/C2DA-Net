import sys
import os

sys.path.append('../../Domain')
import numpy as np
from datetime import datetime
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib

matplotlib.use('agg')
cuda = True if torch.cuda.is_available() else False
BCE = torch.nn.BCELoss(reduction='mean')
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


class dice_loss(nn.Module):
    def forward(self, uout, label):
        # def forward(self, uout, uout_1, label, label_1):
        """soft dice loss"""
        eps = 1e-7
        iflat = uout.view(-1)
        tflat = label.view(-1)
        intersection = (iflat * tflat).sum()
        dice_0 = 1 - 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)
        return dice_0


class dice_batch_loss(nn.Module):
    def forward(self, uout, label):
        # def get_acc(uout, uout_1, label, label_1):
        """soft dice score"""
        eps = 1e-7
        iflat_0, iflat_1, iflat_2, iflat_3, iflat_4, iflat_5, iflat_6 = uout[:, 0, :, :], uout[:, 1, :, :], uout[:, 2,
                                                                                                            :, :], uout[
                                                                                                                   :, 3,
                                                                                                                   :,
                                                                                                                   :], uout[
                                                                                                                       :,
                                                                                                                       4,
                                                                                                                       :,
                                                                                                                       :], uout[
                                                                                                                           :,
                                                                                                                           5,
                                                                                                                           :,
                                                                                                                           :], uout[
                                                                                                                               :,
                                                                                                                               6,
                                                                                                                               :,
                                                                                                                               :]
        tflat_0, tflat_1, tflat_2, tflat_3, tflat_4, tflat_5, tflat_6 = label[:, 0, :, :], label[:, 1, :, :], label[:,
                                                                                                              2, :,
                                                                                                              :], label[
                                                                                                                  :, 3,
                                                                                                                  :,
                                                                                                                  :], label[
                                                                                                                      :,
                                                                                                                      4,
                                                                                                                      :,
                                                                                                                      :], label[
                                                                                                                          :,
                                                                                                                          5,
                                                                                                                          :,
                                                                                                                          :], label[
                                                                                                                              :,
                                                                                                                              6,
                                                                                                                              :,
                                                                                                                              :]
        iflat_0 = iflat_0.contiguous().view(-1)
        tflat_0 = tflat_0.contiguous().view(-1)
        intersection = (iflat_0 * tflat_0).sum()
        dice_0 = 2. * intersection / ((iflat_0 ** 2).sum() + (tflat_0 ** 2).sum() + eps)

        iflat_1 = iflat_1.contiguous().view(-1)
        tflat_1 = tflat_1.contiguous().view(-1)
        intersection = (iflat_1 * tflat_1).sum()
        dice_1 = 2. * intersection / ((iflat_1 ** 2).sum() + (tflat_1 ** 2).sum() + eps)

        iflat_2 = iflat_2.contiguous().view(-1)
        tflat_2 = tflat_2.contiguous().view(-1)
        intersection = (iflat_2 * tflat_2).sum()
        dice_2 = 2. * intersection / ((iflat_2 ** 2).sum() + (tflat_2 ** 2).sum() + eps)

        # iflat_3 = (iflat_3.contiguous().view(-1) > 0.5).float()
        iflat_3 = iflat_3.contiguous().view(-1)
        tflat_3 = tflat_3.contiguous().view(-1)
        intersection = (iflat_3 * tflat_3).sum()

        dice_3 = 2. * intersection / ((iflat_3 ** 2).sum() + (tflat_3 ** 2).sum() + eps)

        # iflat_4 = (iflat_4.contiguous().view(-1) > 0.5).float()
        iflat_4 = iflat_4.contiguous().view(-1)
        tflat_4 = tflat_4.contiguous().view(-1)
        intersection = (iflat_4 * tflat_4).sum()
        dice_4 = 2. * intersection / ((iflat_4 ** 2).sum() + (tflat_4 ** 2).sum() + eps)

        # iflat_5 = (iflat_5.contiguous().view(-1) > 0.5).float()
        iflat_5 = iflat_5.contiguous().view(-1)
        tflat_5 = tflat_5.contiguous().view(-1)
        intersection = (iflat_5 * tflat_5).sum()
        dice_5 = 2. * intersection / ((iflat_5 ** 2).sum() + (tflat_5 ** 2).sum() + eps)

        # iflat_6 = (iflat_6.contiguous().view(-1) > 0.5).float()
        iflat_6 = iflat_6.contiguous().view(-1)
        tflat_6 = tflat_6.contiguous().view(-1)
        intersection = (iflat_6 * tflat_6).sum()
        dice_6 = 2. * intersection / ((iflat_6 ** 2).sum() + (tflat_6 ** 2).sum() + eps)

        dice_loss = 1 - (dice_0 + dice_1 + dice_2 + dice_3 + dice_4 + dice_5 + dice_6) / 7

        return dice_loss


def get_acc(uout, label):
    eps = 1e-7
    dice_0 = 0
    for i in range(len(uout)):
        iflat = (uout[i].view(-1) > 0.5).float()
        tflat = label[i].view(-1)
        intersection = (iflat * tflat).sum()
        dice = 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)
        dice_0 = dice_0 + dice
    dice_0 = dice_0 / (len(uout) + 1)
    return dice_0


def get_batch_acc(uout, label):
    # def get_acc(uout, uout_1, label, label_1):
    """soft dice score"""
    eps = 1e-7
    iflat_0, iflat_1, iflat_2, iflat_3, iflat_4, iflat_5, iflat_6 = uout[:, 0, :, :], uout[:, 1, :, :], uout[:, 2, :,
                                                                                                        :], uout[:, 3,
                                                                                                            :, :], uout[
                                                                                                                   :, 4,
                                                                                                                   :,
                                                                                                                   :], uout[
                                                                                                                       :,
                                                                                                                       5,
                                                                                                                       :,
                                                                                                                       :], uout[
                                                                                                                           :,
                                                                                                                           6,
                                                                                                                           :,
                                                                                                                           :]
    tflat_0, tflat_1, tflat_2, tflat_3, tflat_4, tflat_5, tflat_6 = label[:, 0, :, :], label[:, 1, :, :], label[:, 2, :,
                                                                                                          :], label[:,
                                                                                                              3, :,
                                                                                                              :], label[
                                                                                                                  :, 4,
                                                                                                                  :,
                                                                                                                  :], label[
                                                                                                                      :,
                                                                                                                      5,
                                                                                                                      :,
                                                                                                                      :], label[
                                                                                                                          :,
                                                                                                                          6,
                                                                                                                          :,
                                                                                                                          :]
    iflat_0 = (iflat_0.contiguous().view(-1) > 0.5).float()
    tflat_0 = tflat_0.contiguous().view(-1)
    intersection = (iflat_0 * tflat_0).sum()
    dice_0 = 2. * intersection / ((iflat_0 ** 2).sum() + (tflat_0 ** 2).sum() + eps)

    iflat_1 = (iflat_1.contiguous().view(-1) > 0.5).float()
    tflat_1 = tflat_1.contiguous().view(-1)
    intersection = (iflat_1 * tflat_1).sum()
    dice_1 = 2. * intersection / ((iflat_1 ** 2).sum() + (tflat_1 ** 2).sum() + eps)

    iflat_2 = (iflat_2.contiguous().view(-1) > 0.5).float()
    tflat_2 = tflat_2.contiguous().view(-1)
    intersection = (iflat_2 * tflat_2).sum()
    dice_2 = 2. * intersection / ((iflat_2 ** 2).sum() + (tflat_2 ** 2).sum() + eps)

    # iflat_3 = (iflat_3.contiguous().view(-1) > 0.5).float()
    iflat_3 = (iflat_3.contiguous().view(-1)).float()
    tflat_3 = tflat_3.contiguous().view(-1)
    intersection = (iflat_3 * tflat_3).sum()
    dice_3 = 2. * intersection / ((iflat_3 ** 2).sum() + (tflat_3 ** 2).sum() + eps)

    # iflat_4 = (iflat_4.contiguous().view(-1) > 0.5).float()
    iflat_4 = (iflat_4.contiguous().view(-1)).float()
    tflat_4 = tflat_4.contiguous().view(-1)
    intersection = (iflat_4 * tflat_4).sum()
    dice_4 = 2. * intersection / ((iflat_4 ** 2).sum() + (tflat_4 ** 2).sum() + eps)

    # iflat_5 = (iflat_5.contiguous().view(-1) > 0.5).float()
    iflat_5 = (iflat_5.contiguous().view(-1)).float()
    tflat_5 = tflat_5.contiguous().view(-1)
    intersection = (iflat_5 * tflat_5).sum()
    dice_5 = 2. * intersection / ((iflat_5 ** 2).sum() + (tflat_5 ** 2).sum() + eps)

    # iflat_6 = (iflat_6.contiguous().view(-1) > 0.5).float()
    iflat_6 = (iflat_6.contiguous().view(-1)).float()
    tflat_6 = tflat_6.contiguous().view(-1)
    intersection = (iflat_6 * tflat_6).sum()
    dice_6 = 2. * intersection / ((iflat_6 ** 2).sum() + (tflat_6 ** 2).sum() + eps)

    return dice_0, dice_1, dice_2, dice_3, dice_4, dice_5, dice_6


def get_good_acc(uout, label):
    # def get_acc(uout, uout_1, label, label_1):
    """soft dice score"""
    eps = 1e-7
    iflat = (uout.view(-1) > 0.5).float()
    tflat = label.view(-1)
    intersection = (iflat * tflat).sum()
    dice_0 = 2. * intersection / ((iflat ** 2).sum() + (tflat ** 2).sum() + eps)

    return dice_0


def get_label_2(label):
    label[label == 7] = 1
    label[label == 8] = 2
    label[label == 9] = 3
    label[label == 10] = 4
    label[label == 11] = 5
    label[label == 12] = 6
    return label


def get_label(label):
    realB2 = label.cpu().numpy()
    # realB2 = get_label_2(label)
    real = np.zeros((realB2.shape[0], 7, 128, 192))
    for i in range(7):
        if i == 6:
            re = np.expand_dims(real[:, i, :, :], 1)
            re[realB2 == 0] = 1
        else:
            re = np.expand_dims(real[:, i, :, :], 1)
            re[realB2 == i + 1] = 1
    real = torch.Tensor(real.astype("float32")).cuda()
    return real


from skimage.measure import label as la


def crop_func(np_image):
    loc_img, num = la(np_image, background=0, return_num=True, connectivity=2)
    loc_img[loc_img != 0] = 1
    max_label = 0
    max_num = 0
    for i in range(1, num + 1):
        if np.sum(loc_img == i) > max_num:
            max_num = np.sum(loc_img == i)
            max_label = i
    mcr = (loc_img == max_label)
    mcr = mcr + 0
    y_true, x_true, z_true = np.where(mcr)
    box = np.array([[np.min(y_true), np.max(y_true)],
                    [np.min(x_true), np.max(x_true)],
                    [np.min(z_true), np.max(z_true)]])
    y_min, y_max = box[0]
    x_min, x_max = box[1]
    z_min, z_max = box[2]
    return y_min, y_max, x_min, x_max, z_min, z_max


def get_patch(uout, out_feature, im, i):
    uout[uout != i + 1] = 0
    y_min, y_max, x_min, x_max, z_min, z_max = crop_func(uout)
    feature_out = out_feature[y_min:y_max, i, x_min:x_max, z_min:z_max]
    im_out = im[y_min:y_max, 0, x_min:x_max, z_min:z_max]
    return feature_out.unsqueeze(1), im_out.unsqueeze(1)


def get_low(im_tensor):
    fft_src_np = torch.fft.fftn(im_tensor, dim=(-4, -3, -2, -1))
    fshift = torch.fft.fftshift(fft_src_np).cuda()
    np_zero = torch.zeros_like(fshift).cuda()
    b=im_tensor.shape[2]//2
    c=im_tensor.shape[3]//2
    s_b = b//20
    s_c = c//20
    np_zero[:, :,b - s_b:b + s_b, c - s_c:c + s_c] = 1
    fshift = fshift * np_zero
    ishift = torch.fft.ifftshift(fshift).cuda()
    iimg = abs(torch.fft.ifftn(ishift, dim=(-4,-3, -2, -1)).cuda())
    return iimg
def get_high(im_tensor):
    fft_src_np = torch.fft.fftn(im_tensor, dim=(-4, -3, -2, -1))
    fshift = torch.fft.fftshift(fft_src_np).cuda()
    np_zero = torch.ones_like(fshift).cuda()
    b=im_tensor.shape[2]//2
    c=im_tensor.shape[3]//2
    s_b = b //20
    s_c = c //20
    np_zero[:, :, b - s_b:b + s_b, c - s_c:c + s_c] = 0
    fshift = fshift * np_zero
    ishift = torch.fft.ifftshift(fshift).cuda()
    iimg = abs(torch.fft.ifftn(ishift, dim=(-4,-3, -2, -1)).cuda())
    # iimg=torch.cat((im_tensor,iimg),1)
    return iimg


def G_opt_step(SS_1,SS_img,optimizer_G,discriminator_g,criterion_pixelwise,criterion_GAN):
    optimizer_G.zero_grad()
    SS_1_D=discriminator_g(SS_1)
    SS_1_loss = criterion_pixelwise(SS_1, SS_img) + \
                criterion_GAN(SS_1_D,Variable(torch.FloatTensor(SS_1_D.data.size()).fill_(1)).cuda())

    SS_1_loss.backward()
    optimizer_G.step()
    return SS_1_loss
def D_opt_step(SS_1,SS_img,optimizer_D_G,discriminator_g,criterion_GAN):
    optimizer_D_G.zero_grad()
    SS_1_F = discriminator_g(SS_1)
    SS_1_T = discriminator_g(SS_img)

    SS_1_D_loss = criterion_GAN(SS_1_F, Variable(torch.FloatTensor(SS_1_F.data.size()).fill_(0)).cuda())\
                  +criterion_GAN(SS_1_T, Variable(torch.FloatTensor(SS_1_T.data.size()).fill_(1)).cuda())
    # print("D_loss: ", SS_1_D_loss.item())
    SS_1_D_loss.backward()
    optimizer_D_G.step()
    return SS_1_D_loss

def train(Unet, generator, discriminator,discriminator_g1,discriminator_g2, train_data, test_data, epoch_num, optimizer_U, optimizer_G, optimizer_D,optimizer_D_G1,optimizer_D_G2, criterion_dice, criterion_pixelwise, criterion_GAN):
    torch.autograd.set_detect_anomaly(True)
    if torch.cuda.is_available():
        Unet = Unet.cuda()
        generator = generator.cuda()
        discriminator_g1 = discriminator_g1.cuda()
        discriminator_g2 = discriminator_g2.cuda()
        discriminator = discriminator.cuda()
    prev_time = datetime.now()
    best_acc_t = 0
    for epoch in range(epoch_num):
        if epoch > 0 and epoch <= 10:
            if epoch % 1 == 0:
                for p in optimizer_U.param_groups:
                    p['lr'] = p['lr'] - p['lr'] * 0.05
        elif epoch >= 11 and epoch <= 21:
            if epoch % 2 == 0:
                for p in optimizer_U.param_groups:
                    p['lr'] = p['lr'] - p['lr'] * 0.05
        elif epoch >= 25 and epoch <= 45:
            if epoch % 3 == 0:
                for p in optimizer_U.param_groups:
                    p['lr'] = p['lr'] - p['lr'] * 0.05
        elif epoch >= 50 and epoch <= 100:
            if epoch % 11 == 0:
                for p in optimizer_U.param_groups:
                    p['lr'] = p['lr'] - p['lr'] * 0.05
        elif epoch > 150 and epoch <= 250:
            if epoch % 11 == 0:
                for p in optimizer_U.param_groups:
                    p['lr'] = p['lr'] - p['lr'] * 0.02
        for p in optimizer_U.param_groups:
            print("####################Epoch, LR: {},{}#####################".format(str(epoch),p['lr']))
        Unet = Unet.train()
        generator = generator.train()
        discriminator_g1 = discriminator_g1.train()
        discriminator_g2 = discriminator_g2.train()
        discriminator = discriminator.train()
        D_loss_avg=0
        G_loss_avg=0
        U_loss_avg=0
        G_num=0
        D_num=0
        U_num=0
        for SS_img, SS_label, ST_img, ST_label, TT_img, TS_img in train_data:
            G_num+=4
            U_num+=2
            D_num+=4

            SS_label=get_label(SS_label)
            ST_label=get_label(ST_label)

            SS_style = get_low(SS_img)
            ST_style = get_low(ST_img)
            TT_style = get_low(TT_img)

            SS_content = get_high(SS_img)
            ST_content = get_high(ST_img)
            TT_content = get_high(TT_img)

            # optimizer_U.zero_grad()
            # S_mask = Unet(SS_img)  # # [(bs, 1, h, w),(bs, 1, h, w),(bs, 1, h, w),(bs, 1, h, w)]
            # loss_seg_S = criterion_dice(S_mask, SS_label)  # dice loss
            # loss_bce_S = BCE(S_mask, SS_label)
            # SS_1 = generator(SS_style, S_mask)
            # SS_1_D = discriminator_g(SS_1)
            # SS_pixel_loss = 3 * criterion_pixelwise(SS_1, SS_img)
            # SS_D_loss = 0.1 * criterion_GAN(SS_1_D, Variable(torch.FloatTensor(SS_1_D.data.size()).fill_(1)).cuda())
            # S_mask_fake = Unet(SS_1)
            # loss_seg_S_fake = criterion_dice(S_mask_fake, S_mask)
            # loss_U = loss_seg_S + loss_bce_S + loss_seg_S_fake + SS_pixel_loss + SS_D_loss
            # print("loss_U: ",loss_U.item(),loss_seg_S.item(),loss_bce_S.item(),loss_seg_S_fake.item(),SS_pixel_loss.item(),SS_D_loss.item())
            # U_loss_avg += loss_U
            # loss_U.backward()
            # optimizer_U.step()

            optimizer_U.zero_grad()
            T_mask,feature = Unet(ST_content)
            loss_seg_T = criterion_dice(T_mask, ST_label)  # dice loss
            loss_bce_T = BCE(T_mask, ST_label)
            ST_1 = generator(ST_style, T_mask)
            ST_1_content=get_high(ST_1)
            ST_1_D = discriminator_g1(ST_1)
            ST_2_D = discriminator(feature)
            ST_pixel_loss = 10*criterion_pixelwise(ST_1, ST_img)
            ST_D_loss = 0.1*(criterion_GAN(ST_1_D, Variable(torch.FloatTensor(ST_1_D.data.size()).fill_(1)).cuda())+\
                        criterion_GAN(ST_2_D,Variable(torch.FloatTensor(ST_2_D.data.size()).fill_(0)).cuda()))
            T_mask_fake,_ = Unet(ST_1_content)
            loss_seg_T_fake=criterion_dice(T_mask_fake,T_mask)
            loss_U = loss_seg_T+loss_bce_T+loss_seg_T_fake+ST_pixel_loss+ST_D_loss
            # print("loss_U: ",loss_U.item(),loss_seg_T.item(),loss_bce_T.item(),loss_seg_T_fake.item(),ST_pixel_loss.item(),ST_D_loss.item())
            U_loss_avg += loss_U
            loss_U.backward()
            optimizer_U.step()

            optimizer_U.zero_grad()
            T_mask,T_feature = Unet(TT_content)
            TT_1 = generator(TT_style, T_mask)
            TT_1_content = get_high(TT_1)
            TT_1_D = discriminator_g2(TT_1)
            TT_2_D = discriminator(T_feature)
            TT_pixel_loss = 10*criterion_pixelwise(TT_1, TT_img)
            TT_D_loss = 0.1*(criterion_GAN(TT_1_D, Variable(torch.FloatTensor(TT_1_D.data.size()).fill_(1)).cuda())+\
                        criterion_GAN(TT_2_D, Variable(torch.FloatTensor(TT_2_D.data.size()).fill_(1)).cuda()))
            T_mask_fake,_ = Unet(TT_1_content)
            loss_seg_T_fake = criterion_dice(T_mask_fake, T_mask)
            loss_U = loss_seg_T_fake+TT_pixel_loss + TT_D_loss
            # print("loss_U,loss_seg_T_fake,TT_pixel_loss,TT_D_loss: ", loss_U.item(), loss_seg_T_fake.item(),TT_pixel_loss.item(), TT_D_loss.item())
            U_loss_avg += loss_U
            loss_U.backward()
            optimizer_U.step()

            if epoch % 3 == 0:
                # SS_mask=Unet(SS_img)
                # SS_1 = generator(SS_style, SS_mask)
                # D_loss_avg += D_opt_step(SS_1, SS_img, optimizer_D_G,discriminator_g,criterion_GAN)
                ST_mask,_ = Unet(ST_content)
                ST_2 = generator(ST_style, ST_mask)
                D_loss_avg += D_opt_step(ST_2, ST_img, optimizer_D_G1, discriminator_g1, criterion_GAN)

                TT_mask,_ = Unet(TT_content)
                TT_2 = generator(TT_style, TT_mask)
                D_loss_avg += D_opt_step(TT_2, TT_img, optimizer_D_G2, discriminator_g2, criterion_GAN)

                _, ST_f = Unet(ST_content)
                _, TT_f = Unet(TT_content)
                D_loss_avg += D_opt_step(TT_f, ST_f, optimizer_D, discriminator, criterion_GAN)


        print("G_loss: ", (G_loss_avg/G_num))
        print("U_loss: ", (U_loss_avg/U_num))
        print("D_loss: ", (D_loss_avg/U_num))


        print("\n+++++++++++++++++++++++++++++  Start Testing  +++++++++++++++++++++++++++++")
        cur_time = datetime.now()
        h, remainder = divmod((cur_time - prev_time).seconds, 3600)
        m, s = divmod(remainder, 60)
        time_str = "Tnew_ime %02d:%02d:%02d" % (h, m, s)
        index = 0
        if test_data is not None:
            val_acc_0 = 0
            val_sum_batch_0 = 0
            val_sum_batch_1 = 0
            val_sum_batch_2 = 0
            val_sum_batch_3 = 0
            val_sum_batch_4 = 0
            val_sum_batch_5 = 0
            val_sum_batch_6 = 0
            with torch.no_grad():
                Unet = Unet.eval()
                generator = generator.eval()
                for img, label in test_data:
                    index = index + 1
                    label = get_label(label)
                    img_content=get_high(img)
                    mask,_ = Unet(img_content)

                    dice_0 = get_acc(mask, label)
                    val_acc_0 = val_acc_0 + dice_0
                    dice_batch_0, dice_batch_1, dice_batch_2, dice_batch_3, dice_batch_4, dice_batch_5, dice_batch_6 = get_batch_acc(
                        mask, label)
                    dice_batch_a = (dice_batch_0 + dice_batch_1 + dice_batch_2 + dice_batch_3 + dice_batch_4 + dice_batch_5) / 6
                    print("Teacher net: name, train_batch_a, dice_0 ", dice_batch_a.item(), dice_0.item())
                    print("(Teacher net)train dice are: %f, %f, %f, %f, %f, %f, %f,"
                          % (dice_batch_0.item(), dice_batch_1.item(), dice_batch_2.item(),
                             dice_batch_3.item(), dice_batch_4.item(), dice_batch_5.item(),
                             dice_batch_6.item()))
                    # if index==5:
                        # res = mask.detach().cpu().numpy()
                        # mask = np.argmax(res, axis=1)
                        # mask = np.squeeze(mask)
                        # mask = mask.transpose((0, 2, 1))
                        # mask = mask + 1
                        # mask[mask == 7] = 0
                        # mask = sitk.GetImageFromArray(mask)
                        # sitk.WriteImage(mask, "mask.nii.gz")
                        #
                        # res = label.detach().cpu().numpy()
                        # mask = np.argmax(res, axis=1)
                        # mask = np.squeeze(mask)
                        # mask = mask.transpose((0, 2, 1))
                        # mask = mask + 1
                        # mask[mask == 7] = 0
                        # mask = sitk.GetImageFromArray(mask.astype("int16"))
                        # sitk.WriteImage(mask, "label.nii.gz")
                        #
                        # res = img.detach().cpu().numpy()
                        # mask = np.squeeze(res)
                        # mask = mask.transpose((0, 2, 1))
                        # mask = mask + 1
                        # mask[mask == 7] = 0
                        # mask = sitk.GetImageFromArray(mask)
                        # sitk.WriteImage(mask, "img.nii.gz")

                    val_sum_batch_0 = val_sum_batch_0 + dice_batch_0
                    val_sum_batch_1 = val_sum_batch_1 + dice_batch_1
                    val_sum_batch_2 = val_sum_batch_2 + dice_batch_2
                    val_sum_batch_3 = val_sum_batch_3 + dice_batch_3
                    val_sum_batch_4 = val_sum_batch_4 + dice_batch_4
                    val_sum_batch_5 = val_sum_batch_5 + dice_batch_5
                    val_sum_batch_6 = val_sum_batch_6 + dice_batch_6

                val_batch_0 = val_sum_batch_0 / index
                val_batch_1 = val_sum_batch_1 / index
                val_batch_2 = val_sum_batch_2 / index
                val_batch_3 = val_sum_batch_3 / index
                val_batch_4 = val_sum_batch_4 / index
                val_batch_5 = val_sum_batch_5 / index
                val_batch_a = (val_batch_0 + val_batch_1 + val_batch_2 + val_batch_3 + val_batch_4 + val_batch_5) / 6
                sys.stdout.flush()
        else:
            print("===========================ERROR===========================")
        prev_time = cur_time
        print(time_str)
        if test_data:
            print("(Teacher net) val_batch_a:", val_batch_a)
            if val_batch_a > best_acc_t:
                best_acc_t = val_batch_a
                # 打印现在最好的准确率
                print('(Teacher net) New best acc: %.4f' % best_acc_t)
                torch.save(Unet, "./save_0.2/U.pth")
                torch.save(generator, "./save_0.2/G.pth")
                torch.save(discriminator_g1, "./save_0.2/D_G1.pth")
                torch.save(discriminator_g2, "./save_0.2/D_G2.pth")

        else:
            print("===========================ERROR===========================")

        print("(Teacher net)现在最好的准确率为：%.4f" % (best_acc_t))
import surface_distance as surfdist
import SimpleITK as sitk
def test(valid_data,Unet):
    index=0
    if valid_data is not None:
        dice=[]
        dice_1=[]
        dice_2=[]
        dice_3=[]
        dice_4=[]
        dice_5=[]
        dice_6=[]

        assd = []
        assd_1 = []
        assd_2 = []
        assd_3 = []
        assd_4 = []
        assd_5 = []
        assd_6 = []

        with torch.no_grad():
            Unet = Unet.eval()
            for img, label,name in valid_data:
                index = index + 1
                # img=img.transpose(1,2).squeeze(0)
                # label=label.transpose(1,2).squeeze(0)
                label = get_label(label)
                img_content = get_high(img)
                mask,_ = Unet(img_content)

                # for i in range(6):
                #     mask_ = np.array(mask[:, i].cpu())
                #     mask_[mask_ > 0.5] = 1
                #     mask_[mask_ <= 0.5] = 0
                #     surface_distance = surfdist.compute_surface_distances(np.array(label[:, i].cpu()).astype("bool"),
                #                                                           mask_.astype("bool"),
                #                                                           spacing_mm=(0.8, 0.8, 0.8))
                #     # avg_surf_dist=surfdist.compute_robust_hausdorff(surface_distance,95)
                #     avg_surf_dist = surfdist.compute_average_surface_distance(surface_distance)
                #     if i == 0:
                #         avg_surf_dist_1 = (avg_surf_dist[0] + avg_surf_dist[1]) / 2
                #     elif i == 1:
                #         avg_surf_dist_2 = (avg_surf_dist[0] + avg_surf_dist[1]) / 2
                #     elif i == 2:
                #         avg_surf_dist_3 = (avg_surf_dist[0] + avg_surf_dist[1]) / 2
                #     elif i == 3:
                #         avg_surf_dist_4 = (avg_surf_dist[0] + avg_surf_dist[1]) / 2
                #     elif i == 4:
                #         avg_surf_dist_5 = (avg_surf_dist[0] + avg_surf_dist[1]) / 2
                #     elif i == 5:
                #         avg_surf_dist_6 = (avg_surf_dist[0] + avg_surf_dist[1]) / 2
                # avg_surf_dist_a = (avg_surf_dist_1 + avg_surf_dist_2 + avg_surf_dist_3 + avg_surf_dist_4 + avg_surf_dist_5 + avg_surf_dist_6) / 6
                # assd.append(avg_surf_dist_a)
                # assd_1.append(avg_surf_dist_1)
                # assd_2.append(avg_surf_dist_2)
                # assd_3.append(avg_surf_dist_3)
                # assd_4.append(avg_surf_dist_4)
                # assd_5.append(avg_surf_dist_5)
                # assd_6.append(avg_surf_dist_6)
                #
                # dice_batch_0, dice_batch_1, dice_batch_2, dice_batch_3, dice_batch_4, dice_batch_5, dice_batch_6 = get_batch_acc(
                #     mask, label)
                # dice_batch_a = (dice_batch_0 + dice_batch_1 + dice_batch_2 + dice_batch_3 + dice_batch_4 + dice_batch_5) / 6
                # dice.append(dice_batch_a.item())
                # dice_1.append(dice_batch_0.item())
                # dice_2.append(dice_batch_1.item())
                # dice_3.append(dice_batch_2.item())
                # dice_4.append(dice_batch_3.item())
                # dice_5.append(dice_batch_4.item())
                # dice_6.append(dice_batch_5.item())

                res = mask.detach().cpu().numpy()
                mask = np.argmax(res, axis=1)
                mask = np.squeeze(mask)
                # mask = mask.transpose((0, 2, 1))
                mask = mask + 1
                mask[mask == 7] = 0
                mask = sitk.GetImageFromArray(mask.astype("int16"))
                # sitk.WriteImage(mask, name.replace(".nii", "_mask.nii"))
                sitk.WriteImage(mask, "1.nii.gz")
                #
                # res = img.detach().cpu().numpy()
                # mask = np.squeeze(res)
                # # mask = mask.transpose((0, 2, 1))
                # mask = mask + 1
                # mask[mask == 7] = 0
                # mask = sitk.GetImageFromArray(mask)
                # sitk.WriteImage(mask, name)
                #
                # res = label.detach().cpu().numpy()
                # mask = np.argmax(res, axis=1)
                # mask = np.squeeze(mask)
                # # mask = mask.transpose((0, 2, 1))
                # mask = mask + 1
                # mask[mask == 7] = 0
                # mask = sitk.GetImageFromArray(mask.astype("int16"))
                # sitk.WriteImage(mask, name.replace(".nii", "_label.nii"))
            # print(round(100 * np.mean(dice_1), 2), round(100 * np.std(dice_1, ddof=1), 2), round(np.mean(assd_1), 2),
            #       round(np.std(assd_1, ddof=1), 2))
            # print(round(100 * np.mean(dice_2), 2), round(100 * np.std(dice_2, ddof=1), 2), round(np.mean(assd_2), 2),
            #       round(np.std(assd_2, ddof=1), 2))
            # print(round(100 * np.mean(dice_3), 2), round(100 * np.std(dice_3, ddof=1), 2), round(np.mean(assd_3), 2),
            #       round(np.std(assd_3, ddof=1), 2))
            # print(round(100 * np.mean(dice_4), 2), round(100 * np.std(dice_4, ddof=1), 2), round(np.mean(assd_4), 2),
            #       round(np.std(assd_4, ddof=1), 2))
            # print(round(100 * np.mean(dice_5), 2), round(100 * np.std(dice_5, ddof=1), 2), round(np.mean(assd_5), 2),
            #       round(np.std(assd_5, ddof=1), 2))
            # print(round(100 * np.mean(dice_6), 2), round(100 * np.std(dice_6, ddof=1), 2), round(np.mean(assd_6), 2),
            #       round(np.std(assd_6, ddof=1), 2))
            # print(round(100 * np.mean(dice), 2), round(100 * np.std(dice, ddof=1), 2), round(np.mean(assd), 2),
            #       round(np.std(assd, ddof=1), 2))