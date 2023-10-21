from torch.utils.data import Dataset
import pickle
import os
from medpy.io import load, save
import numpy as np
import torch
import cv2
from skimage.measure import label as la
from torch.utils.data import DataLoader
import SimpleITK as sitk
from scipy import ndimage

"""
    定义一个 读取数据集.pkl文件的类：
"""


# 定义一个子类叫 custom_dataset，继承与 Dataset
class custom_dataset(Dataset):
    def __init__(self, path, transform=None):
        self.transform = transform  # 传入数据预处理
        self.SS_image_data = []
        self.ST_image_data = []
        self.TT_image_data = []
        self.TS_image_data = []

        self.SS_label_data = []
        self.ST_label_data = []
        # self.TT_label_data = []
        # self.TS_label_data = []
        self.name = []

        path_name=["SS","ST","TT","TS"]
        for n in range(4):
            labels = 'label'
            label_path = os.path.join(path,path_name[n], labels)
            label_path_list = os.listdir(label_path)

            for label_name in label_path_list:
                label, _ = load(label_path + '/' + label_name)
                label = label.transpose(2, 0, 1)
                tem = label_name

                image, _ = load(label_path.replace("label","image") + '/' + label_name)  # Load data 加载当前读取的这个nii数据
                image = image.astype("float32")
                image = image.transpose(2, 0, 1)
                TA = (1, 128 / label.shape[1], 192 / label.shape[2])
                image = ndimage.zoom(image, TA, order=0)
                label = ndimage.zoom(label, TA, order=0)

                for i in range(image.shape[0]):
                    if label[i].sum()!=0:
                        if n==0:
                            self.SS_image_data.append(image[i, :, :])
                            self.SS_label_data.append(label[i, :, :])
                        elif n==1:
                            self.ST_image_data.append(image[i, :, :])
                            self.ST_label_data.append(label[i, :, :])
                        elif n==2:
                            self.TT_image_data.append(image[i, :, :])
                            # self.TT_label_data.append(label[i, :, :])
                        elif n==3:
                            self.TS_image_data.append(image[i, :, :])
                            # self.TS_label_data.append(label[i, :, :])
                        # self.name.append(tem)

                print("new_img.shape, and new_label.shape:", image.shape, label.shape,tem)
    def __getitem__(self, idx):  # 根据 idx 取出其中一个name
        SS_img = self.SS_image_data[idx % len(self.SS_image_data)]
        SS_label = self.SS_label_data[idx % len(self.SS_label_data)]
        ST_img = self.ST_image_data[idx % len(self.ST_image_data)]
        ST_label = self.ST_label_data[idx % len(self.ST_label_data)]
        TT_img = self.TT_image_data[idx % len(self.TT_image_data)]
        TS_img = self.TS_image_data[idx % len(self.TS_image_data)]
        if self.transform is not None:
            SS_img, SS_label, ST_img, ST_label, TT_img, TS_img, = \
                self.transform(SS_img, SS_label, ST_img, ST_label, TT_img, TS_img)
        return SS_img, SS_label, ST_img, ST_label, TT_img, TS_img
            # img, label,name = self.transform(img, label, name)
        # return img, label,name
    def __len__(self):  # 总数据的多少
        return len(self.SS_label_data)


# 重写collate_fn函数，其输入为一个batch的sample数据
def collate_fn(batch):
    SS_img_list=np.zeros((len(batch),1,128,192))
    SS_label_list=np.zeros((len(batch),1,128,192))
    ST_img_list=np.zeros((len(batch),1,128,192))
    ST_label_list=np.zeros((len(batch),1,128,192))
    TT_img_list=np.zeros((len(batch),1,128,192))
    TS_img_list=np.zeros((len(batch),1,128,192))
    for i in range(len(batch)):
        SS_img, SS_label, ST_img, ST_label, TT_img, TS_img, = \
            batch[i][0], batch[i][1], batch[i][2], batch[i][3],batch[i][4],batch[i][5],
        Max = SS_img.max() + 0.0000001
        Min = SS_img.min()
        SS_img = (SS_img) / (Max - Min)  # 标准化，这个技巧之后会讲到
        SS_img[SS_img<0] = 0
        SS_img = SS_img[np.newaxis, :, :]
        SS_label = SS_label[np.newaxis, :, :]
        SS_img_list[i]=SS_img
        SS_label_list[i]=SS_label

        Max = ST_img.max() + 0.0000001
        Min = ST_img.min()
        ST_img = (ST_img) / (Max - Min)  # 标准化，这个技巧之后会讲到
        ST_img[ST_img < 0] = 0
        ST_img = ST_img[np.newaxis, :, :]
        ST_label = ST_label[np.newaxis, :, :]
        ST_img_list[i] = ST_img
        ST_label_list[i] = ST_label

        Max = TT_img.max() + 0.0000001
        Min = TT_img.min()
        TT_img = (TT_img) / (Max - Min)  # 标准化，这个技巧之后会讲到
        TT_img[TT_img < 0] = 0
        TT_img = TT_img[np.newaxis, :, :]
        TT_img_list[i] = TT_img

        Max = TS_img.max() + 0.0000001
        Min = TS_img.min()
        TS_img = (TS_img) / (Max - Min)  # 标准化，这个技巧之后会讲到
        TS_img[TS_img < 0] = 0
        TS_img = TS_img[np.newaxis, :, :]
        TS_img_list[i] = TS_img

    return torch.Tensor(SS_img_list).cuda(), torch.Tensor(SS_label_list).cuda(), \
           torch.Tensor(ST_img_list).cuda(), torch.Tensor(ST_label_list).cuda(), \
           torch.Tensor(TT_img_list).cuda(), \
           torch.Tensor(TS_img_list).cuda()

from PIL import Image
# image, label, mask
def data_tf(SS_img, SS_label, ST_img, ST_label, TT_img, TS_img):
    degree = np.random.randint(0, 360, dtype='int')
    SS_img = Image.fromarray(SS_img)
    SS_img = SS_img.rotate(degree)
    SS_label = Image.fromarray(SS_label)
    SS_label = SS_label.rotate(degree)
    SS_img = np.array(SS_img)
    SS_label = np.array(SS_label)

    ST_img = Image.fromarray(ST_img)
    ST_img = ST_img.rotate(degree)
    ST_label = Image.fromarray(ST_label)
    ST_label = ST_label.rotate(degree)
    ST_img = np.array(ST_img)
    ST_label = np.array(ST_label)

    TT_img = Image.fromarray(TT_img)
    TT_img = TT_img.rotate(degree)
    TT_img = np.array(TT_img)

    TS_img = Image.fromarray(TS_img)
    TS_img = TS_img.rotate(degree)
    TS_img = np.array(TS_img)
    return SS_img, SS_label, ST_img, ST_label, TT_img, TS_img
def read_train(path):
    train_dataset = custom_dataset(path, transform=data_tf)
    train_data = DataLoader(train_dataset, 20, shuffle=True, collate_fn=collate_fn)
    return train_data