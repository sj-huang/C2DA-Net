from torch.utils.data import Dataset
import pickle
import os
from medpy.io import load, save
import numpy as np
import cv2
import torch
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
        self.image_data = [] # length is the number of cases, and each case have a list, that have a sequence data
        self.label_data = []
        labels = 'label'

        label_path = os.path.join(path, labels)
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
                # if label[i].sum()!=0:
                self.image_data.append(image[i, :, :])
                self.label_data.append(label[i, :, :])

            print("new_img.shape, and new_label.shape:", image.shape, label.shape,tem)
    def __getitem__(self, idx):  # 根据 idx 取出其中一个name
        img = self.image_data[idx % len(self.image_data)]
        label = self.label_data[idx % len(self.label_data)]
        if self.transform is not None:
            img, label = self.transform(img, label)
        return img, label

    def __len__(self):  # 总数据的多少
        return len(self.label_data)


def collate_fn(batch):
    img_sequence_list=np.zeros((len(batch),1,128,192))
    label_sequence_list=np.zeros((len(batch),1,128,192))
    for i in range(len(batch)):
        img_sequence, label_sequence = batch[i][0], batch[i][1]
        Max = img_sequence.max() + 0.0000001
        Min = img_sequence.min()
        img_sequence = (img_sequence) / (Max - Min)  # 标准化，这个技巧之后会讲到
        img_sequence[img_sequence<0] = 0
        img_sequence = img_sequence[np.newaxis, :, :]
        label_sequence = label_sequence[np.newaxis, :, :]
        img_sequence_list[i]=img_sequence
        label_sequence_list[i]=label_sequence
    img_sequence = torch.Tensor(img_sequence_list).cuda()
    label_sequence = torch.Tensor(label_sequence_list).cuda()
    return img_sequence, label_sequence

from PIL import Image
# image, label, mask
def data_tf(img, label):
    degree = np.random.randint(0, 360, dtype='int')
    img_tem = Image.fromarray(img)
    img = img_tem.rotate(degree, center=(64, 96))
    label_tem = Image.fromarray(label)
    label = label_tem.rotate(degree, center=(64, 96))
    img = np.array(img)
    label = np.array(label)
    return img, label

def read_test(path):
    test_dataset = custom_dataset(path)
    test_data = DataLoader(test_dataset, 155, shuffle=True, collate_fn=collate_fn)
    return test_data

# test_data = read_test('../data/val')