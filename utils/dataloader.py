
import numpy as np
import os
import cv2

import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader

from utils.event_utils import *
# from utilybu import *


class DL_NMNIST(Dataset):

    def __init__(self, HW=[34, 34],
                 set=np.zeros(0),
                 input_len=30,
                 set_size=1000,
                 train_percent=0.5,
                 device=torch.device("cuda:0")):
        '''

        dataloader Neuromorphic mnist
        '''
        self.h = HW[0]
        self.w = HW[1]
        self.set=set
        self.length = set_size

        self.trainSize=int(set_size*train_percent)
        self.testSize=self.length-self.trainSize

        self.input_len = input_len
        self.device = device

    def __getitem__(self, index):

        # (10, 1000, 30, 2, 34, 34)
        ii,target=divmod(index,10)

        xx=self.set[target,ii].astype(np.float32)

        xx = torch.from_numpy(xx).to(self.device)
        xx = xx[:,:,1:-1,1:-1] # 34 -> 32


        # pause()

        return xx, target

    def __len__(self):
        return self.trainSize


class DL_NMNIST_Test(Dataset):

    def __init__(self, HW=[34, 34],
                 set=np.zeros(0),
                 input_len=30,
                 set_size=1000,
                 train_percent=0.5,
                 device=torch.device("cuda:0"),
                 is_train=False):
        '''

        dataloader Neuromorphic mnist
        '''
        self.h = HW[0]
        self.w = HW[1]
        self.set = set
        self.length = set_size
        self.is_train=is_train

        self.trainSize = int(set_size * train_percent)
        self.testSize = self.length - self.trainSize

        self.input_len = input_len
        self.device = device

    def __getitem__(self, index):
        # (10, 1000, 30, 2, 34, 34)

        if self.is_train:
            ii, target = divmod(index, 10)
        else:
            ii, target = divmod(index+self.trainSize, 10)

        xx = self.set[target, ii].astype(np.float32)

        xx = torch.from_numpy(xx).to(self.device) # torch.Size([30, 2, 34, 34])
        xx = xx[:,:,1:-1,1:-1] # 34 -> 32
        # print(xx.shape)
        # xx=torch.zeros((30, 11, 34, 34)).to(self.device)
        # pause()

        return xx, target

    def __len__(self):
        return self.testSize





class DL_BouncingBall64(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, HW=[256, 256],
                 input_len=60,
                 set_size=50,
                 events_folder='',
                 device=torch.device("cuda:0")):
        '''

        这个loading 是 bouncingball 的function

        G:/Cynthia/Q/dataset/bouncingball/events

        '''
        self.h = HW[0]
        self.w = HW[1]
        self.length = set_size

        self.input_len = input_len
        self.events_folder = events_folder
        self.device = device

    def __getitem__(self, index):
        # sample_path = os.path.join(self.events_folder, index2name(index))
        sample_path = os.path.join(self.events_folder, str(index) + '.npy')

        if not os.path.exists(sample_path):
            print('Sample Not Exist,', sample_path)
        sample_gt = np.load(sample_path)[0:self.input_len + 2]
        # sample_gt = torch.from_numpy(sample_gt).to(self.device).to(torch.float32)
        '''
        sequence 第一帧是空的 所以从第二帧开始截
        gt 相比于输入延迟一帧   
        '''
        sample = sample_gt[1:self.input_len + 1]
        gt = sample_gt[2:self.input_len + 2]


        sample = torch.from_numpy(sample_gt[1:self.input_len + 1]).to(self.device).to(torch.float32)
        gt = torch.from_numpy(sample_gt[2:self.input_len + 2]).to(self.device).to(torch.float32)
        # rgb = event2rgb_(sample)
        # for i in range(rgb.shape[0]):
        #
        #     cv2.imshow('image', rgb[i])
        #     cv2.waitKey(1000)
        # pause()

        # print_info(sample, 'x')
        #
        # print_info(gt, 'target')
        return sample, gt

    def __len__(self):
        return self.length

def loader(datatype='bouncingball64',batch_size=16,shuffle=True):

    from utils.dataloader import DL_BouncingBall64
    dataset=None
    if datatype == 'bouncingball64':
        dataset=DL_BouncingBall64(HW=[64,64],
                        input_len=30,
                        set_size=10,
                        events_folder='E:/su/snn/code/dataset/bouncingball/bb64',
                        )
    elif datatype == 'bouncingball256':
        dataset=DL_BouncingBall64(HW=[64,64],
                        input_len=30,
                        set_size=10,
                        events_folder='E:/su/snn/code/dataset/bouncingball/bb64',
                        )

    loader=DataLoader(dataset=dataset,
                     batch_size=batch_size,
                     shuffle=shuffle)


    return loader

def getX(datatype='bouncingball64',input_len=30,index=1,is_batch=True,is_torch=False):
    from resource import PATH_BB64,PATH_BB256
    if datatype == 'bouncingball64':

        sample_path = os.path.join(PATH_BB64,str(index)+'.npy')
        sample_gt = np.load(sample_path)[0:input_len + 2]
        # sample_gt = torch.from_numpy(sample_gt).to(device).to(torch.float32)

    _, C, H, W=sample_gt.shape

    # ([T,C,H,W])

    sample = sample_gt[1:input_len + 1]
    gt = sample_gt[2:input_len + 2]

    if is_batch:
        # sample = sample.expand_dim(dim=0)
        # gt = gt.unsqueeze(dim=0)
        '''x shape应该是 [T, B, C, H, W]'''
        sample = np.expand_dims(sample, axis=1)
        gt = np.expand_dims(gt, axis=1) # [T, C, H, W] -> [T, B, C, H, W]
    if is_torch:
        sample = torch.from_numpy(sample).to(device).to(torch.float32)
        gt = torch.from_numpy(gt).to(device).to(torch.float32)

    return sample, gt
