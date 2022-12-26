import cv2, torch, torch.nn as nn
import numpy as np, os
import pickle
import kornia as k
import h5py
from data.augmentation import Augmentation
import math

class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, mode):
        self.data_path = opt.Data['data_path']  # "datasets/rlbench/packaged/dataset/"
        self.mode = mode  # train or eval
        self.seq_length = opt.Data['sequence_length']  # * why 17? 第一帧x0用于与z一起生成v,目的是预测接下来的16帧,所以是17
        self.do_aug = opt.Data['aug']  # True
    
        print(f"Setup dataloder {mode}")
        self.data = h5py.File(self.data_path, 'r')      
        self._images = self.data[f'{self.mode}_data']
        self._idx = self.data[f'{self.mode}_idx']         
        self.length = len(self._idx) - 1
        for i in range(self.length):
            if self._idx[i+1] < self._idx[i]:
                print(i)
        if mode == 'train' and self.do_aug:  # train data需要增强
            self.aug = Augmentation(opt.Data['img_size'], opt.Data.Augmentation)
        else:
            self.aug = torch.nn.Sequential(
                        k.Resize(size=(opt.Data['img_size'], opt.Data['img_size'])),
                        k.augmentation.Normalize(0.5, 0.5))

    def __len__(self):
        return self.length


    def __getitem__(self, idx):
        start = self._idx[idx]
        end = self._idx[idx + 1]
        assert end - start >= 0, "Invalid idx:{} with start:{} and end: {}".format(idx, start, end)
        seq = torch.tensor(self._images[start:end]).float() / 255.0  # (frames_num, 1, 3, 128, 128) [0, 1]
        frames_num = len(seq)
        image_type = seq.size(1)
            
        ## Load sequence
        if self.seq_length > 1:  # video sequence instead of initial frame
            num_padding = self.seq_length - 2
            # seq_padding = np.repeat(seq[-1:].numpy(), num_padding, axis=0)  # * 填充最后一帧图片
            seq_padding = np.repeat(torch.zeros(seq[-1:].shape).numpy(), num_padding, axis=0)  # * 填充全黑图片
            seq = torch.from_numpy(np.concatenate([seq.numpy(), seq_padding], axis=0))  # self.seq_length + frames_num - 2
        
        if self.seq_length > 1:
            start = np.random.randint(0, frames_num - 1)
        else:
            start = np.random.randint(0, frames_num)
         
        select_type = np.random.randint(0, image_type)
        seq = seq[start:start + self.seq_length, select_type, ...]  # (frames_num, 3, 128, 128)
        
        return {'seq': self.aug(seq), 'start_frame': start, 'last_frame': frames_num-1}

