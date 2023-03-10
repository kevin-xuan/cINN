import cv2, torch, torch.nn as nn
import numpy as np, os
import kornia as k
from data.augmentation import Augmentation

class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, mode):
        self.data_path = opt.Data['data_path']
        self.mode = mode  # train or eval
        self.seq_length = opt.Data['sequence_length']  # * why 17? 第一帧x0用于与z一起生成v,目的是预测接下来的16帧,所以是17
        self.do_aug = opt.Data['aug']  # True

        print(f"Setup dataloder {mode}")
        self.videos = []
        videos = os.listdir(self.data_path + mode + '/')
        for vid in videos:  # e.g., traj_0_to_255
            subvideos = os.listdir(self.data_path + mode + '/' + vid + '/')  # [1, 256]
            for svid in subvideos:
                self.videos.append(mode + '/' + vid + '/' + svid + '/')

        self.length = len(self.videos)

        if mode == 'train' and self.do_aug:  # train data需要增强
            self.aug = Augmentation(opt.Data['img_size'], opt.Data.Augmentation)
        else:
            self.aug = torch.nn.Sequential(
                        k.Resize(size=(opt.Data['img_size'], opt.Data['img_size'])),
                        k.augmentation.Normalize(0.5, 0.5))

    def __len__(self):
        return self.length

    def load_img(self, video, frame):
        img = cv2.imread(self.data_path + video + f'/{frame}.png')
        return k.image_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))/255.0

    def __getitem__(self, idx):
        video  = self.videos[idx]
        frames = np.arange(0, 30)

        ## Load sequence
        start = 0 if self.mode == 'test' else np.random.randint(0, 30 - self.seq_length + 1)
        seq = torch.stack([self.load_img(video, frames[start + i]) for i in range(self.seq_length)], dim=0)
        return {'seq': self.aug(seq)}

