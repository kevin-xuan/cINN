import cv2, torch, torch.nn as nn
import numpy as np, os
import pickle
import kornia as k
from data.augmentation import Augmentation
import math

class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, mode, variation=[6]):
        self.data_path = opt.Data['data_path']  # "datasets/rlbench/packaged/dataset/"
        self.mode = mode  # train or eval
        self.seq_length = opt.Data['sequence_length']  # * why 17? 第一帧x0用于与z一起生成v,目的是预测接下来的16帧,所以是17
        self.do_aug = opt.Data['aug']  # True
        # self.variation = variation if mode == 'train' else [5]
        self.variation = variation

        print(f"Setup dataloder {mode}")
        self.videos = []
        # with open(self.data_path + '/' + 'instructions.pkl') as f:
        #     self.instructions = pickle.load(f)
        for seed in self.variation:
            videos = os.listdir(self.data_path + "/" + str(seed))
            for vid in videos:  # e.g., stack_blocks+0
                subvideos = os.listdir(self.data_path + "/" + str(seed) + '/' + vid + '/')  # [epX.npy, ...]
                upper = math.ceil(len(subvideos) * 0.95)  # 0.95不行,试试0.8,也不行,没有任何变化, loss与数据增强有关
                if mode =='train':
                    for svid in subvideos[:upper]:
                        self.videos.append(str(seed) + '/' + vid + '/' + svid)  # e.g., 6/stack_blocks+0/ep1.npy
                        # self.videos.append(np.load(os.path.join(self.data_path, str(seed) + '/' + vid + '/' + svid), allow_pickle=True)[1])
                else:
                    for svid in subvideos[upper:]:
                        self.videos.append(str(seed) + '/' + vid + '/' + svid)  # e.g., 6/stack_blocks+0/ep1.npy
                        # self.videos.append(np.load(os.path.join(self.data_path, str(seed) + '/' + vid + '/' + svid), allow_pickle=True)[1])
                
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
        video  = np.load(os.path.join(self.data_path, self.videos[idx]), allow_pickle=True)  # xxx/epX.npy
        images = video[1] # [(1, 1, 3, 128, 128), ...]  # 1个front rgb camera
        # images = self.videos[idx]
        image_type = images[0].size(1)  # 图片类型
        seq = (torch.cat(images, dim=0) + 1) / 2  # (frames_num, 1, 3, 128, 128) [-1, 1] -> [0, 1]
        frames_num = len(seq)
            
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
        seq = seq[start:start + self.seq_length, select_type, ...]  # (seq_length, 3, 128, 128)
        
        return {'seq': self.aug(seq), 'start_frame': start, 'last_frame': len(frames_num)-1}

