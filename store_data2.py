import os
import argparse
import numpy as np
import h5py
import math
import torch
import cv2
import kornia as k
from tqdm import tqdm
VARIATION = "variation%d"
EPISODE = "episode%d"
DEMO_AUGMENTATION_EVERY_N = 10

def load_img(path, frame):
        img = cv2.imread(path + f'/{frame}.png')
        return k.image_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))/255.0 * 2 - 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-package_dir", type=str, default='datasets/rlbench/packaged/dataset/6')
    parser.add_argument("-raw_dir", type=str, default='datasets/rlbench/raw/dataset/6')
    parser.add_argument("-output_dir", type=str, default="datasets/rlbench/processed")
    parser.add_argument("-tasks", type=tuple, default=("pick_and_lift", "pick_up_cup", "push_button", "reach_target"))
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    f = h5py.File(os.path.join(args.output_dir, 'simple_tasks' + '.hdf5'), 'a')
    
    package_dir = os.listdir(args.package_dir)  # [stack_blocks+0, ...]
    package_dir_filter = [x for x in package_dir if not x.startswith("stack_blocks")]
    train_trajs, train_idx = [], [0]
    test_trajs, test_idx = [], [0]
    
    for task in args.tasks:
        print(f"current task: %s" % task)
        package_dir = [x for x in package_dir_filter if x.startswith(task)]  # [reach_target+0, ...]
        raw_dir = os.path.join(args.raw_dir, task, VARIATION, 'episodes', EPISODE, 'front_rgb')
        package_dir.sort(key=lambda x:int(x[-2:]))
        
        task_iterator = tqdm(range(len(package_dir)), ascii=True, position=1)  # 60 task variations
        for task_dir, variation_id in zip(package_dir, task_iterator):
            task_iterator.set_description("Dealing with {}".format(task_dir))
            np_data = os.listdir(os.path.join(args.package_dir, task_dir))  # * no 不排序,形成随机效果
            np_data.sort(key=lambda x:int(x[2:-4]))  # [ep0.npy, ep1.npy, ...]
            train_len = math.ceil(len(np_data) * 0.95)
            for ind, p_name in enumerate(np_data):
                p_data_path = os.path.join(args.package_dir, task_dir, p_name)
                raw_data_path = raw_dir % (variation_id, ind)  # int(task_dir.split('+')[1])
                # TODo future instructions
                raw_imgs = os.listdir(raw_data_path)
                raw_imgs.sort(key=lambda x:int(x[:-4]))  # %.png
                stored_data = np.load(p_data_path, allow_pickle=True)
                key_frames_ids = stored_data[0].tolist()
                video = stored_data[1].tolist() 
                # for i in range(len(raw_imgs) - 1):
                #     if i % DEMO_AUGMENTATION_EVERY_N != 0:
                #         continue
                #     obs = load_img(raw_data_path, i)
                #     # If our starting point is past one of the keypoints, then remove it
                #     while len(key_frames_ids) > 0 and i >= key_frames_ids[0]:
                #         key_frames_ids = key_frames_ids[1:]
                #         video = video[1:]
                #     if len(key_frames_ids) == 0:
                #         break
                #     key_frames_ids.insert(0, i)
                #     video.insert(0, obs.unsqueeze(0).unsqueeze(0))
                #     traj = (torch.cat(video, dim=0) + 1 ) / 2  # T1CHW [0, 1]
                #     traj = (traj * 255).ceil().int()  # int32
                #     if ind < train_len:
                #         train_trajs.append(traj.numpy())  
                #         train_idx.append(len(traj))
                #     else:
                #         test_trajs.append(traj.numpy())
                #         test_idx.append(len(traj))
                traj = (torch.cat(video, dim=0) + 1 ) / 2  # T1CHW [0, 1]
                traj = (traj * 255).ceil().int()  # int32
                if ind < train_len:
                    train_trajs.append(traj.numpy())  
                    train_idx.append(len(traj))
                else:
                    test_trajs.append(traj.numpy())
                    test_idx.append(len(traj))
                    
    train_trajs = np.concatenate(train_trajs, axis=0) # (NT)1CHW
    test_trajs = np.concatenate(test_trajs, axis=0)
    train_idx = np.cumsum(np.array(train_idx))
    test_idx = np.cumsum(np.array(test_idx))
    
    f.create_dataset(f'train_data', data=train_trajs)
    print(f'\timages: {f[f"train_data"].shape}, {f[f"train_data"].dtype}')
    f.create_dataset(f'test_data', data=test_trajs)
    print(f'\timages: {f[f"test_data"].shape}, {f[f"test_data"].dtype}')
    f.create_dataset(f'train_idx', data=train_idx)
    f.create_dataset(f'test_idx', data=test_idx)
    print('Done!')