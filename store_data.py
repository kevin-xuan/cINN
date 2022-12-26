import os
import argparse
import numpy as np
import h5py
import math
import torch
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-package_dir", type=str, default='datasets/rlbench/packaged/dataset/6')
    parser.add_argument("-output_dir", type=str, default="datasets/rlbench/processed")
    parser.add_argument("-task", type=str, default="stack_blocks")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    f = h5py.File(os.path.join(args.output_dir, args.task + '.hdf5'), 'a')
    
    package_dir = os.listdir(args.package_dir)  # [stack_blocks+0, ...]
    package_dir.sort(key=lambda x:int(x[-2:]))
    train_trajs, train_idx = [], [0]
    test_trajs, test_idx = [], [0]
    
    task_iterator = tqdm(range(len(package_dir)), ascii=True, position=1)  # 60 task variations
    for task_dir, _ in zip(package_dir, task_iterator):
        task_iterator.set_description("Dealing with {}".format(task_dir))
        np_data = os.listdir(os.path.join(args.package_dir, task_dir))  # 不排序,形成随机效果
        # np_data.sort(key=lambda x:int(x[2:-4]))  # [ep0.npy, ep1.npy, ...]
        train_len = math.ceil(len(np_data) * 0.95)
        # episode_iterator = tqdm(range(len(np_data)), ascii=True, position=1)  # 100 episodes
        for ind, p_name in enumerate(np_data):
            # episode_iterator.set_description("Dealing with {}".format(p_name))
            p_data_path = os.path.join(args.package_dir, task_dir, p_name)
            video = np.load(p_data_path, allow_pickle=True)[1] 
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
    