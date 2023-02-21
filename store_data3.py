import os
import argparse
import numpy as np
import h5py
import pickle
from typing import Dict
import math
import torch
import cv2
import kornia as k
from tqdm import tqdm
import clip
VARIATION = "variation%d"
EPISODE = "episode%d"
DEMO_AUGMENTATION_EVERY_N = 10
device = "cuda:1" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

def load_img(path, frame):
        img = cv2.imread(path + f'/{frame}.png')
        return k.image_to_tensor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))/255.0 * 2 - 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-package_dir", type=str, default='datasets/rlbench/packaged/dataset/6')
    parser.add_argument("-raw_dir", type=str, default='datasets/rlbench/raw/dataset/6')
    parser.add_argument("-output_dir", type=str, default="datasets/rlbench/processed")
    parser.add_argument("-instruction_name", type=str, default="simple_tasks_instruction_clip_right.pkl")
    parser.add_argument("-tasks", type=tuple, default=("reach_target", ))
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    f = h5py.File(os.path.join(args.output_dir, 'simple_tasks_instruction_clip_right' + '.hdf5'), 'a')
    instructions: Dict[str, Dict[str, np.array]] = {}
    
    package_dir = os.listdir(args.package_dir)  # [stack_blocks+0, ...]
    package_dir_filter = [x for x in package_dir if not x.startswith("stack_blocks")]  # 过滤stack_blocks task
    train_trajs, train_idx, train_instruction = [], [0], [""]
    test_trajs, test_idx, test_instruction = [], [0], [""]
    
    for task in args.tasks:
        print(f"current task: %s" % task)
        package_dir = [x for x in package_dir_filter if x.startswith(task)]  # [reach_target+0, ...]
        raw_dir = os.path.join(args.raw_dir, task, VARIATION, 'episodes', EPISODE, 'front_rgb')
        raw_instruction = os.path.join(args.raw_dir, task, VARIATION, 'variation_descriptions.pkl')  # 语言路径
        package_dir.sort(key=lambda x:int(x[-2:]))
        
        task_iterator = tqdm(range(len(package_dir)), ascii=True, position=1)  # different task variations
        for task_dir, variation_id in zip(package_dir, task_iterator):  # reach_target+0, ...
            task_iterator.set_description("Dealing with {}".format(task_dir))
            np_data = os.listdir(os.path.join(args.package_dir, task_dir))  # * no 不排序,形成随机效果
            np_data.sort(key=lambda x:int(x[2:-4]))  # [ep0.npy, ep1.npy, ...]
            train_len = math.ceil(len(np_data) * 0.95)
            raw_instruction_path = raw_instruction % variation_id  # TODo future instructions
            raw_instruction_name = task + "&" + VARIATION % variation_id
            
            # * CLIP model for pre-trained language embeddings
            with open(raw_instruction_path, "rb") as fid:
                data = pickle.load(fid)
            text = clip.tokenize(data).to(device)
            with torch.no_grad():
                text_features = model.encode_text(text)
            if not instructions.get(task, None):
                instructions[task] = {}
            instructions[task][VARIATION % variation_id] = text_features.cpu().detach().numpy()
                
            for ind, p_name in enumerate(np_data):  # p_name: ep0.npy
                p_data_path = os.path.join(args.package_dir, task_dir, p_name)
                raw_data_path = raw_dir % (variation_id, ind)  # ind: episode0
                raw_imgs = os.listdir(raw_data_path)
                raw_imgs.sort(key=lambda x:int(x[:-4]))  # %.png
                stored_data = np.load(p_data_path, allow_pickle=True)
                key_frames_ids = stored_data[0].tolist()
                video = stored_data[1].tolist() 
                # traj = (torch.cat(video, dim=0) + 1 ) / 2  # T1CHW [0, 1]
                # traj = (traj * 255).ceil().int()  # int32
                # if ind < train_len:
                #         train_trajs.append(traj.numpy())  
                #         train_idx.append(len(traj))
                #         train_instruction.append(raw_instruction_name)
                # else:
                #     test_trajs.append(traj.numpy())
                #     test_idx.append(len(traj))
                #     test_instruction.append(raw_instruction_name)
                
                for i in range(len(raw_imgs) - 1):
                    if i % DEMO_AUGMENTATION_EVERY_N != 0:
                        continue
                    obs = load_img(raw_data_path, i)
                    # If our starting point is past one of the keypoints, then remove it
                    while len(key_frames_ids) > 0 and i >= key_frames_ids[0]:
                        key_frames_ids = key_frames_ids[1:]
                        video = video[1:]
                    if len(key_frames_ids) == 0:
                        break
                    key_frames_ids.insert(0, i)
                    video.insert(0, obs.unsqueeze(0).unsqueeze(0))
                    traj = (torch.cat(video, dim=0) + 1 ) / 2  # T1CHW [0, 1]
                    traj = (traj * 255).ceil().int()  # int32
                
                    if ind < train_len:
                        train_trajs.append(traj.numpy())  
                        train_idx.append(len(traj))
                        train_instruction.append(raw_instruction_name)
                    else:
                        test_trajs.append(traj.numpy())
                        test_idx.append(len(traj))
                        test_instruction.append(raw_instruction_name)
                    
    train_trajs = np.concatenate(train_trajs, axis=0) # (NT)1CHW
    test_trajs = np.concatenate(test_trajs, axis=0)
    train_idx = np.cumsum(np.array(train_idx))
    test_idx = np.cumsum(np.array(test_idx))
    train_instruction = np.array(train_instruction, dtype=object)
    test_instruction = np.array(test_instruction, dtype=object)
    
    f.create_dataset(f'train_data', data=train_trajs)
    print(f'\timages: {f[f"train_data"].shape}, {f[f"train_data"].dtype}')
    f.create_dataset(f'test_data', data=test_trajs)
    print(f'\timages: {f[f"test_data"].shape}, {f[f"test_data"].dtype}')
    f.create_dataset(f'train_idx', data=train_idx)
    f.create_dataset(f'test_idx', data=test_idx)
    f.create_dataset(f'train_instruction', data=train_instruction, dtype=h5py.special_dtype(vlen=str))
    f.create_dataset(f'test_instruction', data=test_instruction, dtype=h5py.special_dtype(vlen=str))
    f.close()
    
    with open(os.path.join(args.output_dir, args.instruction_name), "ab") as f:
        pickle.dump(instructions, f)
    print('Done!')