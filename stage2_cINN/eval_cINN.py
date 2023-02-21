import argparse, os, torch, random
from tqdm import tqdm
import lpips, numpy as np
import torchvision
import wandb
import sys
sys.path.insert(0, '.')
import warnings
warnings.filterwarnings("ignore")
from data.get_dataloder import get_eval_loader
from get_model import Model
from metrics.FVD.evaluate_FVD import compute_fvd
from metrics.FID.FID_Score import calculate_FID
from metrics.FID.inception import InceptionV3
from metrics.DTFVD import DTFVD_Score
from utils.auxiliaries import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('-gpu', type=str, default='1', help="Define GPU on which to run")
parser.add_argument('-dataset', type=str, default='rlbench')
parser.add_argument('-texture', type=str, required=False, help='Specify texture when using DTDB')
parser.add_argument('-ckpt_path', type=str, required=False, help="Specify path if outside of repo for chkpt")
parser.add_argument('-data_path', type=str, default='datasets/rlbench/processed/simple_tasks_instruction_clip_right.hdf5', help="Path to dataset arranged as described in readme")
parser.add_argument('-seq_length', type=int, default=16)
parser.add_argument('-bs', type=int, default=1, help='Batchsize')  # 30
parser.add_argument('-control', type=bool, default=True, help='Language')
parser.add_argument('-FID', type=bool, default=False)
parser.add_argument('-FVD', type=bool, default=True)
parser.add_argument('-DTFVD', type=bool, default=True)
parser.add_argument('-LPIPS', type=bool, default=False)
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
set_seed(249)

## Load model from config
path_ds = f'{args.dataset}/{args.texture}/' if args.dataset == 'DTDB' else f'{args.dataset}'
ckpt_path = f'./models/{path_ds}/stage2/' if not args.ckpt_path else args.ckpt_path  # models/rlbench/stage2
save_path = f'./assets/results/rlbench/reach_target/instructions_clip_last_pad_right/'
if not os.path.exists(save_path):
    os.mkdir(save_path)
model = Model(ckpt_path, args.seq_length)

# set up dataloader
dataset = get_eval_loader(args.dataset, args.seq_length + 1, args.data_path, model.config)
dataloader = torch.utils.data.DataLoader(dataset, num_workers=1, batch_size=args.bs, shuffle=False)  # False workers=10

## Generate samples
clip = 50
count = 0
with torch.no_grad():
    for batch_idx, file_dict in enumerate(tqdm(dataloader)):
        seq = file_dict["seq"].type(torch.FloatTensor).cuda()  # (bs, seq_length, 3, 64, 64)
        task = file_dict["task"]
        wrong_variation = file_dict["wrong_variation"].item()
        current_variation = file_dict["current_variation"].item()
        if task != 'reach_target' and task != ['reach_target']:
            continue
        # if task != 'pick_and_lift' and task != ['pick_and_lift']:
        #     continue 
        # if task != 'pick_up_cup' and task != ['pick_up_cup']:
        #     continue 
        count += 1
        if count >= clip:
            break
        seq_gen = model(seq[:, 0]) if not args.control else model(seq[:, 0], file_dict["cond"].type(torch.FloatTensor).cuda())
        seq_gen_wrong = model(seq[:, 0]) if not args.control else model(seq[:, 0], file_dict["wrong_cond"].type(torch.FloatTensor).cuda())
        ## Save images
        seq_gen = torch.cat([seq[:, [0]], seq_gen], dim=1)  # * add initial frame
        seq_gen_wrong = torch.cat([seq[:, [0]], seq_gen_wrong], dim=1)
        torchvision.utils.save_image(torch.cat((seq[:, 0:].squeeze(0).cpu(), seq_gen[:, 0:].squeeze(0).detach().cpu()), dim=2),  # dim=2
                                    save_path + f'{batch_idx}_eval_recon_current_{current_variation}.jpg', normalize=True, nrow=17)
        torchvision.utils.save_image(torch.cat((seq[:, 0:].squeeze(0).cpu(), seq_gen_wrong[:, 0:].squeeze(0).detach().cpu()), dim=2),  # dim=2
                                    save_path + f'{batch_idx}_eval_recon_wrong_{wrong_variation}.jpg', normalize=True, nrow=17)

