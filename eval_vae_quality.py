import numpy as np, os
import argparse, torch
from tqdm import tqdm
from omegaconf import OmegaConf
import imageio
from stage1_VAE.modules import resnet3D
from stage1_VAE.modules import decoder as net
from utils import auxiliaries as aux
from data.get_dataloder import get_loader
import warnings
warnings.filterwarnings("ignore")

def main(opt):
    """================= Create Model, Optimizer and Scheduler =========================="""
    decoder = net.Generator(opt.Decoder).cuda()
    checkpoint_name = 'models/bair/stage1/decoder.pth'  
    decoder.load_state_dict(torch.load(checkpoint_name)['state_dict'])  # TODO load pre-trained network
    encoder = resnet3D.Encoder(opt.Encoder).cuda()
    checkpoint_name = 'models/bair/stage1/encoder.pth'  
    encoder.load_state_dict(torch.load(checkpoint_name)['state_dict'])  # TODO load pre-trained network
    
    """==================== Dataloader ========================"""
    dataset       = get_loader(opt.Data['dataset'])
    train_dataset = dataset.Dataset(opt, mode='train')

    train_data_loader = torch.utils.data.DataLoader(train_dataset, num_workers=opt.Training['workers'],
                                                    batch_size=opt.Training['bs'], shuffle=True, drop_last=True)
    print("Batchsize for training: % 2d and for testing: % 2d" % (opt.Training['bs'], opt.Training['bs_eval']))
    ## Save video as gif
    save_path = f'./assets/results/rlbench/'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with torch.no_grad():
        for batch_idx, file_dict in enumerate(tqdm(train_data_loader)):
            if batch_idx == 5:
                break
            seq = file_dict["seq"].type(torch.FloatTensor).cuda()
            seq_orig = seq[:, 1:]
            motion, mu, covar = encoder(seq_orig.transpose(1, 2))
            seq_gen = decoder(seq[:, 0], motion)
            gif = aux.convert_seq2gif(seq_gen)
            imageio.mimsave(save_path + f'{batch_idx}_gen_results.gif', gif.astype(np.uint8), fps=3)
            gif = aux.convert_seq2gif(seq_orig)
            imageio.mimsave(save_path + f'{batch_idx}_org_results.gif', gif.astype(np.uint8), fps=3)
            print(f'Animations saved in {save_path}')
    
    
    
    
    
"""============================================"""
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--config", type=str, default='stage1_VAE/configs/rlbench_config.yaml', help="Define config file")
    parser.add_argument("-gpu", type=str, default='0')
    args = parser.parse_args()

    conf = OmegaConf.load(args.config)

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    aux.set_seed(42)
    main(conf)