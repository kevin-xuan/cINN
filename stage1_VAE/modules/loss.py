import torch.nn as nn
import torch
import numpy as np
import wandb

from pytorch_lightning.metrics.functional import ssim, psnr
from stage2_cINN.AE.modules.LPIPS import LPIPS


def KL(mu, logvar):
    ## computes KL-divergence loss between NormalGaussian and parametrized learned distribution
    return -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), axis=1))

def fmap_loss(fmap1, fmap2, metric):
    recp_loss = 0
    for idx in range(len(fmap1)):
        if metric == 'L1':
            recp_loss += torch.mean(torch.abs((fmap1[idx] - fmap2[idx])))
        if metric == 'L2':
            recp_loss += torch.mean((fmap1[idx] - fmap2[idx]) ** 2)
    return recp_loss / len(fmap1)


def hinge_loss(fake_data, orig_data, update):
    ## hinge loss implementation
    ## update determines if loss should be computed for generator or discrimnator
    if update == 'disc':
        L_disc1 = torch.mean(torch.nn.ReLU()(1.0 - orig_data))
        L_disc2 = torch.mean(torch.nn.ReLU()(1.0 + fake_data))
        return (L_disc1 + L_disc2) / 2
    elif update == 'gen':
        return -torch.mean(fake_data)


def gradient_penalty(pred, x):  # pred:(bs, 1) x:(bs, T, 3, 64, 64)
    batch_size = x.size(0)
    grad_dout = torch.autograd.grad(
                    outputs=pred.mean(), inputs=x, allow_unused=True,
                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_dout2 = grad_dout.pow(2)  # (bs, T, 3, 64, 64)
    assert (grad_dout2.size() == x.size())
    reg = grad_dout2.reshape(batch_size, -1).sum(1)
    return reg.mean()  # (bs, )


## Backward incl Loss
class Backward(nn.Module):
    def __init__(self, opt):
        super(Backward, self).__init__()
        self.dic = opt
        self.w_kl = opt.Training['w_kl']
        self.lpips = LPIPS().cuda()
        self.gan_loss = opt.Training['GAN_Loss']
        self.w_coup_t = opt.Training['w_coup_t']
        self.w_fmap_t = opt.Training['w_fmap_t']
        self.w_coup_s = opt.Training['w_coup_s']
        self.subsample_length = opt.Training['subsample_length']
        self.w_mse = opt.Training['w_recon']
        self.w_GP = opt.Training['w_GP']
        self.w_percep = opt.Training['w_percep']
        self.seq_length = opt.Data['sequence_length']
        self.pretrain = opt.Training['pretrain']

    def forward(self, decoder, encoder, disc_t, disc_s, seq_o, optimizers, epoch, logger, file_dict):

        opt_all, opt_d_t, opt_d_s = optimizers

        ## Perform forward pass through network
        seq_orig = seq_o[:, 1:]  # (bs, 16, 3, 64, 64)
        motion, mu, covar = encoder(seq_orig.transpose(1, 2))  # (bs, 64)
        seq_gen = decoder(seq_o[:, 0], motion)  # (bs, 16, 3, 64, 64)

        seq_gen_trunc = torch.cat([seq_gen[i, :min(last - start, self.seq_length-1)] for i, start, last in zip(range(seq_gen.size(0)), file_dict['start_frame'], file_dict['last_frame'])])
        seq_orig_trunc = torch.cat([seq_orig[i, :min(last - start, self.seq_length-1)] for i, start, last in zip(range(seq_gen.size(0)), file_dict['start_frame'], file_dict['last_frame'])])
        
        ## PSNR
        # PSNR = psnr(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:]))
        PSNR = psnr(seq_orig_trunc, seq_gen_trunc)

        ## SSIM
        # SSIM = ssim(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:]))
        SSIM = ssim(seq_orig_trunc, seq_gen_trunc)

        # Subsample 16 frames if sequence length is bigger than or equal to 16
        # if seq_gen.size(1) >= 17:  # * 默认16, True,这里我改成17
        #     length = self.subsample_length  # 12
        #     rand_start = np.random.randint(0, seq_gen.size(1) - length + 1)
        #     seq_fake = seq_gen[:, rand_start:rand_start+length]
        #     seq_real = seq_orig[:, rand_start:rand_start+length]
        # else:
        #     seq_fake = seq_gen
        #     seq_real = seq_orig
        
        seq_fake = seq_gen
        seq_real = seq_orig
        
        
        ## Sample subset of images from sequence for spatial discriminator
        # rand_k = np.random.randint(0, seq_orig.size(0) * seq_orig.size(1), 20)
        # data_fake = torch.cat([seq_gen.reshape(-1, *seq_gen.shape[2:])[i].unsqueeze(0) for i in rand_k])  # (20, 3, 64, 64)
        # data_real = torch.cat([seq_orig.reshape(-1, *seq_orig.shape[2:])[i].unsqueeze(0) for i in rand_k])
        
        rand_k = np.random.randint(0, seq_gen_trunc.size(0), min(20, seq_gen_trunc.size(0)))
        data_fake = torch.cat([seq_gen_trunc[i].unsqueeze(0) for i in rand_k])  # (20, 3, 64, 64)
        data_real = torch.cat([seq_orig_trunc[i].unsqueeze(0) for i in rand_k])

        ## Update (temporal) 3D discriminator
        if self.w_GP:  # 10 True
            seq_real.requires_grad_() ## needs to be set to true due to GP

        pred_gen_t, _ = disc_t(seq_fake.transpose(1, 2).detach())  # (bs, 1)
        pred_orig_t, _ = disc_t(seq_real.transpose(1, 2))
        # pred_gen_t = torch.cat([disc_t(seq_fake[[i], :min(last - start, self.seq_length-1)].transpose(1, 2).detach())[0] for i, start, last in zip(range(seq_gen.size(0)), file_dict['start_frame'], file_dict['last_frame'])])
        # pred_orig_t = torch.cat([disc_t(seq_real[[i], :min(last - start, self.seq_length-1)].transpose(1, 2).detach())[0] for i, start, last in zip(range(seq_gen.size(0)), file_dict['start_frame'], file_dict['last_frame'])])
        L_d_t = hinge_loss(pred_gen_t, pred_orig_t, update='disc')
        if self.w_GP:  # 10 True
            # L_GP = gradient_penalty(pred_orig_t, seq_real)
            L_GP = torch.zeros(1).to(L_d_t.device)
        else:
            L_GP = torch.zeros(1)

        if epoch >= self.pretrain:  # pretrain=1, 即第一轮不需要梯度清零
            opt_d_t.zero_grad()
            (L_d_t + self.w_GP * L_GP).backward()
            opt_d_t.step()

        ## Update spatial discriminator (patch disc)
        pred_gen_s = disc_s(data_fake.detach())  # (20, 1, 6, 6)
        pred_orig_s = disc_s(data_real)
        L_d_s = hinge_loss(pred_gen_s, pred_orig_s, update='disc')
        if epoch >= self.pretrain:
            opt_d_s.zero_grad()
            L_d_s.backward()
            opt_d_s.step()

        ## Update VAE
        Loss_VAE = 0
        pred_gen_s = disc_s(data_fake)  # (20, 1, 6, 6)
        loss_gen_s = hinge_loss(pred_gen_s, pred_orig_s, update='gen')
        if epoch >= self.pretrain:
            Loss_VAE += loss_gen_s

        pred_gen_t, fmap_gen_t  = disc_t(seq_fake.transpose(1, 2))
        pred_orig_t, fmap_orig_t = disc_t(seq_real.transpose(1, 2))
        # _, fmap_gen_t  = disc_t(seq_fake.transpose(1, 2))
        # _, fmap_orig_t = disc_t(seq_real.transpose(1, 2))
        # pred_gen_t, pred_orig_t = [], []
        # for i, start, last in zip(range(seq_gen.size(0)), file_dict['start_frame'], file_dict['last_frame']):
        #     x_gen, y_gen = disc_t(seq_fake[[i], :min(last - start, self.seq_length-1)].transpose(1, 2))
        #     x_orig, y_orig = disc_t(seq_real[[i], :min(last - start, self.seq_length-1)].transpose(1, 2))
        #     pred_gen_t.append(x_gen)
        #     # fmap_gen_t.append(y_gen)
        #     pred_orig_t.append(x_orig)
        #     # fmap_orig_t.append(y_orig)  
        # pred_gen_t = torch.cat(pred_gen_t)
        # pred_orig_t = torch.cat(pred_orig_t)
        coup_t = hinge_loss(pred_gen_t, pred_orig_t, update='gen')

        ## Feature Map Loss
        L_fmap_t = fmap_loss(fmap_gen_t, fmap_orig_t, metric='L1')

        ## Generator loss
        L_temp = self.w_coup_t * coup_t + self.w_fmap_t * L_fmap_t
        if epoch >= self.pretrain:
            Loss_VAE += L_temp

        # LPIPS = self.lpips(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:])).mean()
        LPIPS = self.lpips(seq_orig_trunc, seq_gen_trunc).mean()

        ## L1 Error
        # L_recon = torch.mean(torch.abs((seq_gen - seq_orig)))
        L_recon = torch.mean(torch.abs((seq_gen_trunc - seq_orig_trunc)))

        ## KL Loss
        L_kl = KL(mu, covar)

        Loss_VAE += self.w_percep * LPIPS + self.w_kl * L_kl + self.w_mse * L_recon

        opt_all.zero_grad()
        Loss_VAE.backward()
        opt_all.step()

        loss_dic = {
            ## Losses for VAE
            "Loss_VAE": Loss_VAE.item(),
            "Loss_L1": L_recon.item(),
            "LPIPS": LPIPS.item(),
            "Loss_KL": L_kl.item(),
            "Loss_GEN_S": loss_gen_s.item(),
            "Loss_GEN_T": coup_t.item(),
            ## Losses for temporal discriminator
            "Loss_Disc_T": L_d_t.item(),
            "Loss_Fmap_T": L_fmap_t.item(),
            "L_GP": L_GP.item(),
            "Logits_Real_T": pred_orig_t.mean().item(),
            "Logits_Fake_T": pred_gen_t.mean().item(),
            ## Losses for spatial discriminator
            "Loss_Disc_S": L_d_s.item(),
            "Logits_Real_S": pred_orig_s.mean().item(),
            "Logits_Fake_S": pred_gen_s.mean().item(),
            ## Additional
            "PSNR": PSNR.item(),
            "SSIM": SSIM.item(),
        }

        ## Log dic online and offline
        wandb.log(loss_dic)
        logger.append(loss_dic)

        return [seq_gen.detach().cpu(), seq_orig.cpu()]


    def eval(self, decoder, encoder, seq_o, logger, file_dict):

        ## Perform forward pass through network
        seq_orig = seq_o[:, 1:]
        motion, mu, covar = encoder(seq_orig.transpose(1, 2))
        seq_gen = decoder(seq_o[:, 0], motion)
        seq_gen_trunc = torch.cat([seq_gen[i, :min(last - start, self.seq_length-1)] for i, start, last in zip(range(seq_gen.size(0)), file_dict['start_frame'], file_dict['last_frame'])])
        seq_orig_trunc = torch.cat([seq_orig[i, :min(last - start, self.seq_length-1)] for i, start, last in zip(range(seq_gen.size(0)), file_dict['start_frame'], file_dict['last_frame'])])
        ## PSNR
        # PSNR = psnr(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:]))
        PSNR = psnr(seq_orig_trunc, seq_gen_trunc)

        ## SSIM
        # SSIM = ssim(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:]))
        SSIM = ssim(seq_orig_trunc, seq_gen_trunc)

        # LPIPS = self.lpips(seq_orig.reshape(-1, *seq_orig.shape[2:]), seq_gen.reshape(-1, *seq_gen.shape[2:]))
        LPIPS = self.lpips(seq_orig_trunc, seq_gen_trunc)

        ## L1 Error
        # L_recon = torch.mean(torch.abs((seq_gen - seq_orig)))
        L_recon = torch.mean(torch.abs((seq_gen_trunc - seq_orig_trunc)))

        ## KL Loss
        L_kl = KL(mu, covar)

        loss_dic = {"Loss_L1": L_recon.item(),
                    "LPIPS": LPIPS.mean().item(),
                    "L_KL": L_kl.item(),
                    "PSNR": PSNR.item(),
                    "SSIM": SSIM.item()
        }

        ## Log dic online and offline
        logger.append(loss_dic)
        loss_dic = {'eval_' + key:val for key, val in loss_dic.items()}
        wandb.log(loss_dic)

        return [seq_gen.detach().cpu(), seq_orig.cpu()]



