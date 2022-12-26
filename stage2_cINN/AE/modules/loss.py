import torch.nn as nn
import torch
from stage2_cINN.AE.modules.LPIPS import LPIPS
import torch.nn.functional as F
import wandb


def calculate_adaptive_weight(nll_loss, g_loss, discriminator_weight, last_layer=None):
    if last_layer is not None:  # True
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    else:
        nll_grads = torch.autograd.grad(nll_loss, last_layer[0], retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer[0], retain_graph=True)[0]

    d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
    d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
    d_weight = d_weight * discriminator_weight
    return d_weight


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def adopt_weight(weight, epoch, threshold=0, value=0.):
    if epoch < threshold:
        weight = value
    return weight


class Loss(nn.Module):
    def __init__(self, dic):
        super(Loss, self).__init__()
        self.vgg_loss = LPIPS().cuda()  # pretrained LPIPS
        self.kl_weight = dic['w_kl']  # 1e-5
        self.disc_factor = 1
        self.disc_weight = 1
        self.logvar = nn.Parameter(torch.ones(size=()) * 0.0)
        self.disc_start = dic['pretrain']  # 20

    def forward(self, inp, generator, discriminator, optimizers, epoch, logger, training=True):  

        if training:  # True
            opt_gen, opt_disc = optimizers
        recon, _, p = generator(inp)  # generator inp:(bs, 3, 64, 64), recon是生成的img: (bs, 3, 64, 64), p是分布
        rec_loss = torch.abs(inp.contiguous() - recon.contiguous())  # (bs, 3, 64, 64)
        p_loss = self.vgg_loss(inp.contiguous(), recon.contiguous())  # (bs, 1, 1, 1)
        rec_loss = rec_loss + p_loss  # (bs, 3, 64, 64)

        kl_loss = p.kl()  # scaler

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar  # (bs, 3, 64, 64)
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]  # scaler

        # generator update
        logits_fake = discriminator(recon)  # (bs, 3, 64, 64) -> (bs, 1, 6, 6)
        g_loss = -torch.mean(logits_fake)  # scaler

        try:
            d_weight = calculate_adaptive_weight(nll_loss, g_loss, self.disc_weight, last_layer=list(generator.parameters())[-1])  # scaler
        except RuntimeError:
            assert not training
            d_weight = torch.tensor(0.0)

        disc_factor = adopt_weight(self.disc_factor, epoch, threshold=self.disc_start)  # 前20epoch为0,之后为self.disc_factor=1
        loss = nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

        if training:
            opt_gen.zero_grad()
            loss.backward()
            opt_gen.step()

        logits_real = discriminator(inp.contiguous().detach())  # (bs, 3, 64, 64) -> (bs, 1, 6, 6)
        logits_fake = discriminator(recon.contiguous().detach())

        disc_factor = adopt_weight(self.disc_factor, epoch, threshold=self.disc_start)  # 前20epoch为0,之后为self.disc_factor=1
        d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)

        if training and d_loss.item() > 0:  # 20epoch之后再更新判别器
            opt_disc.zero_grad()
            d_loss.backward()
            opt_disc.step()

        loss_dic = {
            "Loss": loss.item(),
            "Loss_recon": rec_loss.mean().item(),
            "Loss_nll": nll_loss.item(),
            "Logvar": self.logvar.detach().item(),
            "L_KL": kl_loss.item(),
            "Loss_G": g_loss.item(),
            "L_disc": d_loss.item(),
            "Logits_real": logits_real.mean().item(),
            "Logits_fake": logits_fake.mean().item(),
            "Disc_weight": d_weight.item(),
            "Disc_factor": disc_factor,
        }

        logger.append(loss_dic)

        prefix = 'train' if training else 'eval'
        loss_dic = {prefix + '_' + key:val for key, val in loss_dic.items()}
        wandb.log(loss_dic)

        return recon.cpu(), rec_loss.mean().item()

