import torch.nn as nn, torch
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from stage1_VAE.modules.normalization_layer import Spade, ADAIN, Norm3D


class GeneratorBlock(nn.Module):

    def __init__(self, n_in, n_out, use_spectral, z_dim):
        super().__init__()
        self.learned_shortcut = (n_in != n_out)  # False or True
        n_middle = min(n_in, n_out)  # n_out

        self.conv_0 = nn.Conv3d(n_in, n_middle, 3, 1, 1)
        self.conv_1 = nn.Conv3d(n_middle, n_out, 3, 1, 1)

        if self.learned_shortcut:
            self.conv_s = nn.Conv3d(n_in, n_out, 1, bias=False)

        if use_spectral:  # True
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)

            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        self.norm_0 = Spade(n_in)
        self.norm_1 = ADAIN(n_middle, z_dim)  # z_dim=64

        if self.learned_shortcut:
            self.norm_s = Norm3D(n_in)

    def forward(self, x, cond1, cond2):

        x_s = self.shortcut(x)

        dx = self.conv_0(self.actvn(self.norm_0(x, cond2)))
        dx = self.conv_1(self.actvn(self.norm_1(dx, cond1)))

        out = x_s + dx

        return out

    def shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)


class Generator(nn.Module):
    def __init__(self, dic):
        super().__init__()

        nf = dic['channel_factor']  # 64
        self.z_dim = dic['z_dim']  # 64
        self.fmap_start = 16*nf  # 1024

        self.fc = nn.Linear(dic['z_dim'], 4*4*16*nf)  # Linear(64, 16384)

        use_spectral = dic["spectral_norm"]  # True
        self.upsample_s = dic["upsample_s"]  # [2, 1]
        self.upsample_t = dic["upsample_t"]  # [2, 1]

        self.fc = nn.Linear(self.z_dim, 4*4*16*nf)  # Linear(64, 16384), 为什么2个fc?
        self.head_0 = GeneratorBlock(16*nf, 16*nf, use_spectral, self.z_dim)

        self.fc = nn.Linear(self.z_dim, 4*4*16*nf)  # why, 第三次了?
        self.head_0 = GeneratorBlock(16*nf, 16*nf, use_spectral, self.z_dim)

        self.g_0 = GeneratorBlock(16*nf, 16*nf, use_spectral, self.z_dim)
        self.g_1 = GeneratorBlock(16*nf, 8*nf, use_spectral, self.z_dim)
        self.g_2 = GeneratorBlock(8*nf, 4*nf, use_spectral, self.z_dim)
        self.g_3 = GeneratorBlock(4*nf, 2*nf, use_spectral, self.z_dim)
        self.g_4 = GeneratorBlock(2*nf, 1*nf, use_spectral, self.z_dim)

        self.conv_img = nn.Conv3d(nf, 3, 3, padding=1)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data, gain=0.02)
            # nn.init.orthogonal_(m.weight.data, gain=0.02)
            if not isinstance(m.bias, type(None)):
                nn.init.constant_(m.bias.data, 0)

    def reset_params(self):
        for _, m in enumerate(self.modules()):
            self.weight_init(m)

    def forward(self, img, motion):  # img: (bs, 3, 64, 64), motion (z): (bs, 64), e.g., bs=6

        x = self.fc(motion).reshape(img.size(0), -1, 1, 4, 4)  # (6, 1024, 1, 4, 4)
        x = self.head_0(x, motion, img)  # (6, 1024, 1, 4, 4)

        x = F.interpolate(x, scale_factor=2)  # (6, 1024, 2, 8, 8)
        x = self.g_0(x, motion, img)

        x = F.interpolate(x, scale_factor=2)  # (6, 1024, 4, 16, 16)
        x = self.g_1(x, motion, img)  # (6, 512, 4, 16, 16)

        x = F.interpolate(x, scale_factor=2)  # (6, 512, 8, 32, 32)
        x = self.g_2(x, motion, img)  # (6, 256, 8, 32, 32)

        x = F.interpolate(x, scale_factor=(self.upsample_t[0], self.upsample_s[0], self.upsample_s[0]))  # (6, 256, 16, 64, 64)
        x = self.g_3(x, motion, img)  # (6, 128, 16, 64, 64)

        x = F.interpolate(x, scale_factor=(self.upsample_t[1], self.upsample_s[1], self.upsample_s[1]))
        x = self.g_4(x, motion, img)  # (6, 64, 16, 64, 64)

        x = self.conv_img(F.leaky_relu(x, 2e-1))  # (6, 3, 16, 64, 64)
        x = torch.tanh(x)

        return x.transpose(1, 2)  # (6, 16, 3, 64, 64)
