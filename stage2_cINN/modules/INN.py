import torch, numpy as np, os
import torch.nn as nn
from stage2_cINN.modules.flow_blocks import ConditionalFlow
from stage2_cINN.modules.modules import BasicFullyConnectedNet
from stage2_cINN.AE.modules.AE import BigAE, ResnetEncoder
from omegaconf import OmegaConf

class SupervisedTransformer(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        in_channels = kwargs["flow_in_channels"]  # 64
        mid_channels = kwargs["flow_mid_channels"]  # 512
        hidden_depth = kwargs["flow_hidden_depth"]  # 2
        n_flows = kwargs["n_flows"]  # 20
        conditioning_option = kwargs["flow_conditioning_option"]  # None
        embedding_channels = (
            kwargs["flow_embedding_channels"]
            if "flow_embedding_channels" in kwargs
            else kwargs["flow_in_channels"]
        )  # 64

        self.control = kwargs["control"]  # False
        self.cond_size = 10 if self.control else 0  # 0

        self.flow = ConditionalFlow(
            in_channels=in_channels,
            embedding_dim=embedding_channels + self.cond_size*3,
            hidden_dim=mid_channels,
            hidden_depth=hidden_depth,
            n_flows=n_flows,
            conditioning_option=conditioning_option,
            control=self.control
        )

        dic = kwargs['dic']
        model_path = dic['model_path'] + dic['model_name'] + '/'  # './models/bair/stage2/'
        config = OmegaConf.load(model_path + 'config_stage2_AE.yaml')
        self.embedder = ResnetEncoder(config.AE).cuda()
        self.embedder.load_state_dict(torch.load(model_path + dic['checkpoint_name'] + '.pth')['state_dict'])
        _ = self.embedder.eval()

    def sample(self, shape, cond):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        z_tilde = torch.randn(shape).to(device)
        sample = self.reverse(z_tilde, cond)
        return sample

    def embed_pos(self, pos):
        pos = pos * self.cond_size - 1e-4
        embed1 = torch.zeros((pos.size(0), self.cond_size))
        embed2 = torch.zeros((pos.size(0), self.cond_size))
        embed3 = torch.zeros((pos.size(0), self.cond_size))
        embed1[np.arange(embed1.size(0)), pos[:, 0].long()] = 1
        embed2[np.arange(embed2.size(0)), pos[:, 1].long()] = 1
        embed3[np.arange(embed3.size(0)), pos[:, 2].long()] = 1
        return torch.cat((embed1, embed2, embed3), dim=1).cuda()

    def forward(self, input, cond, reverse=False, train=False):
        # input: (bs, 64), cond: [(bs, 3, 64, 64), None], reverse: True or False
        with torch.no_grad():
            embed = self.embedder.encode(cond[0]).mode().reshape(input.size(0), -1).detach()  # (bs, 64, 1, 1) -> (bs, 64)
            embed = torch.cat((embed, self.embed_pos(cond[1])), dim=1) if self.control else embed

        if reverse:  # True or False
            return self.reverse(input, embed)

        out, logdet = self.flow(input, embed)

        return out, logdet  # (bs, 64, 1, 1) (bs)

    def reverse(self, out, cond):
        return self.flow(out, cond, reverse=True)


