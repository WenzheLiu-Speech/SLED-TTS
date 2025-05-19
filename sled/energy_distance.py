import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math
import pdb


class ScoreLossZ(nn.Module):
    """Energy Distance Loss"""
    def __init__(self, target_channels, z_channels, depth, width, beta=1, gamma=1, noise_channels=16):
        super(ScoreLossZ, self).__init__()
        self.noise_channels = noise_channels
        self.net = SimpleMLPAdaLN(
            in_channels=self.noise_channels,
            model_channels=width,
            out_channels=target_channels,
            z_channels=z_channels,
            num_res_blocks=depth,
        )
        self.beta = beta
        self.gamma = gamma


    def forward(self, target, z, mask=None, additional_targets=None):
        noise_1 = torch.randn((z.shape[0], self.noise_channels), dtype=z.dtype, device=z.device)
        sample_1 = self.net(noise_1, z)
        noise_2 = torch.randn((z.shape[0], self.noise_channels), dtype=z.dtype, device=z.device)
        sample_2 = self.net(noise_2, z)

        score = self.energy_score(sample_1, sample_2, target, additional_targets)
        loss = - score
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()

    def energy_distance(self, x_1, x_2):
        return torch.pow(torch.linalg.norm(x_1 - x_2, ord=2, dim=-1), self.beta)

    def energy_score(self, sample_1, sample_2, target, additional_targets = None):
        distance_1 = self.energy_distance(sample_1, target)
        distance_2 = self.energy_distance(sample_2, target)
        variance = self.energy_distance(sample_1, sample_2)
        score = variance - distance_1 - distance_2
        return score

    def kernel_distance(self, x_1, x_2):
        return - torch.exp(- torch.sum(torch.pow(x_1 - x_2, 2), dim = -1).div(2 * self.gamma**2))

    def kernel_score(self, sample_1, sample_2, target):
        distance_1 = self.kernel_distance(sample_1, target)
        distance_2 = self.kernel_distance(sample_2, target)
        variance = self.kernel_distance(sample_1, sample_2)
        score = variance - distance_1 - distance_2
        return score

    def sample(self, z, temperature=1.0, cfg=1.0):
        if cfg != 1.0:
            z_1, z_2 = z.chunk(2, dim=0)
            z = z_1 * cfg + (1 - cfg) * z_2

        noise = torch.randn((z.shape[0], self.noise_channels), dtype=z.dtype, device=z.device)
        return self.net(noise, z)



def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels, bias=True),
            nn.SiLU(),
            nn.Linear(channels, channels, bias=True),
        )
        self.noise_ln = nn.LayerNorm(channels, eps=1e-6, elementwise_affine=True)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(self.noise_ln(y)).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Energy Distance Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        grad_checkpointing=False
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing

        self.cond_embed = nn.Linear(z_channels, model_channels)

        self.input_proj = nn.Linear(in_channels, model_channels)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            module._is_hf_initialized = True
        self.apply(_basic_init)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C x ...] Tensor of outputs.
        """
        y = self.input_proj(x)
        x = self.cond_embed(c)



        if self.grad_checkpointing and not torch.jit.is_scripting():
            raise NotImplementedError
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
