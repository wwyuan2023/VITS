
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm, remove_weight_norm

from commons import init_weights, get_padding


LRELU_SLOPE = 0.1


class Swish(nn.Module):
    
    __constants__ = ['num_parameters']
    num_parameters: int
    
    def __init__(self, num_parameters=1, init=1.0, device=None, dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_parameters = num_parameters
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(num_parameters, **factory_kwargs).fill_(init))
    
    def forward(self, x):
        return x * torch.sigmoid(self.weight * x)
    
    def extra_repr(self) -> str:
        return 'num_parameters={}'.format(self.num_parameters)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConditionalLayerNorm(nn.Module):

    __constants__ = ['input_size', 'embed_size', 'eps']
    input_size: int
    embed_size: int
    eps: float

    def __init__(
        self,
        input_size,
        embed_size,
        eps=1e-5,
    ):
        super(ConditionalLayerNorm, self).__init__()

        self.input_size = input_size
        self.embed_size = embed_size
        self.eps = eps
        self.scale_layer = nn.Linear(embed_size, input_size, bias=True)
        self.bias_layer = nn.Linear(embed_size, input_size, bias=True)
        
        self.reset_parameters()
        self.infer = self.forward

    def reset_parameters(self):
        nn.init.uniform_(self.scale_layer.weight, -1, 1)
        nn.init.uniform_(self.bias_layer.weight, -1, 1)
        nn.init.zeros_(self.scale_layer.bias)
        nn.init.zeros_(self.bias_layer.bias)

    def forward(self, x, s):
        # input: (B, C, T)
        # s: speaker embeded, (B, C)
        x = x.transpose(1, -1)
        y = F.layer_norm(x, (self.input_size,), eps=self.eps)

        scale = self.scale_layer(s)
        bias = self.bias_layer(s)
        y = y * scale.unsqueeze(1) + bias.unsqueeze(1)  # (B, T, C)

        return y.transpose(1, -1) # (B, C, T)

    def extra_repr(self):
        return '{input_size}, embed_size={embed_size}, eps={eps}'.format(**self.__dict__)


class WN(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super(WN, self).__init__()
        assert(kernel_size % 2 == 1)
        self.hidden_channels =hidden_channels
        self.kernel_size = kernel_size,
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList()
        self.res_skip_layers = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)

        if gin_channels != 0:
            cond_layer = nn.Linear(gin_channels, 2*hidden_channels*n_layers)
            self.cond_layer = weight_norm(cond_layer)

        for i in range(n_layers):
            dilation = dilation_rate ** i
            padding = int((kernel_size * dilation - dilation) / 2)
            in_layer = nn.Conv1d(hidden_channels, 2*hidden_channels, kernel_size,
                                dilation=dilation, padding=padding)
            in_layer = weight_norm(in_layer)
            self.in_layers.append(in_layer)

            # last one is not necessary
            if i < n_layers - 1:
                res_skip_channels = 2 * hidden_channels
            else:
                res_skip_channels = hidden_channels

            res_skip_layer = nn.Conv1d(hidden_channels, res_skip_channels, 1)
            res_skip_layer = weight_norm(res_skip_layer)
            self.res_skip_layers.append(res_skip_layer)

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)

        if self.gin_channels != 0:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if self.gin_channels != 0:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset+2*self.hidden_channels].unsqueeze(-1)
                acts = x_in + g_l
            else:
                acts = x_in

            acts_a, acts_b = torch.split(acts, self.hidden_channels, 1)
            acts = torch.tanh(acts_a) * torch.sigmoid(acts_b)
            acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:,:self.hidden_channels,:]
                x = (x + res_acts) * x_mask
                output = output + res_skip_acts[:,self.hidden_channels:,:]
            else:
                output = output + res_skip_acts
        return output * x_mask
    
    def infer(self, x, g=None, **kwargs):
        output = 0

        if self.gin_channels != 0:
            g = self.cond_layer(g)
        for i in range(self.n_layers):
            x_in = self.in_layers[i](x)
            if self.gin_channels != 0:
                cond_offset = i * 2 * self.hidden_channels
                g_l = g[:, cond_offset:cond_offset+2*self.hidden_channels].unsqueeze(-1)
                acts = x_in + g_l
            else:
                acts = x_in
            
            acts_a, acts_b = torch.split(acts, self.hidden_channels, 1)
            acts = torch.tanh(acts_a) * torch.sigmoid(acts_b)
            #acts = self.drop(acts)

            res_skip_acts = self.res_skip_layers[i](acts)
            if i < self.n_layers - 1:
                res_acts = res_skip_acts[:, :self.hidden_channels, :]
                x = (x + res_acts)
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output


class ConvNeXtBlock1(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int = 1536,
        kernel_size: int = 7,
        spk_dim: int = 0,
    ):
        super().__init__()
        assert spk_dim > 0 and intermediate_dim % 2 == 0 and kernel_size % 2 == 1
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Linear(intermediate_dim//2, dim)
        self.cond = nn.Linear(spk_dim, intermediate_dim)

    def forward(self, x, g):
        residual = x # (B, C, T)
        x = self.dwconv(x)
        x = self.norm(x.transpose(1, 2))
        x = self.pwconv1(x)
        g = self.cond(g)
        x = torch.tanh(x + g.unsqueeze(1))
        x = self.pwconv2(x)
        x = residual + x.transpose(1, 2)
        return x # (B, C, T)


class ConvNeXtBlock2(nn.Module):
    def __init__(
        self,
        dim: int,
        intermediate_dim: int = 2048,
        kernel_size: int = 7,
        spk_dim: int = 0,
    ):
        super().__init__()
        assert spk_dim > 0 and intermediate_dim % 2 == 0 and kernel_size % 2 == 1
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size, padding=kernel_size//2, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.pwconv2 = nn.Linear(intermediate_dim//2, dim)
        self.cond = nn.Linear(spk_dim, intermediate_dim)

    def forward(self, x, g):
        residual = x # (B, C, T)
        x = self.dwconv(x)
        x = self.norm(x.transpose(1, 2))
        x = self.pwconv1(x)
        g = self.cond(g)
        xa, xb = torch.chunk(x, 2, dim=-1)
        sa, sb = torch.chunk(g, 2, dim=-1)
        x = torch.tanh(xa + sa.unsqueeze(1)) * torch.sigmoid(xb + sb.unsqueeze(1))
        x = self.pwconv2(x)
        x = residual + x.transpose(1, 2)
        return x # (B, C, T)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), gin_channels=0):
        super().__init__()
        inter_channels = (channels // 16) * 16
        self.convs1 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels, inter_channels*2, kernel_size, 1, dilation=d, 
                    padding=get_padding(kernel_size, d)
                )
            ) for d in dilation
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    inter_channels, channels, kernel_size, 1, dilation=1,
                    padding=get_padding(kernel_size, 1)
                )
            ) for _ in dilation
        ])
        self.convs2.apply(init_weights)

        self.conds = nn.ModuleList([
            weight_norm(nn.Linear(gin_channels, inter_channels*2)) for _ in dilation
        ])
    
        self.infer = self.forward

    def forward(self, x, g=None):
        for c1, c2, cs in zip(self.convs1, self.convs2, self.conds):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            gs = cs(g)
            xt = torch.tanh(xt + gs.unsqueeze(-1))
            xt = c2(xt)
            x = xt + x
        return x


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5), gin_channels=0):
        super().__init__()
        inter_channels = (channels // 16) * 16
        self.convs1 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    channels, inter_channels, kernel_size, 1, dilation=d, 
                    padding=get_padding(kernel_size, d)
                )
            ) for d in dilation
        ])
        self.convs1.apply(init_weights)

        self.convs2 = nn.ModuleList([
            weight_norm(
                nn.Conv1d(
                    inter_channels//2, channels, kernel_size, 1, dilation=1,
                    padding=get_padding(kernel_size, 1)
                )
            ) for _ in dilation
        ])
        self.convs2.apply(init_weights)

        self.conds = nn.ModuleList([
            weight_norm(nn.Linear(gin_channels, inter_channels)) for _ in dilation
        ])
        
        self.infer = self.forward

    def forward(self, x, g=None):
        for c1, c2, cs in zip(self.convs1, self.convs2, self.conds):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            gs = cs(g)
            xa, xb = torch.chunk(xt, 2, dim=1)
            sa, sb = torch.chunk(gs, 2, dim=1)
            xt = torch.tanh(xa + sa.unsqueeze(-1)) * torch.sigmoid(xb + sb.unsqueeze(-1))
            xt = c2(xt)
            x = xt + x
        return x


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x
    
    def infer(self, x, reverse=True, **kwargs):
        x = torch.exp(x)
        return x
        

class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x
    
    def infer(self, x, *args, reverse=True, **kwargs):
        x = torch.flip(x, [1])
        return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels,1))
        self.logs = nn.Parameter(torch.zeros(channels,1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1,2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x
    
    def infer(self, x, reverse=True, **kwargs):
        x = (x - self.m) * torch.exp(-self.logs)
        return x


class ResidualCouplingLayer(nn.Module):
    def __init__(self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels]*2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels]*2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1,2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x
    
    def infer(self, x, g, reverse=True):
        x0, x1 = torch.split(x, self.half_channels, 1)
        h = self.pre(x0)
        h = self.enc.infer(h, g=g)
        stats = self.post(h)
        if not self.mean_only:
            m, logs = torch.split(stats, self.half_channels, 1)
            x1 = (x1 - m) * torch.exp(-logs)
        else:
            m = stats
            x1 = x1 - m

        x = torch.cat([x0, x1], 1)
        return x


class TorchSTFT(nn.Module):
    def __init__(self, fft_size, hop_size, win_size=None):
        super().__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size if win_size is not None else fft_size
        self.register_buffer("window", torch.hann_window(self.win_size), persistent=False)
    
    def stft(self, x):
        # x: (B, t)
        spec = torch.stft(x,
            n_fft=self.fft_size, hop_length=self.hop_size, 
            win_length=self.win_size, window=self.window,
            center=True, pad_mode='reflect', return_complex=False) # (B, F, T, 2), F=n_fft//2+1, T=t//hop_size+1
        return spec[..., 0], spec[..., 1]
    
    def istft(self, real, imag):
        # real/imag: (B, F, T), n_fft//2+1
        x = torch.istft(torch.complex(real, imag),
            n_fft=self.fft_size, hop_length=self.hop_size, 
            win_length=self.win_size, window=self.window,
            center=True, return_complex=False)
        return x # (B, t), t=(T-1)*hop_size
