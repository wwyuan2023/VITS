import warnings
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

import commons
import modules
import attentions
try:
    import monotonic_align
except:
    warnings.warn("You must build `monotonic_align` if training.", UserWarning)

from commons import init_weights, get_padding, gen_sin_table


class DurationPredictor(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size=5, p_dropout=0.5, gin_channels=0):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels
        
        self.cond = nn.Linear(gin_channels, 256)

        self.drop = nn.Dropout(p_dropout)
        self.pre = nn.Conv1d(in_channels, filter_channels, 1)
        self.norm_1 = modules.LayerNorm(filter_channels)
        self.resblock = modules.ResBlock2(filter_channels, kernel_size, (1,3,5), gin_channels=256)
        self.norm_2 = modules.LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)

    def forward(self, x, x_mask, g):
        x, g = torch.detach(x), torch.detach(g)
        g = self.cond(g)
        x = self.drop(self.norm_1(self.pre(x)))
        x = self.resblock(x, x_mask, g=g)
        x = self.drop(self.norm_2(x))
        x = self.proj(x * x_mask)
        return x * x_mask

    def infer(self, x, g):
        g = self.cond(g)
        x = self.norm_1(self.pre(x))
        x = self.resblock.infer(x, g=g)
        x = self.norm_2(x)
        x = self.proj(x)
        return x


class TextEncoder(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.emb = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
        )
        self.emo_proj = nn.Linear(1024, hidden_channels)
        
        self.spk_proj = nn.Linear(gin_channels, hidden_channels)
        self.spk_adaptor = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, 1, groups=1),
            nn.ReLU(), modules.LayerNorm(hidden_channels), nn.Dropout(p_dropout),
            nn.Conv1d(hidden_channels, hidden_channels, 7, groups=4),
            nn.ReLU(), modules.LayerNorm(hidden_channels), nn.Dropout(p_dropout),
            nn.Conv1d(hidden_channels, hidden_channels, 1, groups=1),
            nn.ReLU(), modules.LayerNorm(hidden_channels), nn.Dropout(p_dropout),
        )

        # positional encoding
        self.register_buffer(
            "sin_table",
            gen_sin_table(256+128, hidden_channels),
            persistent=False,
        )
        self.xscale = math.sqrt(hidden_channels)
        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.encoder = attentions.Encoder(
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            p_dropout,
        )
        self.proj= nn.Conv1d(hidden_channels, out_channels * 2, 1)
    
    def positional_encoding(self, x, alpha):
        # x: (B,T,d)
        T, max_len = x.size(1), self.sin_table.size(1)
        assert not self.training or T < max_len, f"The input T={T} > max_len{max_len}, pls resolve it!"
        if T <= max_len:
            pe = self.sin_table[:, :T]
        else:
            pe = gen_sin_table(x.size(1), x.size(2)).to(x.device)
        x = x * self.xscale + pe * alpha
        return x

    def forward(self, x, x_lengths, emo, g):
        x = self.emb(x) # [b, t, h]
        x = x + self.emo_proj(emo).unsqueeze(1)
        x = self.positional_encoding(x, self.alpha)
        x = torch.transpose(x, 1, -1) # [b, h, t]
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)

        x = self.encoder(x * x_mask, x_mask)
        h = x + self.spk_proj(g).unsqueeze(2) # [b, h, t]
        h = self.spk_adaptor(h * x_mask) * x_mask
        stats = self.proj(x + h) * x_mask

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs, x_mask
    
    def infer(self, x, emo, g):
        x = self.emb(x)  # [b, t, h]
        x = x + self.emo_proj(emo).unsqueeze(1)
        x = self.positional_encoding(x, self.alpha)
        x = torch.transpose(x, 1, -1) # [b, h, t]
        x = self.encoder.infer(x)
        h = x + self.spk_proj(g).unsqueeze(2) # [b, h, t]
        h = self.spk_adaptor(h)
        stats = self.proj(x + h)

        m, logs = torch.split(stats, self.out_channels, dim=1)
        return x, m, logs


class ResidualCouplingBlock(nn.Module):
    def __init__(self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0
    ):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for _ in range(n_flows):
            self.flows.append(modules.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(modules.Flip())
        
        self.flows_reversed = list(self.flows)[::-1]
        
    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x
    
    def infer(self, x, g, reverse=True):
        if not reverse:
            for flow in self.flows:
                x = flow.infer(x, g=g, reverse=reverse)
        else:
            for flow in self.flows_reversed:
                x = flow.infer(x, g=g, reverse=reverse)
        return x


class PosteriorEncoder(nn.Module):
    def __init__(self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            modules.LayerNorm(hidden_channels),
        )
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask
    
    def infer(self, x, n, g=None):
        x = self.pre(x)
        x = self.enc.infer(x, g=g)
        stats = self.proj(x)
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = m + n * torch.exp(logs)
        return z


class Generator(nn.Module):
        def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
            super(Generator, self).__init__()
            self.num_kernels = len(resblock_kernel_sizes)
            self.num_upsamples = len(upsample_rates)
            self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
            resblock = modules.ResBlock1 if resblock == '1' else modules.ResBlock2

            self.ups = nn.ModuleList()
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
                self.ups.append(weight_norm(
                    ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                    k, u, padding=(k-u)//2)))

            self.resblocks = nn.ModuleList()
            for i in range(len(self.ups)):
                ch = upsample_initial_channel//(2**(i+1))
                for _, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                    self.resblocks.append(resblock(ch, k, d, gin_channels))

            self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
            self.ups.apply(init_weights)

        def forward(self, x, g):
            x = self.conv_pre(x)
            for i in range(self.num_upsamples):
                x = F.leaky_relu(x, modules.LRELU_SLOPE)
                x = self.ups[i](x)
                xs = 0
                for j in range(self.num_kernels):
                    xs += self.resblocks[i*self.num_kernels+j](x, g=g)
                x = xs / self.num_kernels
            x = F.leaky_relu(x)
            x = self.conv_post(x)
            x = torch.tanh(x)
            return x
        
        def infer(self, x, g):
            x = self.conv_pre(x)
            for i in range(self.num_upsamples):
                x = F.leaky_relu(x, modules.LRELU_SLOPE)
                x = self.ups[i](x)
                xs = 0
                for j in range(self.num_kernels):
                    xs += self.resblocks[i*self.num_kernels+j].infer(x, g=g)
                x = xs / self.num_kernels
            x = F.leaky_relu(x)
            x = self.conv_post(x)
            x = torch.tanh(x)
            return x


class DiscriminatorP(nn.Module):
        def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
            super(DiscriminatorP, self).__init__()
            self.period = period
            self.use_spectral_norm = use_spectral_norm
            norm_f = weight_norm if use_spectral_norm == False else spectral_norm
            self.convs = nn.ModuleList([
                norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
            ])
            self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

        def forward(self, x):
            fmap = []

            # 1d to 2d
            b, c, t = x.shape
            if t % self.period != 0: # pad first
                    n_pad = self.period - (t % self.period)
                    x = F.pad(x, (0, n_pad), "reflect")
                    t = t + n_pad
            x = x.view(b, c, t // self.period, self.period)

            for l in self.convs:
                    x = l(x)
                    x = F.leaky_relu(x, modules.LRELU_SLOPE)
                    fmap.append(x)
            x = self.conv_post(x)
            fmap.append(x)
            x = torch.flatten(x, 1, -1)

            return x, fmap


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
                x = l(x)
                x = F.leaky_relu(x, modules.LRELU_SLOPE)
                fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
                y_d_r, fmap_r = d(y)
                y_d_g, fmap_g = d(y_hat)
                y_d_rs.append(y_d_r)
                y_d_gs.append(y_d_g)
                fmap_rs.append(fmap_r)
                fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(self, 
        text_channels,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock, 
        resblock_kernel_sizes, 
        resblock_dilation_sizes, 
        upsample_rates, 
        upsample_initial_channel, 
        upsample_kernel_sizes,
        n_speakers=0,
        gin_channels=0,
        **kwargs):

        super().__init__()
        self.text_channels = text_channels
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.gin_channels = gin_channels

        self.enc_p = TextEncoder(text_channels, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=0)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, gin_channels=gin_channels)
        self.dp = DurationPredictor(hidden_channels, 256, 5, p_dropout=0.25, gin_channels=gin_channels)

        assert n_speakers > 1
        self.emb_g = nn.Embedding(n_speakers, gin_channels)
    
    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""
        def _remove_weight_norm(m):
            try:
                remove_weight_norm(m)
            except ValueError: # this module didn't have weight norm
                return
        self.apply(_remove_weight_norm)

    def forward(self, x, x_lengths, y, y_lengths, emo, sid=None):
        g = self.emb_g(sid) # [b, h]
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, emo, g=g)

        z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=None)
        z_p = self.flow(z, y_mask, g=g)

        with torch.no_grad():
            # negative cross-entropy
            s_p_sq_r = torch.exp(-2 * logs_p) # [b, d, t]
            neg_cent1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True) # [b, 1, t_s]
            neg_cent2 = torch.matmul(-0.5 * (z_p ** 2).transpose(1, 2), s_p_sq_r) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent3 = torch.matmul(z_p.transpose(1, 2), (m_p * s_p_sq_r)) # [b, t_t, d] x [b, d, t_s] = [b, t_t, t_s]
            neg_cent4 = torch.sum(-0.5 * (m_p ** 2) * s_p_sq_r, [1], keepdim=True) # [b, 1, t_s]
            neg_cent = neg_cent1 + neg_cent2 + neg_cent3 + neg_cent4

            attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
            attn = monotonic_align.maximum_path(neg_cent, attn_mask.squeeze(1)).detach()

        w = attn.sum(1, keepdim=True)
        logw_ = torch.log(w + 1e-6) * x_mask
        logw = self.dp(x, x_mask, g=g)
        l_length = torch.sum(torch.abs(logw - logw_), [1, 2]) / torch.sum(x_mask) # for averaging

        # expand prior
        m_p = torch.matmul(attn, m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn, logs_p.transpose(1, 2)).transpose(1, 2)

        z_slice, ids_slice = commons.rand_slice_segments(z, y_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        # forward generate
        z_q = self.flow(m_p + torch.randn_like(m_p) * torch.exp(logs_p), y_mask, g=g, reverse=True)
        z_slice = commons.slice_segments(z_q, ids_slice, self.segment_size)
        o_q = self.dec(z_slice, g=g)

        return o, l_length, attn, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), z_q, o_q

    def inference(self, x, x_lengths, emo, sid=None, noise_scale=1, length_scale=1, max_len=None):
        g = self.emb_g(sid) # [b, h]
        x, m_p, logs_p, x_mask = self.enc_p(x, x_lengths, emo, g=g)

        logw = self.dp(x, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(commons.sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = commons.generate_path(w_ceil, attn_mask.squeeze(1))

        m_p = torch.matmul(attn, m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn, logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.dec((z * y_mask)[:, :, :max_len], g=g)
        return o, attn, y_mask, (z, z_p, m_p, logs_p)
    
    @torch.no_grad()
    def infer(self, x, emo, sid, noise_scale=0.35, length_scale=1):
        assert x.size(0) == 1
        x_lengths = x.size(1)
        g = self.emb_g(sid) # [b, h]
        x, m_p, logs_p = self.enc_p.infer(x, emo, g=g)

        logw = self.dp.infer(x, g=g)
        w = torch.exp(logw) * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil), 1).item()
        attn = commons.infer_path(w_ceil, x_lengths, int(y_lengths), dtype=m_p.dtype)

        m_p = torch.matmul(attn, m_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p = torch.matmul(attn, logs_p.transpose(1, 2)).transpose(1, 2) # [b, t', t], [b, t, d] -> [b, d, t']

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * noise_scale
        z = self.flow.infer(z_p, g=g, reverse=True)
        o = self.dec(z, g=g)
        return o
    
    @torch.no_grad()
    def infer_p1(self, x, emo, sid):
        assert x.size(0) == 1
        g = self.emb_g(sid) # [b, h]
        x, m_p, logs_p = self.enc_p.infer(x, emo, g=g)
        s_p = torch.exp(logs_p)

        logw = self.dp.infer(x, g=g)
        return m_p, s_p, logw, g
    
    @torch.no_grad()
    def infer_p2(self, attn, m_p, s_p, g, noise):
        m_p = torch.matmul(attn, m_p.transpose(1, 2)).transpose(1, 2)
        s_p = torch.matmul(attn, s_p.transpose(1, 2)).transpose(1, 2)
        z_p = m_p + noise * s_p
        z = self.flow.infer(z_p, g=g, reverse=True)
        o = self.dec(z, g=g)
        return o

