import copy
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

from modules import LayerNorm


class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0., ffn="FFN2", gin_channels=0, **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for i in range(self.n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(globals()[ffn](hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout, gin_channels=gin_channels))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask, g):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask, g=g)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x
    
    def infer(self, x, g):
        for i in range(self.n_layers):
            y = self.attn_layers[i](x, x, attn_mask=None)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i].infer(x, g=g)
            x = self.norm_layers_2[i](x + y)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout

        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
            
    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x = self.attention(q, k, v, mask=attn_mask)[0]
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1) # [b, n_h, t_t, t_s]
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t) # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn


class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0, gin_channels=0):
        super().__init__()
        assert kernel_size % 2 == 1, f"{kernel_size}"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask, g=None):
        x = F.relu(self.conv_1(x))
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask
    
    def infer(self, x, g=None):
        x = F.relu(self.conv_1(x))
        x = self.conv_2(x)
        return x
    

class FFN2(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0, gin_channels=0):
        super().__init__()
        assert kernel_size % 2 == 1, f"{kernel_size}"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = nn.Conv1d(in_channels, filter_channels*2, kernel_size, padding=kernel_size//2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.drop = nn.Dropout(p_dropout)
        
        self.cond = nn.Linear(gin_channels, filter_channels*2)
        
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.cond.weight)

    def forward(self, x, x_mask, g):
        x = self.conv_1(x)
        x = self.drop(x)
        g = self.cond(g)
        xa, xb = torch.chunk(x, 2, dim=1)
        sa, sb = torch.chunk(g, 2, dim=1)
        x = torch.tanh(xa + sa.unsqueeze(-1)) * torch.sigmoid(xb + sb.unsqueeze(-1))
        x = self.conv_2(x * x_mask)
        return x * x_mask
    
    def infer(self, x, g):
        x = self.conv_1(x)
        g = self.cond(g)
        xa, xb = torch.chunk(x, 2, dim=1)
        sa, sb = torch.chunk(g, 2, dim=1)
        x = torch.tanh(xa + sa.unsqueeze(-1)) * torch.sigmoid(xb + sb.unsqueeze(-1))
        x = self.conv_2(x)
        return x

class FFN3(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0, gin_channels=0):
        super().__init__()
        assert kernel_size % 2 == 1, f"{kernel_size}"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.drop = nn.Dropout(p_dropout)
        
        self.cond = nn.Linear(gin_channels, filter_channels)
        
        nn.init.xavier_uniform_(self.conv_1.weight)
        nn.init.xavier_uniform_(self.conv_2.weight)
        nn.init.xavier_uniform_(self.cond.weight)

    def forward(self, x, x_mask, g):
        x = F.relu(self.conv_1(x))
        x = self.drop(x)
        g = self.cond(g)
        x = self.conv_2((x + g) * x_mask)
        return x * x_mask
    
    def infer(self, x, g):
        x = F.relu(self.conv_1(x))
        g = self.cond(g)
        x = self.conv_2(x + g)
        return x