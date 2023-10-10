# -*- coding: utf-8 -*-

# Copyright 2019 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""STFT-based Loss modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import TorchSTFT


class STFTLoss(TorchSTFT):
    """STFT loss module."""

    def __init__(self, fft_size, hop_size, win_size):
        """Initialize STFT loss module."""
        super().__init__(fft_size, hop_size, win_size)
    
    def spec2mag(self, real, imag):
        return torch.sqrt(real**2 + imag**2) + 1e-5

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Predicted signal (B, t).
            y (Tensor): Groundtruth signal (B, t).

        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
            Tensor: Magnitude of `x` (B, F, T).
            Tensor: Magnitude of `y` (B, F, T).

        """
        x_mag = self.spec2mag(*self.stft(x))
        y_mag = self.spec2mag(*self.stft(y))
        sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
        mag_loss = F.l1_loss(torch.log(x_mag), torch.log(y_mag))

        return sc_loss, mag_loss, x_mag, y_mag


class MultiResolutionSTFTLoss(nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(
        self,
        fft_sizes=[128, 256, 512, 1024, 2048],
        hop_sizes=[32, 64, 128, 256, 512],
        win_sizes=[128, 256, 512, 1024, 2048],
    ):
        """Initialize Multi resolution STFT loss module.

        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_sizes (list): List of window lengths.

        """
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_sizes)
        self.stft_losses = nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_sizes):
            self.stft_losses += [STFTLoss(fs, ss, wl)]

    def forward(self, x, y):
        """Calculate forward propagation.

        Args:
            x (Tensor): Groundtruth signal (B, t).
            y (Tensor): Predicted signal (B, t).

        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
            Tensor: List of magnitude of `x` [(B, F, T), ...]
            Tensor: List of magnitude of `y` [(B, F, T), ...]

        """
        sc_loss, mag_loss = 0.0, 0.0
        xs_mag, ys_mag = [], []
        for f in self.stft_losses:
            sc_l, mag_l, x_mag, y_mag = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
            xs_mag.append(x_mag)
            ys_mag.append(y_mag)
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return sc_loss, mag_loss, xs_mag, ys_mag
