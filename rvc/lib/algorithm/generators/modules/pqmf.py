# -*- coding: utf-8 -*-

# Copyright 2020 Tomoki Hayashi
#  MIT License (https://opensource.org/licenses/MIT)

"""Pseudo QMF modules."""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal.windows import kaiser


def design_prototype_filter(taps=62, cutoff_ratio=0.15, beta=9.0):
    """Design prototype filter for PQMF.
    This method is based on `A Kaiser window approach for the design of prototype
    filters of cosine modulated filterbanks`_.
    Args:
        taps (int): The number of filter taps.
        cutoff_ratio (float): Cut-off frequency ratio.
        beta (float): Beta coefficient for kaiser window.
    Returns:
        ndarray: Impluse response of prototype filter (taps + 1,).
    .. _`A Kaiser window approach for the design of prototype filters of cosine modulated filterbanks`:
        https://ieeexplore.ieee.org/abstract/document/681427
    """
    # check the arguments are valid
    assert taps % 2 == 0, "The number of taps mush be even number."
    assert 0.0 < cutoff_ratio < 1.0, "Cutoff ratio must be > 0.0 and < 1.0."

    # make initial filter
    omega_c = np.pi * cutoff_ratio
    with np.errstate(invalid="ignore"):
        h_i = np.sin(omega_c * (np.arange(taps + 1) - 0.5 * taps)) / (
            np.pi * (np.arange(taps + 1) - 0.5 * taps)
        )
    h_i[taps // 2] = np.cos(0) * cutoff_ratio  # fix nan due to indeterminate form

    # apply kaiser window
    w = kaiser(taps + 1, beta)
    h = h_i * w

    return h


class PQMF(torch.nn.Module):
    """PQMF module.
    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.
    .. _`Near-perfect-reconstruction pseudo-QMF banks`:
        https://ieeexplore.ieee.org/document/258122
    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.15, beta=9.0):
        """Initilize PQMF module.
        Args:
            subbands (int): The number of subbands.
            taps (int): The number of filter taps.
            cutoff_ratio (float): Cut-off frequency ratio.
            beta (float): Beta coefficient for kaiser window.
        """
        super(PQMF, self).__init__()

        # define filter coefficient
        h_proto = design_prototype_filter(taps, cutoff_ratio, beta)
        h_analysis = np.zeros((subbands, len(h_proto)))
        h_synthesis = np.zeros((subbands, len(h_proto)))
        for k in range(subbands):
            h_analysis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - ((taps - 1) / 2))
                    + (-1) ** k * np.pi / 4
                )
            )
            h_synthesis[k] = (
                2
                * h_proto
                * np.cos(
                    (2 * k + 1)
                    * (np.pi / (2 * subbands))
                    * (np.arange(taps + 1) - ((taps - 1) / 2))
                    - (-1) ** k * np.pi / 4
                )
            )

        # convert to tensor
        analysis_filter = torch.from_numpy(h_analysis).float().unsqueeze(1)
        synthesis_filter = torch.from_numpy(h_synthesis).float().unsqueeze(0)

        # register coefficients as beffer
        self.register_buffer("analysis_filter", analysis_filter)
        self.register_buffer("synthesis_filter", synthesis_filter)

        # filter for downsampling & upsampling
        updown_filter = torch.zeros((subbands, subbands, subbands)).float()
        for k in range(subbands):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.subbands = subbands

        # keep padding info
        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def analysis(self, x):
        """Analysis with PQMF.
        Args:
            x (Tensor): Input tensor (B, 1, T).
        Returns:
            Tensor: Output tensor (B, subbands, T // subbands).
        """
        x = F.conv1d(self.pad_fn(x), self.analysis_filter)
        return F.conv1d(x, self.updown_filter, stride=self.subbands)

    def synthesis(self, x):
        """Synthesis with PQMF.
        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).
        Returns:
            Tensor: Output tensor (B, 1, T).
        """
        # NOTE(kan-bayashi): Power will be dreased so here multipy by # subbands.
        #   Not sure this is the correct way, it is better to check again.
        # TODO(kan-bayashi): Understand the reconstruction procedure
        x = F.conv_transpose1d(
            x, self.updown_filter * self.subbands, stride=self.subbands
        )
        return F.conv1d(self.pad_fn(x), self.synthesis_filter)


class LearnablePQMF(PQMF):
    """Learnable PQMF module.
    This module is based on `Near-perfect-reconstruction pseudo-QMF banks`_.

    Normal PQMF has static filter for `analysis` and `synthesis`
    but this module use learnable filter initialized by originals for `synthesis`.
    """

    def __init__(self, subbands=4, taps=62, cutoff_ratio=0.15, beta=9.0):
        super().__init__(subbands, taps, cutoff_ratio, beta)
        self.learnable_updown_filter = torch.nn.Parameter(
            self.updown_filter.detach().clone(), requires_grad=True
        )
        self.learnable_synthesis_filter = torch.nn.Parameter(
            self.synthesis_filter.detach().clone(), requires_grad=True
        )

    def synthesis(self, x):
        """Synthesis with PQMF.
        Args:
            x (Tensor): Input tensor (B, subbands, T // subbands).
        Returns:
            Tensor: Output tensor (B, 1, T).
        """
        x = F.conv_transpose1d(
            x, self.learnable_updown_filter * self.subbands, stride=self.subbands
        )
        return F.conv1d(self.pad_fn(x), self.learnable_synthesis_filter)


class ResidualPQMF(torch.nn.Module):
    def __init__(self, n=2, subbands=4, taps=62, cutoff_ratio=0.15, beta=9.0):
        super().__init__()
        self.subbands = subbands
        self.pqmfs = torch.nn.ModuleList()
        for _ in range(n):
            self.pqmfs += [
                PQMF(subbands=subbands, taps=taps, cutoff_ratio=cutoff_ratio, beta=beta)
            ]

    def analysis(self, x):
        B, _, T = x.shape
        for pqmf in self.pqmfs:
            x = pqmf.analysis(x)
            x = x.reshape(-1, 1, x.shape[2])
        x = x.reshape(B, -1, T // (pqmf.subbands ** len(self.pqmfs)))
        return x

    def synthesis(self, x):
        for pqmf in reversed(self.pqmfs):
            x = x.reshape(-1, self.subbands, x.shape[2])
            x = pqmf.synthesis(x)
        x = x.squeeze(1)
        return x
