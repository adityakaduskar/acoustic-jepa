"""Perception trunk + Q/policy heads for EchoRL (M3/M4).

Reuses the validated cross-view encoder (jepa/model_xview.TwoStreamEncoder):
range[256] via a 1D CNN (HRTF-free) + binaural coch[2,64,48] via a per-ear 2D CNN
-> block latent [z_range | z_coch]. We drop the JEPA predictor/heads and attach RL
heads. Inputs are normalized with the offline corpus stats (norm_stats.npz); coch
uses (x-mu)/(sigma+eps) because ~40% of coch channels are near-silent (M2 finding).
"""
from __future__ import annotations
import os, sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "jepa"))
from model_xview import TwoStreamEncoder  # noqa: E402

NORM_STATS = os.path.join(_HERE, "norm_stats.npz")


class Normalizer(nn.Module):
    def __init__(self, stats_path=NORM_STATS, eps=1e-3):
        super().__init__()
        d = np.load(stats_path)
        self.register_buffer("r_mean", torch.tensor(d["range_mean"], dtype=torch.float32))
        self.register_buffer("r_std", torch.tensor(d["range_std"], dtype=torch.float32))
        self.register_buffer("c_mean", torch.tensor(d["coch_mean"], dtype=torch.float32))
        self.register_buffer("c_std", torch.tensor(d["coch_std"], dtype=torch.float32))
        self.eps = eps

    def forward(self, rng, coch):
        rn = (rng - self.r_mean) / (self.r_std + self.eps)
        cn = (coch - self.c_mean) / (self.c_std + self.eps)
        return rn, cn


class EchoTrunk(nn.Module):
    """obs dict -> latent [B, emb_range+emb_coch+3] (prop appended)."""
    def __init__(self, emb_range=128, emb_coch=128, stats_path=NORM_STATS):
        super().__init__()
        self.norm = Normalizer(stats_path)
        self.enc = TwoStreamEncoder(emb_range, emb_coch)
        self.out_dim = emb_range + emb_coch + 3

    def forward(self, rng, coch, prop):
        rn, cn = self.norm(rng, coch)
        zr, zc = self.enc(cn, rn)                 # TwoStreamEncoder(coch, rng)
        return torch.cat([zr, zc, prop], dim=1)


class DuelingQNet(nn.Module):
    def __init__(self, n_actions=5, emb_range=128, emb_coch=128, hidden=256,
                 stats_path=NORM_STATS):
        super().__init__()
        self.trunk = EchoTrunk(emb_range, emb_coch, stats_path)
        d = self.trunk.out_dim
        self.val = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.adv = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, n_actions))

    def forward(self, rng, coch, prop):
        z = self.trunk(rng, coch, prop)
        v = self.val(z)
        a = self.adv(z)
        return v + (a - a.mean(dim=1, keepdim=True))
