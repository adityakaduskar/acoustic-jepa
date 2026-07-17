"""Acoustic-JEPA model (ACOUSTIC_JEPA_SPEC.md §3).

Action-conditioned Joint-Embedding Predictive Architecture over binaural
cochleagrams:

    Encoder E_theta : coch[2,64,48] -> z in R^emb   (shared per-ear tower + concat)
    Target  E_xi    : EMA copy of E_theta, stop-grad, m=0.996
    Predictor P_phi : (z_t, a_t) -> z_{t+1}          (residual, action-conditioned)

Trained to predict the *latent* of future observations (no waveform / pixel
reconstruction), rolled out H steps in latent space (V-JEPA-2 style). A VICReg
variance+covariance term on the online latent prevents collapse -- the primary
failure mode of non-contrastive audio SSL, so it is instrumented explicitly.

GroupNorm (not BatchNorm) is used throughout so the online and EMA encoders do
not share batch statistics, which keeps the EMA update to parameters only.
"""
from __future__ import annotations
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class EarCNN(nn.Module):
    """Per-ear conv tower: [B,1,64,48] -> [B, feat]."""

    def __init__(self, feat: int = 128):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(
                nn.Conv2d(ci, co, 3, stride=2, padding=1), _gn(co), nn.GELU()
            )
        self.net = nn.Sequential(
            block(1, 32),     # 64x48 -> 32x24
            block(32, 64),    # -> 16x12
            block(64, 128),   #  -> 8x6
            block(128, feat), #  -> 4x3
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):                     # x: [B,1,64,48]
        h = self.net(x)
        return self.pool(h).flatten(1)        # [B, feat]


class _gn1d(nn.Module):
    """GroupNorm over a [B, C] vector (treat C as channels)."""

    def __init__(self, c):
        super().__init__()
        self.gn = nn.GroupNorm(min(8, c), c)

    def forward(self, x):
        return self.gn(x.unsqueeze(-1)).squeeze(-1)


class Encoder(nn.Module):
    """Binaural encoder: shared per-ear tower, concat ears, project to emb."""

    def __init__(self, emb: int = 256):
        super().__init__()
        feat = emb // 2
        self.ear = EarCNN(feat)
        self.proj = nn.Sequential(nn.Linear(2 * feat, emb), _gn1d(emb), nn.GELU(),
                                  nn.Linear(emb, emb))

    def forward(self, coch):                  # coch: [B,2,64,48]
        l = self.ear(coch[:, 0:1])            # weights SHARED across ears
        r = self.ear(coch[:, 1:2])
        return self.proj(torch.cat([l, r], dim=1))   # [B, emb]


class SeqEncoder(nn.Module):
    """Temporal encoder: run the per-frame binaural CNN over a window of W
    consecutive observations and integrate them with a GRU. This is the
    biologically-motivated fix -- real echolocation integrates a *sequence* of
    echoes across self-motion/head movement rather than reading one static echo
    (Thaler et al. on head movements; Simmons on temporal processing in bats).
    W=1 reduces exactly to the memoryless single-frame encoder.
    Accepts [B,2,64,48] (single frame) or [B,W,2,64,48] (window)."""

    def __init__(self, emb: int = 256, window: int = 1):
        super().__init__()
        self.frame = Encoder(emb)
        self.window = window
        self.gru = nn.GRU(emb, emb, batch_first=True) if window > 1 else None

    def forward(self, x):
        if x.dim() == 4:                                  # [B,2,64,48] single frame
            return self.frame(x)
        B, W = x.shape[0], x.shape[1]                     # [B,W,2,64,48]
        f = self.frame(x.reshape(B * W, *x.shape[2:])).reshape(B, W, -1)
        if self.gru is None:
            return f[:, -1]
        out, _ = self.gru(f)
        return out[:, -1]                                 # integrated latent (last hidden)


class Predictor(nn.Module):
    """Residual action-conditioned latent predictor: (z_t, a_t) -> z_{t+1}."""

    def __init__(self, emb: int = 256, action_dim: int = 2, hidden: int = 512):
        super().__init__()
        self.act_embed = nn.Sequential(nn.Linear(action_dim, emb), nn.GELU())
        self.mlp = nn.Sequential(
            nn.Linear(2 * emb, hidden), _gn1d(hidden), nn.GELU(),
            nn.Linear(hidden, emb),
        )

    def forward(self, z, a):                  # z:[B,emb], a:[B,action_dim]
        ae = self.act_embed(a)
        return z + self.mlp(torch.cat([z, ae], dim=1))   # residual step in latent space


def vicreg_var_cov(z, gamma: float = 1.0, eps: float = 1e-4):
    """VICReg anti-collapse: hinge on per-dim std + off-diagonal covariance."""
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(gamma - std).mean()
    zc = z - z.mean(dim=0)
    cov = (zc.T @ zc) / (z.shape[0] - 1)
    d = z.shape[1]
    cov_loss = (cov.pow(2).sum() - cov.diag().pow(2).sum()) / d
    return var_loss, cov_loss


@torch.no_grad()
def latent_stats(z) -> dict:
    """Collapse watchdog: mean per-dim std and participation-ratio 'rank'."""
    std = torch.sqrt(z.var(dim=0) + 1e-8)
    zc = z - z.mean(dim=0)
    s = torch.linalg.svdvals(zc)
    pr = (s.sum() ** 2) / (s.pow(2).sum() + 1e-12)    # effective rank
    return {"std": float(std.mean()), "rank": float(pr)}


class AcousticJEPA(nn.Module):
    def __init__(self, emb: int = 256, action_dim: int = 2, ema: float = 0.996,
                 var_coeff: float = 1.0, cov_coeff: float = 0.04, gamma: float = 1.0,
                 window: int = 1):
        super().__init__()
        self.window = window                              # W frames integrated per observation
        self.encoder = SeqEncoder(emb, window)
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(emb, action_dim)
        self.ema = ema
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff
        self.gamma = gamma

    @torch.no_grad()
    def update_target(self):
        m = self.ema
        for p, tp in zip(self.encoder.parameters(), self.target.parameters()):
            tp.mul_(m).add_(p, alpha=1.0 - m)
        for b, tb in zip(self.encoder.buffers(), self.target.buffers()):
            tb.copy_(b)

    def forward(self, coch_seq, actions):
        """coch_seq: [B, W+H, 2,64,48], actions: [B, W+H-1, 2] -> loss dict.
        Observation k is the sliding window coch_seq[:, k:k+W]; there are H+1 of
        them (k=0..H). Context = window 0; targets = windows 1..H. Action W-1+k
        moves observation k -> k+1. (W=1 reduces to the original per-frame JEPA.)"""
        W = self.window
        T = coch_seq.shape[1]
        H = T - W

        z0 = self.encoder(coch_seq[:, 0:W])                   # online context latent
        with torch.no_grad():
            self.target.eval()
            tgt = torch.stack([self.target(coch_seq[:, k:k + W]) for k in range(1, H + 1)], dim=1)  # [B,H,emb]

        preds = []
        z = z0
        for k in range(H):
            z = self.predictor(z, actions[:, W - 1 + k])
            preds.append(z)
        preds = torch.stack(preds, dim=1)                     # [B,H,emb]

        # discounted multi-step prediction loss
        per_step = (preds - tgt).pow(2).mean(dim=(0, 2))      # [H]
        w = (self.gamma ** torch.arange(H, device=per_step.device)).clamp_min(1e-6)
        pred_loss = (w * per_step).sum() / w.sum()

        var_loss, cov_loss = vicreg_var_cov(z0)
        total = pred_loss + self.var_coeff * var_loss + self.cov_coeff * cov_loss
        return {
            "loss": total, "pred": pred_loss.detach(),
            "var": var_loss.detach(), "cov": cov_loss.detach(),
            "z": z0.detach(),
        }

    @torch.no_grad()
    def embed(self, coch):
        """Frozen-encoder embedding for probes/planning: coch[...,2,64,48]->[...,emb]."""
        return self.encoder(coch)


if __name__ == "__main__":
    # smoke test: random batch through the full objective + collapse watchdog
    torch.manual_seed(0)
    B, H, W = 8, 8, 4
    model = AcousticJEPA(emb=256, window=W)
    coch = torch.randn(B, W + H, 2, 64, 48)              # W+H frames
    act = torch.rand(B, W + H - 1, 2) * 2 - 1
    out = model(coch, act)
    out["loss"].backward()
    model.update_target()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"params(trainable): {n_params/1e6:.2f}M  window={W}")
    print("loss %.4f  pred %.4f  var %.4f  cov %.4f"
          % (out["loss"].item(), out["pred"], out["var"], out["cov"]))
    print("latent", latent_stats(out["z"]))
    print("embed window", tuple(model.embed(coch[:, 0:W]).shape),
          "| embed frame", tuple(model.embed(coch[:, 0]).shape))
