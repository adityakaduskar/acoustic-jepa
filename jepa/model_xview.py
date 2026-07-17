"""Cross-view Acoustic-JEPA (v2).

The v2 pretext task: agent E emits; predict what listener L RECEIVES, from E's
own (monostatic) observation plus the measured relative transform E<->L. Because
egocentric interpolation cannot cross viewpoints, a latent that solves this must
be allocentric -- the v2 hypothesis (see memory acoustic-jepa-spec). This
replaces v1's action-conditioned self-prediction, whose honest negative result
(never beat a raw-cochleagram linear probe) motivated the redesign.

Architecture (decided 2026-07-17). Two towers -> a BLOCK-STRUCTURED latent
z = [z_range | z_coch], fused only shallowly and only on the CONTEXT side:

  * z_range  <- 1D CNN over the matched-filter range profile [256].  HRTF-FREE.
  * z_coch   <- per-ear 2D CNN over the binaural cochleagram [2,64,48]. HRTF-DEP.

Two design facts drive keeping the blocks separate rather than one fused latent:
  (1) the cross-view TARGET should be the HRTF-free range block, so predicting a
      peer's latent isn't forced to model the peer's ears (the range head needs
      no HRTF id; the coch head is conditioned on the listener's HRTF id);
  (2) v1 showed that added capacity which ENTANGLES cues degrades linear
      decodability -- so fusion is shallow (a single projection for context),
      never cross-attention/FiLM.

Loss = L_range + lambda*L_coch (lambda<1, geometry-driven) + per-block VICReg
(each block gets its own variance/covariance term so neither collapses while the
other carries the variance). Transform is conditioned as normalised offset +
(sin,cos) of bearing -- never raw degrees, whose wrap at +/-180 sits right where
the rear-source pairs live.
"""
from __future__ import annotations
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import EarCNN, _gn1d, vicreg_var_cov, latent_stats  # reuse verified v1 pieces

ROOM_SCALE = 8.0   # metres; normaliser for transform offsets/range


def _gn(c: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class RangeCNN(nn.Module):
    """1D conv tower over the matched-filter range profile: [B,256] -> [B,feat]."""

    def __init__(self, feat: int = 128):
        super().__init__()
        def block(ci, co):
            return nn.Sequential(nn.Conv1d(ci, co, 5, stride=2, padding=2), _gn(co), nn.GELU())
        self.net = nn.Sequential(
            block(1, 32),      # 256 -> 128
            block(32, 64),     # -> 64
            block(64, 128),    # -> 32
            block(128, feat),  # -> 16
        )
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):                      # x: [B,256]
        h = self.net(x.unsqueeze(1))           # [B,feat,16]
        return self.pool(h).flatten(1)         # [B,feat]


class TwoStreamEncoder(nn.Module):
    """coch[B,2,64,48] + range[B,256] -> block latent (z_range, z_coch)."""

    def __init__(self, emb_range: int = 128, emb_coch: int = 128):
        super().__init__()
        self.range_tower = RangeCNN(emb_range)
        self.range_proj = nn.Sequential(nn.Linear(emb_range, emb_range), _gn1d(emb_range), nn.GELU(),
                                        nn.Linear(emb_range, emb_range))
        self.ear = EarCNN(emb_coch // 2)       # per-ear tower, weights shared across ears
        self.coch_proj = nn.Sequential(nn.Linear(emb_coch, emb_coch), _gn1d(emb_coch), nn.GELU(),
                                       nn.Linear(emb_coch, emb_coch))

    def forward(self, coch, rng):
        zr = self.range_proj(self.range_tower(rng))
        l = self.ear(coch[:, 0:1]); r = self.ear(coch[:, 1:2])
        zc = self.coch_proj(torch.cat([l, r], dim=1))
        return zr, zc


class TransformEmbed(nn.Module):
    """measured T row [B,6] = (dx,dy,dz,range,bearing_deg,fb_ok) -> [B,emb_T].
    Offsets/range normalised by room scale; bearing as (sin,cos) to avoid the
    +/-180 deg wrap discontinuity."""

    def __init__(self, emb_T: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(7, emb_T), nn.GELU(), nn.Linear(emb_T, emb_T))

    def forward(self, T):                       # T: [B,6]
        dx, dy, dz, rng, bear, fb = T.unbind(-1)
        b = torch.deg2rad(bear)
        feats = torch.stack([dx / ROOM_SCALE, dy / ROOM_SCALE, dz / ROOM_SCALE,
                             rng / ROOM_SCALE, torch.sin(b), torch.cos(b), fb], dim=-1)
        return self.mlp(feats)


class CrossViewPredictor(nn.Module):
    """Predict the listener's target blocks from the emitter's context blocks +
    the measured transform. The range head is HRTF-free; the coch head is
    conditioned on the listener's HRTF id (nuisance variable, plan point 5)."""

    def __init__(self, emb_range=128, emb_coch=128, emb_T=128, emb_ctx=256,
                 num_hrtf: int = 8, hidden: int = 512):
        super().__init__()
        self.ctx_fuse = nn.Sequential(nn.Linear(emb_range + emb_coch, emb_ctx), _gn1d(emb_ctx), nn.GELU())
        self.tf = TransformEmbed(emb_T)
        self.hrtf_embed = nn.Embedding(num_hrtf, emb_T)
        self.head_range = nn.Sequential(nn.Linear(emb_ctx + emb_T, hidden), _gn1d(hidden), nn.GELU(),
                                        nn.Linear(hidden, emb_range))
        self.head_coch = nn.Sequential(nn.Linear(emb_ctx + emb_T + emb_T, hidden), _gn1d(hidden), nn.GELU(),
                                       nn.Linear(hidden, emb_coch))

    def forward(self, zr_ctx, zc_ctx, T, hrtf_id):
        c = self.ctx_fuse(torch.cat([zr_ctx, zc_ctx], dim=1))
        t = self.tf(T)
        pr = self.head_range(torch.cat([c, t], dim=1))
        h = self.hrtf_embed(hrtf_id)
        pc = self.head_coch(torch.cat([c, t, h], dim=1))
        return pr, pc


class CrossViewJEPA(nn.Module):
    def __init__(self, emb_range=128, emb_coch=128, num_hrtf=8, ema=0.996,
                 lam_coch=0.25, var_coeff=1.0, cov_coeff=0.04, gamma=1.0):
        super().__init__()
        self.encoder = TwoStreamEncoder(emb_range, emb_coch)
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = CrossViewPredictor(emb_range, emb_coch, num_hrtf=num_hrtf)
        self.ema = ema; self.lam = lam_coch
        self.var_coeff = var_coeff; self.cov_coeff = cov_coeff; self.gamma = gamma

    @torch.no_grad()
    def update_target(self):
        m = self.ema
        for p, tp in zip(self.encoder.parameters(), self.target.parameters()):
            tp.mul_(m).add_(p, alpha=1.0 - m)
        for b, tb in zip(self.encoder.buffers(), self.target.buffers()):
            tb.copy_(b)

    def forward(self, ctx_coch, ctx_range, tgt_coch, tgt_range, T, hrtf_id):
        zr_c, zc_c = self.encoder(ctx_coch, ctx_range)              # online, emitter context
        with torch.no_grad():
            self.target.eval()
            zr_t, zc_t = self.target(tgt_coch, tgt_range)           # EMA, listener target (stop-grad)

        pr, pc = self.predictor(zr_c, zc_c, T, hrtf_id)
        L_range = F.mse_loss(pr, zr_t)
        L_coch = F.mse_loss(pc, zc_t)

        vr_r, cv_r = vicreg_var_cov(zr_c, self.gamma)               # per-block anti-collapse
        vr_c, cv_c = vicreg_var_cov(zc_c, self.gamma)
        var = vr_r + vr_c; cov = cv_r + cv_c
        total = L_range + self.lam * L_coch + self.var_coeff * var + self.cov_coeff * cov

        # "no-transform" baseline: regress target block onto the context block
        # unchanged. If L_range beats this, the predictor is USING the transform
        # (crossing the viewpoint), not just echoing the context.
        with torch.no_grad():
            base_range = F.mse_loss(zr_c, zr_t)
        return {"loss": total, "L_range": L_range.detach(), "L_coch": L_coch.detach(),
                "var": var.detach(), "cov": cov.detach(),
                "base_range": base_range.detach(), "zr": zr_c.detach(), "zc": zc_c.detach()}

    @torch.no_grad()
    def embed(self, coch, rng):
        """Frozen block embeddings for probes: -> (z_range, z_coch)."""
        return self.encoder(coch, rng)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B = 16
    model = CrossViewJEPA(num_hrtf=4).to(dev)
    ctx_coch = torch.randn(B, 2, 64, 48, device=dev)
    ctx_range = torch.randn(B, 256, device=dev)
    tgt_coch = torch.randn(B, 2, 64, 48, device=dev)
    tgt_range = torch.randn(B, 256, device=dev)
    T = torch.randn(B, 6, device=dev); T[:, 4] *= 180.0
    hrtf = torch.randint(0, 4, (B,), device=dev)
    out = model(ctx_coch, ctx_range, tgt_coch, tgt_range, T, hrtf)
    out["loss"].backward(); model.update_target()
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"device {dev}  params(trainable) {n/1e6:.2f}M")
    print("loss %.4f  L_range %.4f  L_coch %.4f  base_range %.4f  var %.4f  cov %.4f"
          % (out["loss"].item(), out["L_range"], out["L_coch"], out["base_range"], out["var"], out["cov"]))
    print("z_range", latent_stats(out["zr"]), "| z_coch", latent_stats(out["zc"]))
