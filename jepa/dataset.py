"""Dataset for the Acoustic-JEPA (ACOUSTIC_JEPA_SPEC.md).

Loads the .npz episodes written by jepa_dump.cpp and yields action-conditioned
transition *windows* for JEPA pretraining, plus a single-observation view for the
frozen-latent geometry probes.

Window semantics (matches jepa_dump.cpp): within one episode of K steps, a window
of horizon H starting at step s provides observations coch[s .. s+H] and actions
[s .. s+H-1], where action[t] maps observation t -> observation t+1. Valid starts
are s in [0, K-H).

Episodes are loaded lazily with a small LRU cache, so the full dataset never has
to fit in RAM; only `coch` (+ labels) are touched for training (`raw` is left on
disk). Meant to run in a Python 3.11/3.12 venv with torch installed.
"""
from __future__ import annotations
import functools
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Action ranges must match acoustic_env.h Action (dyaw deg, dstep m).
DYAW_MAX = 45.0
DSTEP_MAX = 1.5


def normalize_action(a: np.ndarray) -> np.ndarray:
    """Map raw (dyaw deg, dstep m) into roughly [-1, 1]. Operates on a copy."""
    out = np.asarray(a, dtype=np.float32).copy()
    out[..., 0] = out[..., 0] / DYAW_MAX                # [-45,45] -> [-1,1]
    out[..., 1] = out[..., 1] / DSTEP_MAX * 2.0 - 1.0   # [0,1.5]  -> [-1,1]
    return out


@functools.lru_cache(maxsize=64)
def _load_episode(path: str) -> dict:
    """Load one .npz, materialising only the arrays we need (coch as f32)."""
    with np.load(path) as d:
        return {
            "coch": d["coch"].astype(np.float32),        # [K,2,64,48]
            "actions": d["actions"].astype(np.float32),  # [K,2]
            "labels_dist": d["labels_dist"].astype(np.float32),
            "open_az": d["open_az"].astype(np.float32),
            "room_vol": d["room_vol"].astype(np.float32),
            "material": d["material"].astype(np.int64),
            "pose": d["pose"].astype(np.float32),
        }


def _find_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no .npz episodes found under {data_dir!r}")
    return files


def _label_at(ep: dict, t: int) -> dict:
    return {
        "dist_wall": torch.from_numpy(ep["labels_dist"][t].copy()),   # [8]
        "open_az": torch.tensor(ep["open_az"][t]),                     # scalar (may be NaN)
        "room_vol": torch.tensor(ep["room_vol"][t]),                  # scalar
        "material": torch.tensor(int(ep["material"][t])),             # scalar long
        "pose": torch.from_numpy(ep["pose"][t].copy()),               # [4]
    }


class JEPADataset(Dataset):
    """Action-conditioned transition windows for JEPA pretraining.

    __getitem__ -> (coch, actions, label0):
        coch     float32 [H+1, 2, 64, 48]   observations t=0..H
        actions  float32 [H, 2]             normalised, action[k]: obs k -> k+1
        label0   dict                       GT labels at the context step (t=0)
    """

    def __init__(self, data_dir: str, horizon: int = 8, window: int = 1):
        self.files = _find_files(data_dir)
        self.H = int(horizon)
        self.W = int(window)
        span = self.W + self.H                            # frames per training window
        self.index: list[tuple[int, int]] = []
        for fi, f in enumerate(self.files):
            with np.load(f) as d:
                K = int(d["actions"].shape[0])
            if K < span:
                continue                                  # episode too short
            for s in range(0, K - span + 1):
                self.index.append((fi, s))
        if not self.index:
            raise ValueError(
                f"no windows: window+horizon={span} but episodes have too few steps. "
                f"Regenerate with more steps or lower --window/--horizon."
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        fi, s = self.index[i]
        ep = _load_episode(self.files[fi])
        span = self.W + self.H
        coch = torch.from_numpy(ep["coch"][s:s + span].copy())              # [W+H,2,64,48]
        act = torch.from_numpy(normalize_action(ep["actions"][s:s + span - 1]))  # [W+H-1,2]
        return coch, act, _label_at(ep, s + self.W - 1)   # label at context-newest frame


class ProbeDataset(Dataset):
    """Single observation + GT labels, for frozen-latent linear probes (§4a)."""

    def __init__(self, data_dir: str, window: int = 1):
        self.files = _find_files(data_dir)
        self.W = int(window)
        self.index: list[tuple[int, int]] = []
        for fi, f in enumerate(self.files):
            with np.load(f) as d:
                K = int(d["actions"].shape[0])
            for t in range(self.W - 1, K):                # need W frames ending at t
                self.index.append((fi, t))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int):
        fi, t = self.index[i]
        ep = _load_episode(self.files[fi])
        W = self.W
        coch = torch.from_numpy(ep["coch"][t - W + 1:t + 1].copy())   # [W,2,64,48]
        return coch, _label_at(ep, t)


def make_loader(data_dir: str, horizon: int = 8, window: int = 1, batch_size: int = 64,
                shuffle: bool = True, num_workers: int = 4, probe: bool = False):
    ds = ProbeDataset(data_dir, window) if probe else JEPADataset(data_dir, horizon, window)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=not probe, pin_memory=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="smoke-test the JEPA dataset")
    ap.add_argument("data_dir")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    dl = make_loader(args.data_dir, horizon=args.horizon,
                     batch_size=args.batch, num_workers=0)
    print(f"windows: {len(dl.dataset)}  (horizon={args.horizon})")
    coch, act, lab = next(iter(dl))
    print("coch    ", tuple(coch.shape), coch.dtype)
    print("actions ", tuple(act.shape), "range", (float(act.min()), float(act.max())))
    print("dist_wall", tuple(lab["dist_wall"].shape))
    print("material ", lab["material"].tolist())

    pl = make_loader(args.data_dir, batch_size=args.batch, num_workers=0, probe=True)
    print(f"probe items: {len(pl.dataset)}")
    pcoch, plab = next(iter(pl))
    print("probe coch", tuple(pcoch.shape))
