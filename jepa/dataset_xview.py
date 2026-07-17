"""Cross-view dataset over the swarm_dump .npz schema (v2).

One training item = one ordered cross-view pair (emitter E, listener L):
  * CONTEXT = emitter E's own MONOSTATIC observation (mono_coch[E], mono_range[E]);
  * TARGET  = listener L's cross-view RECEPTION of E's ping (cross_coch[row],
              cross_range[row]);
  * TRANSFORM = the MEASURED relative geometry (T_meas[row]) -- self-supervised
                (bearing from the listener's own-ear ITD calibration, range from
                the direct arrival); T_true is available for scoring/ablation.
  * hrtf_id = listener index (the listener's ears colour the coch target; with
              the current all-default-HRTF dumps this is a no-op, becomes
              meaningful once swarm_dump assigns a distinct SOFA per agent index).

Predicting L's reception from E's own view REQUIRES crossing the viewpoint, which
is the allocentric pressure the whole v2 design is testing. Occluded pairs are
kept IN and their occlusion flag is NOT given to the model (inferring a blocked
path from geometry is part of the task; feeding the flag would be oracle leakage)
-- the flag is carried only for probe/analysis splits.

Episodes are split BY FILE (room), so no room appears in both train and test.
Lazy per-file load with a small LRU cache. Runs in the torch venv
(AcousticJEPA/.venv).
"""
from __future__ import annotations
import functools
import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _find_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no .npz episodes under {data_dir!r}")
    return files


def split_files(data_dir: str, train_frac: float = 0.8, seed: int = 0):
    """Deterministic episode-level (room-level) split."""
    files = _find_files(data_dir)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    n_tr = max(1, int(round(train_frac * len(files))))
    tr = [files[i] for i in sorted(idx[:n_tr])]
    te = [files[i] for i in sorted(idx[n_tr:])] or tr[-1:]   # never empty
    return tr, te


@functools.lru_cache(maxsize=64)
def _load(path: str) -> dict:
    with np.load(path) as d:
        return {
            "mono_coch": d["mono_coch"].astype(np.float32),   # [N,2,64,48]
            "mono_range": d["mono_range"].astype(np.float32), # [N,256]
            "cross_coch": d["cross_coch"].astype(np.float32), # [P,2,64,48]
            "cross_range": d["cross_range"].astype(np.float32),
            "pair": d["pair"].astype(np.int64),               # [P,2] = (L,E)
            "T_true": d["T_true"].astype(np.float32),         # [P,6]
            "T_meas": d["T_meas"].astype(np.float32),         # [P,6]
            "occluded": d["occluded"].astype(np.int64),       # [P]
            "dist_wall": d["dist_wall"].astype(np.float32),   # [N,8]
            "room_vol": d["room_vol"].astype(np.float32),     # [N]
            "material": d["material"].astype(np.int64),       # [N]
            "agent_pose": d["agent_pose"].astype(np.float32), # [N,4]
        }


class CrossViewDataset(Dataset):
    def __init__(self, files: list[str], use_measured: bool = True):
        self.files = list(files)
        self.use_measured = use_measured
        self.index: list[tuple[int, int]] = []
        for fi, f in enumerate(self.files):
            with np.load(f) as d:
                P = int(d["pair"].shape[0])
            self.index += [(fi, r) for r in range(P)]

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        fi, r = self.index[i]
        ep = _load(self.files[fi])
        L, E = int(ep["pair"][r, 0]), int(ep["pair"][r, 1])
        if self.use_measured:
            T = ep["T_meas"][r]                       # (dx,dy,dz,range,bearing,fb_ok)
        else:
            # T_true is (dx,dy,dz,range,rel_yaw,bearing_local) -- repack into the
            # SAME (dx,dy,dz,range,bearing,fb_ok) layout TransformEmbed expects,
            # so the oracle-transform ablation is a clean apples-to-apples swap.
            tt = ep["T_true"][r]
            T = np.array([tt[0], tt[1], tt[2], tt[3], tt[5], 1.0], dtype=np.float32)
        return {
            "ctx_coch": torch.from_numpy(ep["mono_coch"][E].copy()),
            "ctx_range": torch.from_numpy(ep["mono_range"][E].copy()),
            "tgt_coch": torch.from_numpy(ep["cross_coch"][r].copy()),
            "tgt_range": torch.from_numpy(ep["cross_range"][r].copy()),
            "T": torch.from_numpy(T.copy()),
            "hrtf_id": torch.tensor(L, dtype=torch.long),
            "occluded": torch.tensor(int(ep["occluded"][r])),
            # listener-side labels (for probes; unused in the predictive loss)
            "L_dist_wall": torch.from_numpy(ep["dist_wall"][L].copy()),
            "L_room_vol": torch.tensor(ep["room_vol"][L]),
            "L_material": torch.tensor(int(ep["material"][L])),
            "T_true": torch.from_numpy(ep["T_true"][r].copy()),
        }


def make_loader(files, batch_size=128, shuffle=True, num_workers=0, use_measured=True):
    ds = CrossViewDataset(files, use_measured=use_measured)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, drop_last=shuffle, pin_memory=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="smoke-test the cross-view dataset")
    ap.add_argument("data_dir")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()

    tr, te = split_files(args.data_dir, 0.8, seed=0)
    print(f"episodes: {len(tr)} train / {len(te)} test (room-level split)")
    dl = make_loader(tr, batch_size=args.batch, num_workers=0)
    print(f"train cross-view pairs: {len(dl.dataset)}")
    b = next(iter(dl))
    for k in ["ctx_coch", "ctx_range", "tgt_coch", "tgt_range", "T", "hrtf_id", "occluded"]:
        v = b[k]
        print(f"  {k:11s} {tuple(v.shape)} {v.dtype}")
    print("occluded frac in batch:", float(b["occluded"].float().mean()))
    print("T (measured) row 0:", [round(x, 2) for x in b["T"][0].tolist()])
    print("T_true      row 0:", [round(x, 2) for x in b["T_true"][0].tolist()])
