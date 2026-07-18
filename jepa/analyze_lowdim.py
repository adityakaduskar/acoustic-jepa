"""Why do raw pooling and a random projection decode geometry so well?

Hypothesis: the decodable geometry lives in LOW-ORDER STATISTICS of the echo
(total energy, the energy-vs-delay distribution = a reverberation/size proxy,
mean spectrum, coarse ILD), which pooling and random projection expose almost
losslessly -- so there is nothing high-order to disentangle and "keep
everything" beats "learn a compression".

Test: hand-craft ~7 physically-motivated statistics per cross-view observation
and ridge-probe them (room-level split), comparing to the full 256-d pooled-raw
features. If ~7 numbers match raw, the geometry demonstrably lives in low-order
statistics. Pure numpy (no torch needed).
"""
import argparse
import glob
import os
import numpy as np


def split_files(data_dir, frac=0.8, seed=0):
    files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    idx = np.random.default_rng(seed).permutation(len(files))
    n = max(1, int(round(frac * len(files))))
    return [files[i] for i in sorted(idx[:n])], [files[i] for i in sorted(idx[n:])]


def hand_features(coch, rng):
    """~7 physically-motivated low-order statistics from one cross obs.
    coch: [2,64,48] log-energy;  rng: [256] log-energy lag profile."""
    r = np.clip(rng, 0, None) + 1e-6
    k = np.arange(len(r))
    w = r / r.sum()
    centroid = float((k * w).sum())                        # mean echo delay ~ room size / reverb
    spread = float(np.sqrt(((k - centroid) ** 2 * w).sum()))  # decay spread
    late_early = float(r[128:].mean() / (r[:128].mean() + 1e-6))   # late/early energy ~ reverberation
    r_mean = float(r.mean())                               # total reflected energy
    ct = coch.mean(axis=2)                                 # [2,64] time-pooled spectrum
    c_mean = float(ct.mean())                              # total binaural energy
    bands = np.arange(64)
    spec = ct.mean(axis=0)                                 # [64]
    spec_centroid = float((bands * spec).sum() / (spec.sum() + 1e-6))  # brightness
    ild = float(ct[0].mean() - ct[1].mean())               # broadband ILD ~ bearing
    return np.array([r_mean, centroid, spread, late_early, c_mean, spec_centroid, ild], np.float32)


def load(files):
    Xhand, Xraw, vol, pos = [], [], [], []
    for f in files:
        with np.load(f) as d:
            cc = d["cross_coch"].astype(np.float32)   # [P,2,64,48]
            cr = d["cross_range"].astype(np.float32)  # [P,256]
            pair = d["pair"]; rv = d["room_vol"]; tt = d["T_true"]
        for r in range(cc.shape[0]):
            Xhand.append(hand_features(cc[r], cr[r]))
            raw_r = cr[r].reshape(64, 4).mean(1)                       # 256->64
            raw_c = cc[r].mean(2).reshape(-1)                          # [128]
            Xraw.append(np.concatenate([raw_r, raw_c]))
            L = int(pair[r, 0])
            vol.append(rv[L]); pos.append(tt[r, [0, 2]])
    return (np.array(Xhand), np.array(Xraw, np.float32),
            np.array(vol, np.float32)[:, None], np.array(pos, np.float32))


def ridge_r2(Xtr, ytr, Xte, yte, lam=1.0):
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-6
    Xn = (Xtr - mu) / sd; Xe = (Xte - mu) / sd
    Xa = np.concatenate([Xn, np.ones((len(Xn), 1))], 1)
    Ea = np.concatenate([Xe, np.ones((len(Xe), 1))], 1)
    R = lam * np.eye(Xa.shape[1]); R[-1, -1] = 0
    W = np.linalg.solve(Xa.T @ Xa + R, Xa.T @ ytr)
    yh = Ea @ W
    ss = ((yte - yh) ** 2).sum(0); tot = ((yte - yte.mean(0)) ** 2).sum(0) + 1e-9
    return float((1 - ss / tot).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    args = ap.parse_args()
    tr, te = split_files(args.data_dir)
    Xh_tr, Xr_tr, v_tr, p_tr = load(tr)
    Xh_te, Xr_te, v_te, p_te = load(te)
    print(f"{len(tr)} train / {len(te)} test rooms | {len(Xh_tr)} train / {len(Xh_te)} test obs")
    print(f"hand features ({Xh_tr.shape[1]}): r_mean, delay_centroid, delay_spread, "
          f"late/early, coch_mean, spec_centroid, ILD\n")

    print(f"{'target':10s} {'7 hand-stats':>13s} {'raw(192-d)':>11s} {'3 reverb-only':>14s}")
    # room_vol: full hand set, raw, and just the 3 reverberation stats (centroid,spread,late/early)
    rv_hand = ridge_r2(Xh_tr, v_tr, Xh_te, v_te)
    rv_raw = ridge_r2(Xr_tr, v_tr, Xr_te, v_te)
    rv_rev = ridge_r2(Xh_tr[:, [1, 2, 3]], v_tr, Xh_te[:, [1, 2, 3]], v_te)
    print(f"{'room_vol':10s} {rv_hand:>13.3f} {rv_raw:>11.3f} {rv_rev:>14.3f}")
    pp_hand = ridge_r2(Xh_tr, p_tr, Xh_te, p_te)
    pp_raw = ridge_r2(Xr_tr, p_tr, Xr_te, p_te)
    print(f"{'peer_pos':10s} {pp_hand:>13.3f} {pp_raw:>11.3f} {'-':>14s}")

    print("\nread: if 7 hand-stats (or even 3 reverberation stats) match raw on room_vol, "
          "the geometry lives in low-order statistics that pooling/random projection expose "
          "losslessly -- nothing high-order to learn.")


if __name__ == "__main__":
    main()
