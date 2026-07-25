"""Closed-loop MAPPER-NAV on the cluttered doorway task (EchoRL mapping expt, M2).

The headline. Each step the known-pose agent pings the TRUE server, appends the echo,
re-estimates the map (occupancy) and the doorway (goal), and plans on them with the
same grid receding-horizon planner M0 uses. Physics (collision/success) use TRUE
geometry -- the estimate only informs the plan. The value of CLOSED loop is that the
estimate is re-fused every step from new vantage points, so it can refine as the agent
moves even when the open-loop (static) estimate is poor.

Composes two estimate sources so each half of the map is isolated:
  occupancy : true | scatter (OccMapper U-Net)
  goal      : true | fuse    (EgoVecNet known-pose fusion) | scatter (OccMapper door head)

Ladder (RESULTS.md, nobs=3 goal): planning-oracle 1.00 | grid-planner-true-map (M0)
0.96 | greedy 0.54 | MPC 0.24-0.25 | PPO 0.08-0.18.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import mapping                                             # noqa: E402
from mapping import run_map_nav, true_occupancy, inflate, goal_from_openings  # noqa: E402
from ppo import opening_seeds                              # noqa: E402


def make_provider(occ_src="true", goal_src="true", occ_model=None, goal_model=None,
                  seq_model=None, thr=0.5):
    import mapper, mapper_fuse, mapper_seq
    def provider(srv):
        occ_true_infl = inflate(true_occupancy(srv.Lx, srv.Lz, srv.obstacles))
        goal_true = goal_from_openings(srv.openings)

        def step_map(x, z, yaw, history):
            rng_o, coch_o = srv.observe(x, z, yaw)
            history.append((x, z, yaw, rng_o.astype(np.float32), coch_o.astype(np.float32)))
            poses = np.array([[h[0], h[1], h[2]] for h in history], np.float32)
            rseq = np.stack([h[3] for h in history]); cseq = np.stack([h[4] for h in history])
            # occupancy
            if occ_src == "true":
                occ_infl = occ_true_infl
            else:
                occ_p, gest_s, _ = mapper.predict_grid(occ_model, poses, rseq, cseq)
                occ_infl = inflate(occ_p > thr)
            # goal
            if goal_src == "true":
                goal = goal_true
            elif goal_src == "fuse":
                goal = mapper_fuse.fuse(goal_model, poses, rseq, cseq)
            elif goal_src == "seq":
                goal = mapper_seq.predict(seq_model, poses, rseq, cseq)
            else:  # scatter door head
                _, goal, _ = mapper.predict_grid(occ_model, poses, rseq, cseq)
            return occ_infl, goal
        return step_map
    return provider


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--occ-ckpt", default="echorl/ckpt/mapper_nobs3.pt")
    ap.add_argument("--goal-ckpt", default="echorl/ckpt/egovec_goal_nobs3.pt")
    ap.add_argument("--seq-ckpt", default="echorl/ckpt/seqgoal_nobs3.pt")
    ap.add_argument("--eval-rooms", type=int, default=12)
    ap.add_argument("--episodes-per", type=int, default=10)
    ap.add_argument("--nobs", type=int, default=3)
    ap.add_argument("--t-max", type=int, default=40)
    # each mode = "occSrc+goalSrc"
    ap.add_argument("--modes", nargs="+",
                    default=["true+fuse", "scatter+true", "scatter+fuse", "true+scatter"])
    args = ap.parse_args()

    import mapper, mapper_fuse, mapper_seq
    occ_model = mapper.OccMapper().to(mapper.DEV)
    occ_model.load_state_dict(torch.load(args.occ_ckpt, map_location=mapper.DEV)); occ_model.eval()
    goal_model = mapper_fuse.EgoVecNet().to(mapper.DEV)
    goal_model.load_state_dict(torch.load(args.goal_ckpt, map_location=mapper.DEV)); goal_model.eval()
    seq_model = mapper_seq.SeqGoalNet().to(mapper.DEV)
    if os.path.exists(args.seq_ckpt):
        seq_model.load_state_dict(torch.load(args.seq_ckpt, map_location=mapper.DEV)); seq_model.eval()

    seeds = opening_seeds(600, args.eval_rooms, args.nobs)
    print(f"M2 closed-loop mapper-nav (nobs={args.nobs}, {len(seeds)} held-out doorway rooms "
          f"x {args.episodes_per})")
    print("ladder: planning-oracle 1.00 | true-map planner (M0) 0.96 | greedy 0.54 | "
          "MPC 0.24-0.25 | PPO 0.08-0.18\n")
    desc = {"true+fuse": "TRUE occ + fused goal  (doorway-mapping ceiling, perfect obstacles)",
            "scatter+true": "learned occ + TRUE goal (obstacle-mapping only)",
            "scatter+fuse": "learned occ + fused goal (FULL learned map -- the RESULT)",
            "true+scatter": "TRUE occ + scatter-CNN goal (CNN doorway head)"}
    for mode in args.modes:
        osrc, gsrc = mode.split("+")
        prov = make_provider(osrc, gsrc, occ_model=occ_model, goal_model=goal_model,
                             seq_model=seq_model)
        macro, per = run_map_nav(prov, seeds, nobs=args.nobs, episodes_per=args.episodes_per,
                                 t_max=args.t_max)
        lo, hi = min(per.values()), max(per.values())
        print(f"  {mode:>14} : {macro:.3f}  [{lo:.2f},{hi:.2f}]  -- {desc.get(mode, '')}")


if __name__ == "__main__":
    main()
