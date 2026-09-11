"""
Per-channel exploration report for a checkpoint.

The scalar logged as "entropy" in history.jsonl is a mean over *active* action
channels (models.py rescales by act_dim/active_dims so ppo.py's division by
act_dim lands on the active-mean). That average hides which axis is collapsing:
one channel pinned to the log_std floor and one channel wide open can produce
the same number. This prints the per-channel picture the average is made of.

Usage:
    python scripts/entropy_report.py [checkpoint_path]
"""
import math
import sys

import torch

# 0.5 * log(2 * pi * e) -- the constant term of a 1-D Gaussian's differential
# entropy, so that H = log_std + GAUSS_H_CONST.
GAUSS_H_CONST = 0.5 * math.log(2.0 * math.pi * math.e)

CONT_NAMES = ["throttle", "steer", "pitch", "yaw", "roll"]
BIN_NAMES = ["jump", "boost", "handbrake"]


def bernoulli_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log(p) + (1.0 - p) * math.log(1.0 - p))


def main(path: str, bin_rates=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model_state_dict", ck)
    log_std = sd["actor_log_std"].flatten().tolist()
    ceiling = sd["log_std_max"].flatten().tolist()
    # Checkpoints predating the per-channel floor carry no log_std_min buffer; those are
    # the runs that were on the old shared -2.5.
    if "log_std_min" in sd:
        floors = sd["log_std_min"].flatten().tolist()
    else:
        floors = [-2.5] * len(CONT_NAMES)

    print(f"checkpoint      {path}")
    print(f"iteration       {ck.get('iteration')}")
    start = ck.get("rot_anneal_start_iter")
    if start is not None:
        span = 4000  # hyperparameters.rot_log_std_anneal_iters
        pct = min(100.0, 100.0 * (ck.get("iteration", 0) - start) / span)
        print(f"rot anneal      started iter {start}, {pct:.0f}% complete")
    print()
    print(f"{'channel':>10} {'log_std':>8} {'sigma':>7} {'H':>7} {'floor':>6} {'ceiling':>8} {'headroom':>9}  state")

    total_h = 0.0
    for name, ls, ceil, floor in zip(CONT_NAMES, log_std, ceiling, floors):
        h = ls + GAUSS_H_CONST
        total_h += h
        # How far the channel sits above the floor, as a fraction of its band.
        band = ceil - floor
        frac = (ls - floor) / band if band > 0 else 0.0
        # Distinguish a parameter held AT its bound from one latched UNDER it. clamped_log_std()
        # cuts the gradient only once the parameter passes the bound, so a channel resting a hair
        # above its floor is in a live equilibrium -- the entropy bonus is still reaching it and
        # can lift it the moment the policy gradient eases off. A channel below the floor reads as
        # the floor in the forward pass while its raw parameter is unreachable, which is the
        # failure that went unnoticed for thousands of iterations. Proximity alone cannot tell
        # them apart, and calling both "collapsed" is what made the working state look broken.
        if ls < floor - 1e-4:
            state = f"LATCHED {floor - ls:.3f} BELOW FLOOR - gradient cut"
        elif ls <= floor + 0.05:
            state = "resting on floor - bounded, gradient live"
        elif ls >= ceil - 0.05:
            state = "at ceiling - anneal/entropy bound"
        elif frac < 0.25:
            state = "near floor"
        else:
            state = "healthy"
        print(f"{name:>10} {ls:8.3f} {math.exp(ls):7.3f} {h:7.3f} {floor:6.2f} {ceil:8.2f} {frac*100:8.0f}%  {state}")

    rates = bin_rates or {}
    for name in BIN_NAMES:
        p = rates.get(name)
        if p is None:
            continue
        h = bernoulli_entropy(p)
        total_h += h
        print(f"{name:>10} {'':>8} {'p=%.3f' % p:>7} {h:7.3f} {'':>6} {math.log(2):8.3f} {h/math.log(2)*100:8.0f}%  "
              f"{'near-deterministic' if h < 0.25 else 'healthy'}")

    n = len(CONT_NAMES) + len([k for k in BIN_NAMES if k in rates])
    print()
    print(f"active-mean entropy (airborne, all buttons live): {total_h / n:+.3f}")
    print("  this is the number logged as 'entropy' in history.jsonl")
    print()
    ceil_h = sum(c + GAUSS_H_CONST for c in ceiling) + len(rates) * math.log(2)
    floor_h = sum(f + GAUSS_H_CONST for f in floors)
    print(f"  ceiling (max exploration under current caps): {ceil_h / n:+.3f}")
    print(f"  floor   (every Gaussian axis collapsed):      {floor_h / n:+.3f}")
    cur = total_h / n
    print(f"  current sits {100 * (cur - floor_h / n) / (ceil_h / n - floor_h / n):.0f}% of the way up from the floor")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/latest_model.pt"
    # Button rates come from telemetry in logs/metrics.json (jump_rate_pct etc).
    main(ckpt, {"jump": 0.059, "boost": 0.112, "handbrake": 0.192})
