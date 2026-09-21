"""
League ratings as one batch fit over every series ever played, anchored on a pinned checkpoint.

Why this replaced online TrueSkill (2026-09-21). TrueSkill is a filter for players whose skill
changes between games; it updates each rating once, in order, against whatever the opponent's
rating happened to be at the time. A checkpoint never changes. Its every result stays exactly as
valid as the day it was played, so the right estimate is the one that explains all of them at
once, re-solved as results arrive. That is a probit Bradley-Terry model -- TrueSkill's own
likelihood, P(A beats B) = Phi((mu_A - mu_B) / (sqrt(2) * beta)), so mu keeps its units -- fitted
by maximum a posteriori over the whole results log.

What the online filter did wrong here, and why the batch fit does not:
  - Scale drift. The v5 league climbed from mu 19 to 41 against an anchored Necto while losing to
    it 35-1. The online update is order-dependent and, with ratings locked at 64 series and
    newcomers seeded from their predecessor, not zero-sum; a genuinely improving population
    ratchets up faster than a saturated anchor can pull it down. Simulated with an improving,
    noisy population (scratch gradesim.py, 8 seeds, drift per 100 checkpoints of estimate minus
    truth): online TrueSkill -7.1 +/- 0.6, batch fit -0.9 +/- 0.5, batch fit with Necto in the
    data -0.1 +/- 0.6. Rank correlation with true skill over the last 50: 0.57 vs 0.83.
  - The rating lock existed to stop established ratings random-walking under a stream of
    sigma-8 newcomers. In a joint fit there is no walk to stop: every result is weighed once
    against every other, so the lock is retired.
  - A saturated anchor. Necto wins every series, so pinning it pins the scale to a reference
    no result can reach. Here only one checkpoint is pinned -- the v3 king, which the population
    plays close enough for every series to carry information -- and Necto, Nexto and the
    heuristic are fitted like anyone else. Their results can no longer distort the scale; they
    only place the references on it.

The prior, N(prior_mu, 25/3), is TrueSkill's own starting belief. It is what keeps an undefeated
record finite (Necto has never lost a series to this league) and what lets a newcomer start at
its predecessor's level; after a couple of dozen series the data dominate it.
"""
from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from scipy.special import log_ndtr

BETA = 25.0 / 6.0
SCALE = math.sqrt(2.0) * BETA          # the probit denominator: a 1v1 performance difference
PRIOR_MU = 25.0
PRIOR_SD = 25.0 / 3.0
DEFAULT_RESULTS_PATH = "logs/trueskill_leaderboard.results.jsonl"

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)


def series_score(a_score: int, b_score: int) -> float:
    """1 for an A series win, 0 for a loss, 0.5 for the rare nine-episode tie."""
    return 1.0 if a_score > b_score else (0.0 if a_score < b_score else 0.5)


def append_result(path: str, a: str, b: str, a_score: int, b_score: int, kind: str, at: str) -> Dict:
    """Appends one series to the log. Append-only: a result, once played, is never rewritten."""
    row = {"a": a, "b": b, "a_score": int(a_score), "b_score": int(b_score), "kind": kind, "at": at}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def load_results(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a torn final line from a crash costs one series, not the log
    return rows


def fit_ratings(
    results: Iterable[Dict],
    pinned: Dict[str, float],
    prior_mu: Optional[Dict[str, float]] = None,
    start: Optional[Dict[str, float]] = None,
    prior_sd: float = PRIOR_SD,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> Dict[str, Tuple[float, float]]:
    """
    MAP ratings for everyone in `results`, as {name: (mu, sd)}.

    `pinned` players are held at their value with sd 0 and define the scale. Players connected
    to no pinned player are still fitted, but only relative to each other and the prior.
    `sd` is the posterior standard deviation from the curvature at the optimum, the same
    quantity TrueSkill's sigma estimates.

    Results are aggregated per ordered pair first, so the cost grows with the number of players
    who have met, not with the number of series.
    """
    prior_mu = prior_mu or {}
    start = start or {}
    agg: Dict[Tuple[str, str], List[float]] = defaultdict(lambda: [0.0, 0.0])  # [score, count]
    names: Dict[str, int] = {}
    for r in results:
        a, b = r["a"], r["b"]
        if a == b:
            continue
        for x in (a, b):
            names.setdefault(x, len(names))
        cell = agg[(a, b)]
        cell[0] += series_score(r["a_score"], r["b_score"])
        cell[1] += 1.0
    for x in pinned:
        names.setdefault(x, len(names))
    n = len(names)
    if n == 0:
        return {}

    idx = {k: v for k, v in names.items()}
    i = np.array([idx[a] for a, _ in agg], dtype=np.int64)
    j = np.array([idx[b] for _, b in agg], dtype=np.int64)
    wins = np.array([c[0] for c in agg.values()], dtype=np.float64)       # score of i against j
    games = np.array([c[1] for c in agg.values()], dtype=np.float64)
    losses = games - wins

    m0 = np.array([prior_mu.get(k, PRIOR_MU) for k in names], dtype=np.float64)
    mu = np.array([start.get(k, prior_mu.get(k, PRIOR_MU)) for k in names], dtype=np.float64)
    free = np.ones(n, dtype=bool)
    for k, v in pinned.items():
        mu[idx[k]] = float(v)
        free[idx[k]] = False
    f = np.where(free)[0]
    inv_var = 1.0 / (prior_sd * prior_sd)

    H = np.zeros((n, n))
    for _ in range(max_iter):
        d = (mu[i] - mu[j]) / SCALE
        # Inverse Mills ratios phi(d)/Phi(d) and phi(d)/Phi(-d), in log space so a 500-mu gap
        # neither overflows nor divides by zero.
        lp = np.exp(-0.5 * d * d - _LOG_SQRT_2PI - log_ndtr(d))
        lm = np.exp(-0.5 * d * d - _LOG_SQRT_2PI - log_ndtr(-d))
        g = (wins * lp - losses * lm) / SCALE
        h = (wins * lp * (lp + d) + losses * lm * (lm - d)) / (SCALE * SCALE)   # >= 0
        grad = -(mu - m0) * inv_var
        np.add.at(grad, i, g)
        np.add.at(grad, j, -g)
        H[:] = 0.0
        np.add.at(H, (i, i), h)
        np.add.at(H, (j, j), h)
        np.add.at(H, (i, j), -h)
        np.add.at(H, (j, i), -h)
        H[np.diag_indices(n)] += inv_var
        if f.size == 0:
            break
        step = np.linalg.solve(H[np.ix_(f, f)], grad[f])
        # Damped: the objective is concave, but a first step from a far start can overshoot.
        big = np.max(np.abs(step))
        if big > 10.0:
            step *= 10.0 / big
        mu[f] += step
        if big < tol:
            break

    sd = np.zeros(n)
    if f.size:
        sd[f] = np.sqrt(np.clip(np.diag(np.linalg.inv(H[np.ix_(f, f)])), 0.0, None))
    return {k: (float(mu[v]), float(sd[v])) for k, v in names.items()}
