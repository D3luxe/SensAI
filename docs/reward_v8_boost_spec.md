# Reward v8: boost as a ratchet

Status: **closed, not adopted, 2026-09-23** (§7). Stopped at 226M.
Identity: code `d21ef08af7da3f38`, settings `3ea7be8882fbd05c` (`config/reward_versions/v8.json`).
Start checkpoint: `checkpoints/baselines/v5_iter222000.pt` — lineage stays at v5, since neither v6
nor v7 was adopted.

## 1. Why

Three runs in a row have plateaued against Necto at a goal difference between −27 and −32, and
each failed its gate for a different-looking reason. This version starts from a different question:
not "what else could we reward" but "what does the reward currently *say*".

**v5's live reward is four terms, and two of them provably cannot change behaviour.**
`closeness_weight` has been 0.0 since v4, so what actually runs is:

| | term | type |
|---|---|---|
| T1 | goal, ±10 / −7.5 | sparse |
| T2 | ball_position, weight 5 | potential |
| T3 | touch, 0.5 × min(1, \|Δv_ball\|/2300) | dense, non-potential |
| T5 | boost, weight 1 | potential |

Potential-based shaping telescopes. Over an episode ending in a goal, where φ(terminal) = 0, a
potential term's total discounted contribution is exactly **−w · φ(s₀)** — a constant fixed by the
start state, independent of the policy. That is what makes a potential unfarmable (v3 spec, R4),
and it is also a proof that T2 and T5 cannot change the optimal policy. They move credit around and
nothing else.

So the entire behavioural specification the bot has ever been trained against is: **score goals,
and hit the ball hard.** T3 is the only dense term that can move the policy, and it prices impact
with no regard for direction, possession or position.

This explains several things that had been filed as separate puzzles:

- **Big-pad pickups never moved.** ~0.5/min across the whole of v5, v6 and v7, regardless of every
  other change. T5 is a potential; no setting of `boost_weight` could ever have moved it.
- **v7's regression came from its start distribution with no training pathology at all.** Explained
  variance (0.817–0.831) and value loss (0.070–0.081) are indistinguishable from v5's and v6's. The
  policy optimised v5's reward successfully and arrived somewhere worse at football, because the
  reward under-specifies the game and the start distribution had been silently supplying the rest.
- **Touches per minute fell while play got worse.** T3 rewards hard touches, not many touches, so
  the incentive is the big swing rather than the possession.

## 2. What the replay data says to reward

Before choosing a term, candidate features were scored on the 225 human 1v1 replays in the pool
(781 minutes of live play, 2,019 goals) by how well each ranks "team 0 scores next" above "team 1
scores next", over a forward window separated from the measurement by a gap so that no feature can
see the shot it is predicting. Without that gap every feature is contaminated: possession scores
0.794 and ball position 0.825 purely because whoever last touched the ball a second before a goal
is the scorer.

| feature | AUC (3s back, 4s gap, 8s forward) |
|---|---|
| **boost differential** | **0.618** |
| distance-to-own-goal differential | 0.611 |
| car boost | 0.574 |
| goal-side differential | 0.517 |
| territorial progress under possession | 0.480 |
| possession share | 0.461 |
| ball y | 0.386 *(inverted)* |

Read in order:

- **Boost differential is the strongest feature and survives controlling for the others** (+0.212
  per standard deviation in a joint fit, 2–3× any other). The effect is large: in the bottom
  quintile of boost differential a team scores the next goal 32.4% of the time, in the top quintile
  70.2%.
- **Ball position inverts.** Having the ball advanced in your attacking half predicts *conceding*
  4–12 seconds later. The counter-attack is the dominant dynamic in 1v1, and it is why a
  territorial reward was considered and rejected: it would have paid for entering the state that
  precedes conceding, which is the direction v7 was already drifting.
- **Goal-side differential is at chance** (0.517). That is v6's T6, and it retroactively explains
  why adding it bought nothing.

And the bot is worse at the winning feature than at anything else it is measured on:

| | humans (this pool) | SenseiBot v7 @ 400M |
|---|---|---|
| boost collected | 411 / min | 173 / min |
| **big pads** | **4.1 / min** | **0.5 / min** |
| small pads | 6.9 / min | — |
| mean boost held | 42 | — |

One eighth the big-pad rate, and 60% of its retreats start on low boost.

## 3. The change

**T5 boost stops being a potential and becomes a ratchet, T7 boost_gain.** The same quantity, paid
on the rise and not charged on the fall:

```
T7  boost_gain = boost_gain_weight * max(0, sqrt(b'/100) - sqrt(b/100))
```

where `b` is the car's boost on the previous step and `b'` on this one. `boost_weight` goes to 0.0
and `boost_gain_weight` to 2.0. Every other weight is v5's, unchanged.

The asymmetry is the whole change, and it is what makes T7 non-telescoping and therefore able to
move the policy at all. Because it is not a potential it carries no gamma — it prices a transition,
not a difference of state values — and R4's φ(terminal) = 0 does not apply to it: a pickup on the
goal step is a real pickup.

**This is one behavioural change, not two.** Removing T5 removes a term that provably could not
affect the optimal policy, so the only thing that changes about what the bot is being asked to do
is the ratchet.

Three properties follow from reusing T5's concave `sqrt` rather than raw boost:

- **Path independence.** `sqrt` telescopes across a monotonic rise, so 0 → 100 pays exactly 1.0
  whether it is taken as one big pad or eight small ones. There is nothing to gain by splitting a
  pickup and no incentive to nibble.
- **Refuelling from empty is what pays.** 0 → 30 pays 0.55; 70 → 100 pays 0.16. The bot is paid for
  the pickup that ends a low-boost retreat, which is the failure the eval actually records.
- **Bounded per refill.** A full tank is worth `boost_gain_weight` and no more.

### Weight sizing

At 2.0, a full refill is worth 2.0 against a goal's 10.0. For scale: at v7's rates (2.25 goals for
and 32.5 against per 10 minutes, every episode ending in a goal) an episode lasts about 17 seconds
and its goal term averages about **−6.4**. A full refill is therefore roughly a third of the goal
term's typical magnitude — large enough to be visible after three versions in which the boost term
was exactly zero, and small enough that conceding still dominates everything.

### Start distribution

v7's replay pruning and mirroring are kept; **its tag weighting is dropped.** `tag_weights` are set
to the frozen pool's own frequencies, so drawing a kept frame is uniform again:

| tag | natural (v8) | v7 forced | v7 distortion |
|---|---|---|---|
| open_play | 41.65% | 20% | 0.48× |
| aerial | 22.90% | 17.5% | 0.76× |
| challenge | 19.56% | 30% | 1.53× |
| goal_threat | 13.24% | 20% | 1.51× |
| carry | 2.14% | 7.5% | 3.5× |
| carry_contested | 0.50% | 5% | 10× |

**This is a judgement call and it is the one part of v8 that is not single-variable against v7.**
The reasoning: v7's stated brief was to remove dead frames, which worked; the tag weighting was the
part that went beyond the brief, and it concentrated 67.5% of replay starts into
challenge/goal_threat/aerial duels during the run that regressed. Carrying a setting we believe is
harmful purely to preserve single-variable purity would spend 400M steps defending it. The pool
itself is unchanged — same 1,402,903 kept frames, same sha `8fb61d856a48bc4c`.

If this turns out to matter, it is recoverable: v7's tag weights are frozen in `v7.json` and the
comparison can be run later.

## 4. The run

Start from `checkpoints/baselines/v5_iter222000.pt`. Every checkpoint is evaluated automatically
(`scripts/auto_eval.py`), and decision points pool every checkpoint within ±25M (~20 checkpoints,
~60 seeds), per `ui/eval_pooling.py`.

**Decision points:**

- **100M — the farming check.** T7 is positive-sum in self-play: spending is free, so
  collect → dump → collect pays every cycle. Pads respawn on a timer and a car cycling pads is not
  near the ball, but nothing in the formula prevents it, deliberately — a gate would violate R6 and
  would only hide the behaviour. **`touches_per_min` is the canary.** If pooled touches fall clearly
  below v7's 6.3–6.7 band while `boost_collected_per_min` climbs, the bots are farming pads and the
  run is stopped. Judge boost collection as a success signal only alongside touches holding.
- **150M — the passivity check**, as in v7: stop and review if the pooled head to head is clearly
  negative (below −2 standard errors) or a guardrail is clearly broken. Passivity is judged against
  **v5 at the same step count**, not its 400M numbers.
- **400M — adoption**, on the criterion below.

## 5. The 400M criterion

Written before the run starts. Pooled over every checkpoint within ±25M of 400M, against the v5
400M reference (Necto goal difference −29.4, Nexto −27.4, Necto goals for 2.94, touches 7.49,
kickoff goals against 6.42, retreat conceded 27.1%).

All four conditions must hold:

1. **Not clearly worse than its ancestor.** Pooled head to head at or above −2 standard errors of
   its own seeds.
2. **Clearly better against a fixed opponent.** Necto **or** Nexto goal difference better than the
   reference by more than twice the combined standard error — about **Necto better than −26.2** or
   **Nexto better than −23.0**. Necto and Nexto never train, so they are the only measure of the
   game itself; the head to head measures only whether a run beats its own ancestor, which v4, the
   learning-rate run and v6's last 100M all did while getting worse at football.
3. **No guardrail clearly worse** by the same test: Necto goals for, touches per min, kickoff goals
   against, retreat conceded.
4. **The mechanism actually fired.** Big pads clearly above v7's ~0.5/min. This is not a
   performance gate — it is a validity check. If boost collection did not move, the term did not do
   what it was built to do, and any change in the other numbers is something else wearing its name
   and must not be attributed to it.

Condition 4 is new in v8 and is there because of a specific failure mode this project has already
hit: v6's T6 was judged on downstream metrics without first checking whether it had changed the
behaviour it targeted.

## 6. Implementation

- `env/rewards_v8.py` — `REWARD_V8_DEFAULTS`, `RewardV8`, `RewardManagerV8`. T1–T5 are v3's own
  functions, imported from `env/rewards_v3.py` and hashed into v8's identity.
- `env/reward_registry.py` — `CODE_FILES["v8"]`, `make_reward_manager`, `reward_defaults`.
- `config/reward_versions/v8.json` — frozen settings, identity, and the replay pool fingerprint.
- `test_rewards_v8.py` — the ratchet's values, path independence, concavity, the collect/spend
  cycle paying every time, T1–T5 unchanged against v5, action independence (R2), and the registry.
  (19 tests; full suite 852 passing, 1 skipped.)

Rules kept from v3: no term reads the action (R2); every *potential* term pays
`w·(γ·φ(s′) − φ(s))` with φ(s′) = 0 on a goal step (R4); no gates (R6). T7 is not a potential, so
R4 does not apply to it, and `max(0, ·)` is the term's definition rather than a gate on some other
quantity.

## 7. Outcome

**Stopped at 226M, not adopted.** The term did what it was written to do and the metric it was
written to fix got worse. Successor: `docs/reward_v9_boost_spec.md`.

### Pooled at 200M (±25M, 14 checkpoints), against v7 at the same step count

| | v8 | v7 |
|---|---|---|
| big pads / min | **0.96 ± 0.08** | 0.63 ± 0.05 |
| boost collected / min | **216 ± 8** | 187 ± 5 |
| boost spent / min | **291 ± 6** | 242 ± 4 |
| small pads / min | 10.9 ± 0.2 | 11.7 ± 0.2 |
| mean tank | **9.1 ± 0.5** | 16.3 ± 0.7 |
| empty % | **63.2 ± 0.9** | 33.4 ± 1.2 |
| retreat starts low boost | **77.1 ± 1.6** | 46.9 ± 1.6 |
| boost spent supersonic % | 3.9 ± 0.35 | 0.4 ± 0.08 |
| touches / min | 5.92 ± 0.12 | 6.79 ± 0.10 |
| Necto goal difference | −30.6 ± 0.75 | −29.0 ± 0.66 |
| Nexto goal difference | −36.0 ± 1.35 | −33.1 ± 1.03 |
| Necto goals for / 10 min | 1.39 ± 0.22 | 2.18 ± 0.27 |
| on target / 100 touches | 3.26 ± 0.40 | 5.15 ± 0.52 |
| whiffs / 100 touches | 41.6 ± 1.9 | 33.6 ± 1.9 |
| head to head vs v5 reference | −0.79 ± 0.94 | −0.73 ± 1.11 |

### Trajectory (±15M)

| | 50M | 100M | 150M | 200M |
|---|---|---|---|---|
| touches / min | 5.91 | 5.92 | 5.90 | 5.86 |
| boost collected / min | 176 | 215 | 200 | 218 |
| boost spent / min | 247 | 310 | 287 | 287 |
| mean tank | 9.9 | 11.8 | 8.8 | 8.7 |
| empty % | 56.1 | 46.8 | 63.0 | 64.1 |
| Necto goal difference | −30.1 | −34.9 | −32.4 | −30.3 |

### Findings

1. **The mechanism fired and the target metric moved backwards.** Condition 4 passed — big pads
   up 50% — while `mean_boost_tank` halved and `retreat_starts_low_boost_pct`, the exact failure
   T7 was built to fix, ended 30 points *worse* than the version with no boost incentive at all.
   Collection rate is not evidence that a boost term worked; v9's condition 4 asks for the state.
2. **The cause is the ratchet's two halves reinforcing.** Paying the rise without charging the fall
   makes collect → dump → collect profitable every cycle, and the concave `sqrt` makes the refill
   from empty the largest single payment. Running dry was therefore the profitable state, and
   spend rate (291/min) overtook collection (216/min).
3. **The canary fired and was the wrong canary.** §4 named touches falling while collection climbed,
   and both happened — by 100M, before anyone looked. But it was written for a car wandering the
   map cycling pads, and small pads actually *fell*. Dumping boost is free and instant, so the
   exploit never required going anywhere. A canary has to be pointed at the quantity the term
   prices, not at a guess about how it will be abused.
4. **Discounting the fall would not have saved it.** `w·(max(0, Δφ) − k·max(0, −Δφ))` scales the
   exploit by `(1 − k)` without changing its shape, and at `k = 1` telescopes back into v5's inert
   potential. Any term that prices a change in one's own boost can be cycled.
5. **Entropy drifted, and only in this run.** −0.0066 nats/100M over 0–226M, the steepest in the
   lineage and the only run whose second half fell faster than its first (v6 recovered, v7 rose).
   The whole move is one episode between 100M and 180M (−0.0101 nats/100M) that coincides with
   `boost_empty_pct` stepping 47% → 63%; the last 46M are flat. The level never left the band
   earlier runs visited (v8 −0.396…−0.356, v4 reached −0.394), and the wave amplitude is
   unchanged at 0.0036 nats, so this is a policy narrowing onto a loop, not an entropy collapse.
   Timing only — not separable from ordinary late-run sharpening.
6. **The tag-mix change was never tested.** §3 flagged dropping v7's tag weighting as the one
   non-single-variable part of v8. The boost term dominated the result so completely that this
   bought no information either way. v9 keeps v8's natural frequencies, so the two runs remain
   comparable and the question is still open.
7. **The auto-eval watcher was silently skipping this run's checkpoints.** `evaluated_iterations`
   deduped on iteration number alone, and v8 restarts numbering from v5's final iteration, so v5's
   and v4's old results marked v8's fresh checkpoints as done — 133 of the first 158. Fixed in
   `scripts/auto_eval.py` by scoping the set to the active version. The same bug truncated v7's
   early evals to start at 13M. No result was ever wrong; results are named `<version>_<M>M`.
