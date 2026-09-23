# Reward v9: boost as a state, not a transaction

Status: **running since 2026-09-23.** §2–§4 amended at ~20M steps, before any decision point; see
§4, "Amendments".
Identity: code `2005bf0f49098e34`, settings `e4d83141033d39de` (`config/reward_versions/v9.json`).
Start checkpoint: `checkpoints/baselines/v5_iter222000.pt` — lineage stays at v5, since none of v6,
v7 or v8 was adopted.

## 1. Why

v8 was stopped at 226M. It is the first version in this project that failed for a reason we can
state exactly, and the reason is not that the idea was wrong — it is that the term priced the wrong
object.

The replay study behind v8 (§2 of `reward_v8_boost_spec.md`) measured **boost differential**: a
*state*, how much boost a player is holding relative to the opponent, AUC 0.618 and +0.212 per
standard deviation in a joint fit against every other candidate feature. v8 translated that finding
into a reward for **acquiring** boost. The policy did exactly what it was asked, and the target
metric moved backwards:

| pooled at 200M, ±25M | v8 | v7 |
|---|---|---|
| boost collected / min | **216 ± 8** | 187 ± 5 |
| boost spent / min | **291 ± 6** | 242 ± 4 |
| big pads / min | **0.96 ± 0.08** | 0.63 ± 0.05 |
| mean tank | **9.1 ± 0.5** | 16.3 ± 0.7 |
| empty % | **63.2 ± 0.9** | 33.4 ± 1.2 |
| retreat starts low boost | **77.1 ± 1.6** | 46.9 ± 1.6 |
| touches / min | 5.92 ± 0.12 | 6.79 ± 0.10 |
| Necto goal difference | −30.6 ± 0.75 | −29.0 ± 0.66 |
| Nexto goal difference | −36.0 ± 1.35 | −33.1 ± 1.03 |

The transaction we paid for went up. The state we cared about went down by half, and
`retreat_starts_low_boost_pct` — the precise failure T7 was built to fix — ended **30 points worse**
than the version that had no boost incentive at all.

**Why.** T7 paid the rise in `sqrt(b/100)` and never charged the fall, so collect → dump → collect
pays every cycle. Because `sqrt` is concave, the largest payment is the refill from empty, so the
most profitable place to operate is with the tank dry. Both halves of the design pushed toward the
same behaviour and the policy committed to it: `boost_empty_pct` stepped from 47% to 63% over
100–180M, and policy entropy fell −0.0101 nats/100M over the same window (the steepest in the
lineage) before flattening — a policy converging onto a reliable loop.

**The canary fired and was still the wrong canary.** v8's §4 named `touches_per_min` falling while
`boost_collected_per_min` climbed, and both happened. But it was written for a bot wandering the
map cycling pads, and small-pad pickups actually *fell* (10.9 vs 11.7). It never needed to wander:
dumping boost is free and instant, so any pickup pays full concave value without going anywhere.

### What was rejected

**Discounting the fall**, `w · (max(0, Δφ) − k · max(0, −Δφ))` for `k < 1`, was the obvious repair
and does not work. It scales the exploit by `(1 − k)` without changing its shape — a cycle still
pays in proportion to the φ-amplitude traversed, and under a concave φ that amplitude is still
largest at the bottom of the tank, so the premium on running empty survives at lower gain. And it
cannot be tuned out: at `k = 1` it telescopes into a potential and goes inert, which is v5.

The general lesson, and it is the one worth carrying forward: **any term that prices a change in
one's own boost can be cycled.** The only way out is to stop pricing the change.

## 2. The change

**T7 is retired and T8 boost_edge takes its place** — the same `sqrt`, read as a level and
differenced against the opponent, paid on every step:

```
T8  boost_edge = boost_edge_weight * ( sqrt(b_self/100) - sqrt(b_opp/100) )
```

`boost_gain_weight` goes to 0.0 and `boost_edge_weight` to 0.01. Every other weight is v8's, which
is v5's. This is one behavioural change: the form of the boost term.

Three properties follow, each answering one of v8's failures:

- **No cycle pays.** The term reads the level, so any trajectory of one's own boost that returns to
  where it started has earned exactly nothing. There is no collect/dump loop to find.
- **Concavity now says the right thing.** The first 30 boost is worth more than the last 30, so the
  bot is paid most for *not being empty* — a level, not a refill. Running dry is now the punished
  state rather than the profitable one.
- **It is not a potential, so it can move the policy.** The episode sum of `w·(φ_self − φ_opp)`
  depends on the whole trajectory, not on the endpoints, which is what v5's T5 could never do.

### Why a differential rather than the bare level

A dense, always-on, non-negative bonus is a **survival bonus**: it pays for extending the episode,
and this bot's episodes end on goals, so it would pay for not scoring and not conceding. That is
the single most dangerous failure mode available to a per-step term, and it is exactly the kind of
thing that would show up 200M steps late as "passivity" with no obvious cause.

Differencing against the opponent removes that at the level of the pair: the two cars'
contributions sum to **exactly zero on every step**, so stalling gains the pair nothing; whatever
one banks, the other pays.

*(Amended at ~20M.)* That does not remove the incentive for either car, which is what PPO
optimises. Each car maximises its own return, and the car ahead on boost is still paid for every
step the episode runs; the other car losing the same amount does not change that. And with
`self_play_ratio` at 0.5, half of training is against an opponent that is not learning, so there is
no second learner to cancel against. **What actually rules stalling out is size**: a realistic edge
pays ~0.002 a step, so delaying a goal by a full second is worth ~0.03 against 10 for scoring it.
The stall canary in §3 stays, because size is still an argument, not a measurement.

The differential also happens to be the quantity the replay study actually measured, and it prices boost
starving — denying the opponent a pad is worth as much as taking one — which is a real 1v1 skill
that none of v2–v8 has ever been paid for.

### Weight sizing

T8 is paid **every step**, unlike T3 (per touch) and T1 (per goal), so its total scales with the
step rate: `tick_skip` 8 at 120 Hz is 15 steps/s, and at current goal rates (≈34.4 goals per 10
minutes across both teams) an episode lasts about 17.4 s, or ~261 steps.

- An **entire episode** held at a full tank against an empty opponent: `261 × 0.01 = 2.61`, about a
  quarter of a goal and ~40% of the average episode's goal term (−6.4).
- A **realistic sustained edge** of 0.2 in φ: ~0.52 per episode — the same order as T3's ~0.37
  (1.8 touches/episode at ~0.2 each), which is the only dense term that has ever moved this policy.

That is the calibration: visible next to the term that works, an order below the goal term, and
bounded at the weight per step no matter what. **This weight is only meaningful at `tick_skip` 8;**
`test_rewards_v9.py` pins the config at 8 so a change there cannot silently rescale the reward.

### Everything else is held

Scenario mix, gamma, replay pruning, mirroring, natural tag frequencies and the frozen pool
(sha `8fb61d856a48bc4c`) are v8's, byte for byte — `test_rewards_v9.py` asserts the only settings
that differ between v8 and v9 are the two boost weights. Unlike v8, this version **is** strictly
single-variable against its predecessor.

## 3. The run

Start from `checkpoints/baselines/v5_iter222000.pt`. Every checkpoint is evaluated automatically
(`scripts/auto_eval.py`); decision points pool every checkpoint within ±25M (~20 checkpoints, ~60
seeds), per `ui/eval_pooling.py`.

**Decision points:**

- **100M — the stall check.** This is the risk that replaces farming. The term is too small per
  step to pay for a longer episode (§2), but that is an argument, not a measurement. The canary is
  **total goals per 10 minutes, both teams** (`goals_for_per_10min + goals_against_per_10min`,
  ~34.4 in v7 and v8). If it falls clearly below 30 while the boost metrics improve, episodes are
  being extended and the argument is wrong somewhere — stop the run.
- **100M — the mechanism check, early.** The eval's `boost_mean` against Necto should be clearly
  above v7's and v8's at the same step count (12.0 and 11.8), and `boost_empty_pct` clearly below
  (43.3% and 46.8%). v8's boost damage was visible by 150M and never reversed; there is no reason
  to spend 300M steps finding out. These are eval fields, not the trainer's
  `mean_boost_tank` / `zero_boost_pct` telemetry (see condition 4).
- **100M and 150M — the hoarding check.** This is the canary pointed at what T8 prices. T8 pays
  for the *level*, so the way to abuse it is to hold boost and not use it: the tank rises and
  spending falls. Watch `boost_spent_per_min` against Necto next to `boost_mean`. If the tank is up
  while spending is clearly below v7's at the same step count (258.6 at 100M, 243.2 at 150M, by
  more than twice the combined standard error), and touches or Necto goals for are also down, the
  term is paying for boost the bot does not use: stop and review. Report
  `retreat_starts_low_boost_pct` beside it (v7: 75.6, 52.0, 46.9 at 100, 150, 200M) as the check
  that the boost is where it is needed.
- **150M — the passivity check**, as in v7 and v8: stop and review if the pooled head to head is
  clearly negative (below −2 standard errors) or a guardrail is clearly broken. Head to head is
  against the v5 reference checkpoint as before. **Guardrails are judged against v7 at 150M**, the
  same starting checkpoint pooled at the same density (14 results), with v6 at 150M as a second
  reading (only 3 results). Not against v5 at 150M: v5's run began from an earlier checkpoint, so
  v5 at 150M (Necto −34.8) is weaker than v9 was at step 0.
- **400M — adoption**, on the criterion below.

`touches_per_min` is retained as a general guardrail (condition 3), but it is no longer the
headline canary: v8 showed it registers the damage without identifying it.

## 4. The 400M criterion

Written before the run starts. Pooled over every checkpoint within ±25M of 400M, against the v5
400M reference (Necto goal difference −29.4, Nexto −27.4, Necto goals for 2.94, touches 7.49,
kickoff goals against 6.42, retreat conceded 27.1%).

All four conditions must hold:

1. **Not clearly worse than its ancestor.** Pooled head to head at or above −2 standard errors of
   its own seeds.
2. **Clearly better against a fixed opponent.** Necto **or** Nexto goal difference better than the
   reference by more than twice the combined standard error — about **Necto better than −26.2** or
   **Nexto better than −23.0**. Necto and Nexto never train, so they are the only measure of the
   game itself.
3. **No guardrail clearly worse** by the same test: Necto goals for, touches per min, kickoff goals
   against, retreat conceded.
4. **The mechanism actually fired, in the state and not the transaction.** The eval's
   `boost_mean` against Necto clearly above **16.3** and `boost_empty_pct` clearly below
   **33.4%**. Those are v7's values pooled at 200M, its best stretch for boost; v7 at 400M was
   14.4 and 36.5%, so this bar is deliberately a little stricter than same-step. Note what this
   deliberately does *not* ask for: big-pad rate. v8 raised big pads by 50% and still halved the
   tank, so collection rate is no longer accepted as evidence that the term worked.

Condition 4 is inherited from v8 and re-pointed. v8's version of it passed — big pads did rise —
while the run was failing, because it measured the thing the reward paid for instead of the thing
the reward was for.

**Condition 4 is necessary, not sufficient.** It is still the quantity T8 pays for, so a bot that
hoards boost passes it (§3, the hoarding check). It establishes that the term moved the state it
targets; condition 2 decides whether that was worth anything. If condition 4 passes and
`boost_spent_per_min` is clearly below v7's, a pass on condition 2 is still valid but the write-up
must not attribute it to better boost use.

### Amendments (2026-09-23, ~20M steps, before any decision point)

No threshold was set or changed from v9's data. The changes:

- **Condition 4's fields.** It named `mean_boost_tank` and `zero_boost_pct`, which exist only in
  the trainer's telemetry (`agent/ppo.py`), while its thresholds are eval numbers (v7 against
  Necto, pooled at 200M). The two measure different games: at ~20M the telemetry read tank 30.9 and
  empty 25.9%, which would have passed condition 4 on day one, while the eval read 14.7 and 44.4%.
  It now names the eval fields.
- **The hoarding check** (§3), which applies v8's lesson 2 (point the canary at the quantity the
  term prices) to v9, and the paragraph above, which applies lesson 3 (do not accept the paid-for
  metric as evidence) to condition 4.
- **The stall argument** (§2): the protection is the term's size, not its zero sum.
- **The 150M guardrail comparison** moves from v5 at 150M to v7 at 150M (§3), since v9 starts from
  v5's final checkpoint. `reward_v6_align_spec.md` §6 (correction) also shows v6 is effectively
  v5's reward trained for longer, and it did not reach condition 2's −26.2 against Necto, so a
  condition-2 pass here cannot be put down to extra training alone.

## 5. Implementation

- `env/rewards_v9.py` — `REWARD_V9_DEFAULTS`, `boost_edge`, `RewardV9`, `RewardManagerV9`. T1–T5
  are v3's own functions, imported from `env/rewards_v3.py` and hashed into v9's identity. T7's
  machinery is kept at weight 0 so the breakdown stays comparable across v8 and v9.
- `env/reward_registry.py` — `CODE_FILES["v9"]`, `make_reward_manager`, `reward_defaults`.
- `config/reward_versions/v9.json` — frozen settings, identity, and the replay pool fingerprint.
- `test_rewards_v9.py` — T8's value, zero-sum-per-step, that no boost cycle pays, that running
  empty is punished, concavity, the goal step, the no-opponent fallback, statelessness, T7 retired,
  equivalence with v8 and with v5 when the boost terms are switched back, action independence (R2),
  the registry, single-variable-against-v8, and the `tick_skip` calibration pin. (22 tests.)

Rules kept from v3: no term reads the action (R2); every *potential* term pays
`w·(γ·φ(s′) − φ(s))` with φ(s′) = 0 on a goal step (R4); no gates (R6). T8 is not a potential, so
R4 does not apply to it and it is paid on the goal step like any other; it is a plain continuous
function of the state, so there is nothing in it to gate.

## 6. Outcome

*Filled in when the run closes.*
