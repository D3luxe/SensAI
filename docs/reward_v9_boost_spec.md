# Reward v9: boost as a state, not a transaction

Status: **written 2026-09-23, not yet started.**
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

Differencing against the opponent removes it structurally. The two cars' contributions sum to
**exactly zero on every step**, so stalling gains the pair nothing; whatever one banks, the other
pays. It also happens to be the quantity the replay study actually measured, and it prices boost
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

- **100M — the stall check.** This is the risk that replaces farming. The term is zero-sum so it
  *cannot* pay for a longer episode, but that is an argument, not a measurement. The canary is
  **total goals per 10 minutes, both teams** (`goals_for_per_10min + goals_against_per_10min`,
  ~34.4 in v7 and v8). If it falls clearly below 30 while the boost metrics improve, episodes are
  being extended and the argument is wrong somewhere — stop the run.
- **100M — the mechanism check, early.** `mean_boost_tank` should already be moving toward v7's
  16.3 and `zero_boost_pct` away from v8's 63%. v8's boost damage was visible by 150M and never
  reversed; there is no reason to spend 300M steps finding out.
- **150M — the passivity check**, as in v7 and v8: stop and review if the pooled head to head is
  clearly negative (below −2 standard errors) or a guardrail is clearly broken. Judged against
  **v5 at the same step count**, not its 400M numbers.
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
4. **The mechanism actually fired, in the state and not the transaction.** `mean_boost_tank`
   clearly above **v7's 16.3** and `zero_boost_pct` clearly below **v7's 33.4%**. Note what this
   deliberately does *not* ask for: big-pad rate. v8 raised big pads by 50% and still halved the
   tank, so collection rate is no longer accepted as evidence that the term worked.

Condition 4 is inherited from v8 and re-pointed. v8's version of it passed — big pads did rise —
while the run was failing, because it measured the thing the reward paid for instead of the thing
the reward was for.

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
