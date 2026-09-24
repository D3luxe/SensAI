# v11: v10 plus a pre-tanh magnitude penalty on throttle and steer

Status: **running since 2026-09-24.**
Identity: code `017162ed5368186c`, settings `a73964846d2f2ae6` (`config/reward_versions/v11.json`).
Start checkpoint: `checkpoints/baselines/v5_iter222000.pt`, with the league restored to where v5
ended. Lineage stays at v5, since none of v6–v10 was adopted.

## 1. Why

v6, v7 and v10 trained v5's reward for ~1.4B steps past v5's end, and none beat v5 against a fixed
opponent. v8 and v9 changed the reward, and v7 and v10 changed the starts, without moving that
plateau (`docs/reward_v10_control_spec.md` §7). The cause found since is in the action head, not the
reward.

The action mean is `0.5 · (tanh(raw pass) + tanh(mirrored pass) · sign)`, where each pass is
`actor_mean(features)`, and PPO's Normal is centred on it. Measured with
`scripts/action_saturation.py` (two 3000-step matches against Necto per checkpoint, SensAI's
grounded states only):

| | pretrained | v3 198k | v5 222000 | v10 229000 | v10 253800 |
|---|---|---|---|---|---|
| median d(mean)/d(pre), steer / throttle | 0.96 / 0.89 | 0.057 / 0.043 | 0.031 / 0.019 | 0.038 / 0.051 | 0.024 / 0.083 |
| median \|pre\|, steer / throttle | 0.18 / 0.33 | 4.3 / 3.6 | 4.3 / 4.0 | 4.6 / 3.5 | 4.7 / 2.9 |
| braking (throttle mean < −0.1) | 29.5% | 0.8% | 1.3% | 2.0% | 5.0% |
| steer passes at opposite rails | 0.0% | | 27.8% | | 24.2% |

Full lock (0.95) starts at a pre-activation of 1.83, and every RL checkpoint sits at 3–5. At 4.5 the
tanh gradient is 0.0005, so in most ground states the policy gradient cannot move throttle or
steer, and throttle's exploration (sigma 0.082, its floor) is too narrow to find braking. In a
quarter of grounded states steer is +1 and −1 cancelling to ~0, also with no gradient, so a small
observation change can swing it by up to 0.5. That is a mechanism for the steer dither.

The metrics that regressed in v7 and v10 are ground control: touches, whiffs, first touch on
dropped and bouncing balls. More training cannot fix a channel with no gradient, which is why more
training did not.

## 2. The change

**v11 is v10 plus one term in PPO's loss:**

```
penalty = 0.001 * mean( relu(|pre| - 2.0) ** 2 )
```

over the throttle and steer pre-activations (channels 0 and 1), both passes of the mirrored forward,
every state in the minibatch. Implemented in `agent/pre_tanh_penalty.py`, which reads `actor_mean`'s
outputs through a forward hook for the duration of the minibatch forward pass, so `agent/models.py`
is unchanged. Wired in `agent/ppo.py` and gated on the frozen `action_regularization` settings
section, which only v11 carries.

- **Zero inside the threshold.** It never pulls a mean toward 0; it only stops pre-activations
  drifting into the dead zone. At 2.0 the mean can still reach tanh(2) = 0.964, and the gradient at
  the boundary is 0.071, about 140× what a pre-activation of 4.5 gets. Where the advantage still
  wants full lock, the policy gradient and the penalty balance just beyond 2.0, which keeps the
  gradient alive.
- **Both passes, not the averaged mean.** Opposed cancellation is two passes at opposite rails. A
  term on the averaged mean (or a clip-aware log-probability) sees a mean of 0 and does nothing.
  Pulling each pass back to ±2 leaves them in the region where the advantage can separate them.
- **Throttle and steer only.** Pitch, yaw and roll are masked to zero on the ground, and pitch
  saturation in the air is a separate question (flip discovery); leaving them out keeps this to
  one variable.
- **Every state, not only grounded ones.** The failure was measured on the ground, but the dead
  zone is the same wherever it occurs, and a grounded-only mask would add a second design choice.
- **Not during critic warmup**, so the actor stays untouched there as before.
- **Not a reward.** Returns and advantages are unchanged, and nothing reads the action the env
  receives (R2 is about reward terms).

**Sizing.** On v5 222000's own rollout states against Necto, 71.6% of throttle and steer
pre-activations exceed 2.0, by 3.05 on average (mean squared excess 23.4). At weight 0.001 the
penalty starts at ~0.023, about twice v10's mean |policy loss| (0.0125), and falls to zero as
pre-activations come back inside 2.0.

**Everything else is v10's:** v5's reward, v10's scenario mix and natural-frequency replay starts,
the frozen pool (sha `8fb61d856a48bc4c`), the v5 start checkpoint, and the league restored to v5's
end. `test_rewards_v11.py` asserts that the only settings difference from v10 is
`action_regularization`, and that no other version picks it up. **Against v10 this is a single
variable**, and v10 is the same-step reference throughout.

## 3. Starting conditions and what to expect

- `checkpoints/latest_model.pt` is a copy of `checkpoints/baselines/v5_iter222000.pt`; v10's last
  model is in `checkpoints/archive/v10_run/`.
- The league is v5's end, restored by `scripts/restore_league.py` (King `v5/checkpoint_iter_222200`,
  ten v5 checkpoints in the pool).
- **Restart the auto-eval watcher** after the version change; it reads the version once at startup.
- **Startup transient.** v5's pre-activations start at 3–5, so the penalty is largest at the start
  and sits outside PPO's clipped objective. Pulling a saturated throttle mean from ~1.0 to 0.964 is
  a small move in action space but a real one against sigma 0.082, so `approx_kl` and
  `clip_fraction` may run above v10's (0.012–0.013 and ~0.10) for the first tens of millions of
  steps. Watch that they settle. A KL that stays well above v10's after ~30M means the weight is
  too high, and that is a new version, not a mid-run edit.
- **Telemetry**, per iteration in `history.jsonl`: `pre_tanh_penalty`, `pre_tanh_over_pct` (the share
  of throttle/steer pre-activations over 2.0) and `pre_tanh_pre_abs_median_0` / `_1`.

## 4. Decision points

Judged against **v10 at the same step count** (pooled ±25M), the clean reference this version was
built on:

| step | h2h vs v5 ref | Necto GD | Nexto GD | touches | whiffs / 100 | first touch bounce / drop | time to first touch (bounce) | kickoff first touch | kickoff GA | retreat goal-side |
|---|---|---|---|---|---|---|---|---|---|---|
| 100M | +1.12 ± 1.15 | −30.7 ± 1.3 | −30.6 ± 1.1 | 6.84 | 46.3 | 28.9% / 9.4% | 1.41 s | 9.5% | 10.4 | 96.1% |
| 150M | −0.45 ± 1.24 | −29.9 ± 1.5 | −34.2 ± 1.0 | 6.08 | 56.5 | 28.3% / 13.9% | 1.36 s | 0.2% | 6.2 | 94.7% |
| 200M | −0.27 ± 0.68 | −29.3 ± 1.3 | −34.0 ± 1.4 | 5.73 | 52.6 | 27.8% / 11.7% | 1.43 s | 0.9% | 7.3 | 98.1% |
| 400M | −1.68 ± 0.86 | −29.0 ± 1.3 | −33.6 ± 0.5 | 5.93 | 38.2 | 27.8% / 17.2% | 1.44 s | 1.0% | 8.9 | 91.9% |

- **~25M: the mechanism, early.** Run `scripts/action_saturation.py` on the nearest checkpoint.
  Median |pre| on both channels should be under ~2.5 and falling, against 4.3 / 4.0 at the start.
  If `pre_tanh_over_pct` in the telemetry is not falling by then, the weight is too small: stop,
  and put the fix in a new version.
- **~25M and 100M: the side-effect canary, pointed at what the penalty costs.** The penalty caps
  the deterministic mean at about 0.96, so the price, if any, is speed to the ball. Watch time to
  first touch on bounces, kickoff first touch and Necto goals for against v10 at the same step. If
  they are clearly worse while the saturation readings improve, the cap is costing more than the
  gradient buys: stop and review. *(Noted at start, 2026-09-24:)* kickoff first touch is a weak
  canary. v10 sits at 1–9% against Necto (v5 king 18.9%), so it has little room to get clearly
  worse. The cause is known from play: SensAI drives and boosts on the kickoff but does not flip,
  so it arrives late. A flip is the jump button plus airborne pitch, neither of which this penalty
  touches, so v11 is not expected to change it. Read time to first touch and Necto goals for first.
- **100M: is the gradient being used?** Re-run the probe and read what the penalty does **not**
  pay for: braking share (v10 229000 at ~115M: 2.0%; pretrained 29.5%), steer opposed-cancellation
  share (v10: 24–28%), and `scripts/steering_jitter_probe.py`, alongside touches, whiffs and first
  touch against the table above.
- **150M: the passivity check.** Stop and review if the pooled head to head is below −2 standard
  errors or a guardrail is clearly worse than v10 at 150M.
- **400M: adoption**, on the criterion below.

## 5. The 400M criterion

Pooled ±25M around 400M, against the v5 400M reference (Necto −29.4, Nexto −27.4, Necto goals for
2.94, touches 7.49, kickoff goals against 6.42, retreat conceded 27.1%). All four must hold:

1. **Not clearly worse than its ancestor:** head to head at or above −2 standard errors.
2. **Clearly better against a fixed opponent:** Necto better than **−26.2** or Nexto better than
   **−23.0**. Nexto is the held-out opponent (Necto is 18.75% of training), so a pass on Nexto is the
   stronger one.
3. **No guardrail clearly worse:** Necto goals for, touches per min, kickoff goals against, retreat
   conceded.
4. **The mechanism fired, measured where the penalty does not pay.** Median |pre| on throttle and
   steer below 2.5 is what the penalty prices, so it is necessary and not sufficient (v8's lesson
   3). Also required: grounded braking share clearly above v10's (2.0–5.0%), and steer
   opposed-cancellation clearly below v10's (24–28%), both from `scripts/action_saturation.py` on
   checkpoints in the 400M window.

If conditions 1–3 fail while 4 holds, the action head was not the limit, or not the only one.
That result is worth as much as an adoption.

## 6. Implementation

- `agent/pre_tanh_penalty.py`: `PreTanhPenalty` (from_config, capture, loss, stats). Part of v11's
  code identity.
- `agent/ppo.py`: builds the penalty from `action_regularization`, captures `actor_mean` during the
  minibatch forward, adds the loss outside critic warmup, and logs the telemetry above. Inert for
  every other version.
- `env/reward_registry.py`: `action_regularization` joins `OPTIONAL_SETTINGS_SECTIONS` (hashed only
  when present, so earlier identities are untouched); `CODE_FILES["v11"]` is v10's files plus the
  penalty; v11 builds v5's reward manager.
- `config/reward_versions/v11.json`, `config/default_config.yaml` (`reward_version: v11`).
- `test_rewards_v11.py` (11): single variable against v10; no other version gets the penalty; zero
  inside the threshold; the squared excess; both passes counted; only the actor moves; the hook is
  released; a two-iteration trainer run with warmup off applies the penalty and logs its telemetry.

## 7. Checks during the run

### ~50M (2026-09-24, iteration 225000)

**Training.** No startup transient: approx_kl 0.010–0.012 and clip_fraction ~0.10, level with v10
from the first 5M; entropy and explained variance level too. In training minibatches the share
of throttle/steer pre-activations over 2.0 fell from 68% to 43%, and the medians from 2.84 / 4.16
(throttle / steer) to 1.75 / 1.62. The penalty fell from 0.013 to 0.0017.

**The cost canary fired.** Evals pooled 2–50M against v10 over the same window (14 and 15 results):

| | v11 | v10 | z |
|---|---|---|---|
| time to first touch, bounce | 1.47 s | 1.37 s | +3.0 |
| time to first touch, drop | 1.88 s | 1.34 s | +3.0 |
| retreat time to goal-side | 3.88 s | 3.45 s | +6.2 |
| first touch on drops | 14.3% | 21.9% | −3.1 |
| first touch on wall balls | **93.5%** | 65.8% | **+8.1** |
| Necto GD | −31.7 | −27.8 | −2.7 |
| Necto goals for | 1.6 | 2.8 | −1.7 |
| touches / min | 6.12 | 6.84 | −2.4 |
| h2h vs v5 | −3.75 ± 1.06 | −0.52 ± 0.77 | −2.5 |

Per checkpoint, the times grew as the pre-activations shrank (retreat 3.5 → 4.3 s, drop 1.2–1.8 →
2.1–3.3 s), while the head to head recovered from −8 to −13 (9–26M) to −3 to +4 (29–45M) and
touches recovered from 4.5 to 6.7.

**The review: the non-priced mechanism checks.** `scripts/action_saturation.py` (now reporting the
opposed share) and `scripts/steering_jitter_probe.py` on v5 222000, v10 225000 and v11 225000, all
at the same seeds. Raw results: `logs/saturation_v11_50M.json`, `logs/jitter_v11_50M.json`.

| | v5 king | v10 @ 50M | v11 @ 50M |
|---|---|---|---|
| median gradient, steer / throttle | 0.034 / 0.014 | 0.048 / 0.016 | **0.324 / 0.218** |
| median \|pre\|, steer / throttle | 4.37 / 3.86 | 4.08 / 4.06 | **1.59 / 1.89** |
| steer opposed (passes at opposite rails) | 27.4% | 18.8% | **6.2%** |
| steer at full lock | 28.6% | 35.1% | 14.2% |
| throttle at full forward | 65.9% | 63.5% | **27.3%** |
| braking | 1.2% | 1.6% | 2.8% |

| steer reversals (>0.3 swing), by distance to ball | v5 king | v10 @ 50M | v11 @ 50M |
|---|---|---|---|
| 0–400 uu | 14.7 ± 2.1% | 16.8 ± 2.5% | **9.0 ± 2.1%** |
| 400–800 | 14.1 ± 1.3% | 18.4 ± 1.9% | **9.7 ± 1.1%** |
| 800–1500 | 16.2 ± 1.8% | 18.8 ± 1.9% | **12.4 ± 1.0%** |
| 1500–3000 | 17.1 ± 1.5% | 17.7 ± 1.7% | **7.8 ± 1.2%** |
| 3000+ | 10.9 ± 2.1% | 21.1 ± 2.7% | **8.9 ± 1.6%** |

**Reading.** Steer is fixed as intended: 7–10× the gradient, cancellation down to a third, and
jitter roughly halved against v10 at every distance (−2.4σ to −4.7σ), below even the v5 king. That
fits the wall-ball first touches. Throttle has left the rail, but into part throttle, not braking:
full-forward share fell from 64% to 27%, braking rose only from 1.6% to 2.8% (pretrained 29.5%). The
throttle median (1.89) is below the threshold, so the penalty is not holding it there. Off the rail,
a small shortfall from full throttle barely changes the clipped sampled action, so the policy
gradient pushes it back weakly. The slower times fit this.

**Decision: continue to 100M** (the user's call). Throttle was as much of a pain point as steer in
play, so the run gets the chance to pick its speed back up with the gradient now available. At
100M: if time to the ball and Necto GD are still clearly worse than v10 at the same step, stop. The
fallback is a v12 with the same penalty on steer only.

## 8. Outcome

*Filled in when the run closes.*
