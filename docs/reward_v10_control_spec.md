# Reward v10: v7 at the pool's natural tag frequencies (a control)

Status: **closed at ~523M, not adopted, 2026-09-24** — outcome in §7.
Identity: code `6f7bbc1ec86ffaae` (v7's, unchanged), settings `c51290dcd68a4a89`
(`config/reward_versions/v10.json`).
Start checkpoint: `checkpoints/baselines/v5_iter222000.pt`. The league is restored to where v5 ended
(§3). Lineage stays at v5, since none of v6–v9 was adopted.

## 1. Why a control

Every run since v5 has been judged against references it did not share conditions with, and today
two misreadings came directly from that (`docs/reward_v9_boost_spec.md` §6, findings 6 and 7):

- **The start mix.** v7 forced 67.5% of replay starts into contested situations (challenge,
  goal_threat, aerial) and cut open play from 41.7% to 20%. v8 and v9 went back to the pool's own
  frequencies, and were then read against v7. At 50M this made v9's spending and touches look low
  when they matched v8's. **No run has trained v5's reward on the natural mix**, so neither v8 nor v9
  had a clean same-step reference.
- **The inherited league.** Archived checkpoints stay eligible for the league. The King (25% of
  environments) and elite pool (25%) are half of training, and each run began against the previous
  run's checkpoints: v9 opened against v8's collect-and-dump bot.
- **v7's open question.** v7 is the only run in the lineage that got worse in its second half
  (retreats reaching goal-side 95% → 83%, kickoff goals against 6.4 → 10.6, touches 7.5 → 6.3), with
  no training pathology. v7 §8 finding 4 named the tag weighting as the prime suspect. v8 changed
  the mix and a boost term together, and the boost term swamped the result, so the question was
  never answered.

The reward is not the variable here. `docs/reward_v6_align_spec.md` §6 (correction) shows v6 was
effectively v5's reward trained for longer and stayed at −27.7 to −28.6 against Necto. v10 is not
expected to clear the adoption bar. It is run for what it measures.

## 2. The change

**v10 is v7 with `replay_sampling.tag_weights` set to the pool's own frequencies**, the same weights
v8 and v9 used. Nothing else differs from v7: v5's reward (T1 goal, T2 ball_position, T3 touch, T5
boost at weight 1, gamma 0.9977), v7's scenario mix (`kickoff_prob` 0.24, `replay_prob` 0.36),
pruning, mirroring, and the frozen pool (sha `8fb61d856a48bc4c`).

| tag | pool share | v7 forced | v10 |
|---|---|---|---|
| open_play | 41.7% | 20.0% | 41.7% |
| aerial | 22.9% | 17.5% | 22.9% |
| challenge | 19.6% | 30.0% | 19.6% |
| goal_threat | 13.2% | 20.0% | 13.2% |
| carry | 2.1% | 7.5% | 2.1% |
| carry_contested | 0.5% | 5.0% | 0.5% |

`test_rewards_v10.py` asserts that the tag weights are the only settings that differ from v7, that
they equal the pool's frequencies, and that the code and reward manager are v5's and v7's.

**Against v7 the reward and starts differ in one variable only, the tag weights.** The league does
not match: v7 began against the league as v6 left it, and v10 begins against v5's (§3). Both are
v5-reward policies (v6's T6 was inert), so this is the smallest league difference available, but it
is a difference.

## 3. Starting conditions

- **Start checkpoint:** `checkpoints/latest_model.pt` is a copy of `checkpoints/baselines/v5_iter222000.pt`
  (the trainer resumes from `latest_model.pt`). v9's last model is archived as
  `checkpoints/archive/v9_run/latest_model.pt`.
- **League restored to v5's end.** `scripts/restore_league.py` installed the league files
  `archive_run.py` saved just before v5's run was archived (`*.before_v5_run_202609211333`),
  repointed into `checkpoints/archive/v5_run/`. That gives 133 ratings, every rated checkpoint
  present, King `v5/checkpoint_iter_222200`, and an elite pool of ten v5 checkpoints (213200–222400).
  At 128 environments the split is self-play 56, King 32, Necto 24, pool 16. The league v9 left
  behind is kept as `*.before_restore_202609231458`.
- **Checkpoints:** `checkpoints/` holds no numbered checkpoints; v9's are in `checkpoints/archive/v9_run/`.
- **Auto-eval watcher:** it reads the active version once at startup, so it must be restarted
  after `reward_version` changes. Otherwise it treats v10's checkpoints as another run's and skips
  them.

**For every later version:** archive the run with `archive_run.py`, then restore the league from
the lineage start's snapshot with `restore_league.py`, so each run begins against the same league.

## 4. How v10 is read

Every checkpoint is evaluated automatically; decision points pool every result within ±25M.

**This is a control, so it runs to 400M unless training breaks.** A control stopped early loses
most of its value. Stop only for a training pathology (entropy collapse, explained variance
falling away) or a guardrail clearly broken against **both** v7 and the v5 reference at the same
step. A difference from v7 is the result, not a reason to stop.

**Reference: v7 at the same step count** (pooled ±25M):

| step | h2h vs v5 ref | Necto GD | Nexto GD | touches | kickoff GA | retreat conceded | retreat goal-side | retreat boost used |
|---|---|---|---|---|---|---|---|---|
| 100M | +0.14 ± 0.90 | −29.0 ± 1.5 | −31.5 ± 1.0 | 6.40 | 8.26 | 31.8% | 94.3% | 22.1 |
| 150M | +2.79 ± 1.01 | −29.2 ± 1.3 | −31.3 ± 1.0 | 6.91 | 8.62 | 36.9% | 89.3% | 20.1 |
| 200M | −0.73 ± 1.11 | −29.0 ± 0.9 | −33.1 ± 1.0 | 6.79 | 8.36 | 41.7% | 95.5% | 20.2 |
| 250M | +0.42 ± 0.87 | −29.8 ± 1.7 | −32.2 ± 0.7 | 6.54 | 11.42 | 47.6% | 94.4% | 18.2 |
| 300M | −0.22 ± 1.02 | −32.6 ± 1.1 | −31.9 ± 1.0 | 6.60 | 10.50 | 27.1% | 89.8% | 26.5 |
| 350M | +1.42 ± 1.03 | −35.4 ± 1.4 | −36.3 ± 1.2 | 6.20 | 13.61 | 23.6% | 82.3% | 25.7 |
| 400M | −0.36 ± 0.89 | −31.9 ± 2.3 | −32.0 ± 1.7 | 6.34 | 10.63 | 22.7% | 83.3% | 21.7 |

v5 400M reference: Necto −29.4, Nexto −27.4, touches 7.49, kickoff goals against 6.42, retreat
conceded 27.1%, retreat goal-side 96.5%.

**What each outcome would mean, written before the run:**

- **The tag mix caused v7's regression** if v10 holds through 250–400M where v7 fell: retreat
  goal-side at or near 95%, kickoff goals against near 6.4, and Necto goal difference not sliding
  past −32. Then the natural mix is the lineage default and v7's weighting is retired.
- **The tag mix is cleared** if v10 regresses the same way over the same steps. The cause then lies
  in something v7 and v10 share: pruning, mirroring, the raised `kickoff_prob`, or the replay share
  itself. The next control varies one of those.
- **Either way**, v10's pooled numbers become the same-step reference for any later version that
  trains on natural-frequency starts, and they replace v7 in that role.

**Always report**, at 100M, 150M, 200M and 400M, the context metrics v9 showed matter:
`scenario_retreat.boost_used`, `scenario_retreat.retreat_boosting_pct`, `reached_goalside_pct` and
`time_to_goalside_s`, alongside the guardrails. Read Nexto ahead of Necto: Necto is a training
opponent (18.75% of environments), and Nexto is the only held-out one.

## 5. The 400M criterion

v10 can still be adopted, on v9's criterion without the mechanism condition. Pooled ±25M around
400M, against the v5 400M reference:

1. **Not clearly worse than its ancestor:** head to head at or above −2 standard errors.
2. **Clearly better against a fixed opponent:** Necto better than **−26.2** or Nexto better than
   **−23.0**.
3. **No guardrail clearly worse:** Necto goals for, touches per min, kickoff goals against, retreat
   conceded.

## 6. Implementation

- `config/reward_versions/v10.json`: v7's snapshot with v9's tag weights; identity frozen.
- `env/reward_registry.py`: `CODE_FILES["v10"]` is v7's files; `reward_defaults` and
  `make_reward_manager` treat v10 as v5 and v7 (`RewardManagerV3`).
- `config/default_config.yaml`: `reward_version: v10`.
- `scripts/restore_league.py`: installs a saved league snapshot, repointing its bare checkpoint
  paths into the run's archive; backs up the current league first; refuses while training is live
  or if any rated checkpoint is missing. Tests in `test_restore_league.py` (6).
- `test_rewards_v10.py`: only the tag weights differ from v7; they are the pool's natural
  frequencies; reward and gamma are v5's; code is v7's; same frozen pool; v5's manager; v5 start
  checkpoint. (7 tests.)

## 7. Outcome

**Stopped at ~523M (iteration 253947; last checkpoint 253800), not adopted, 2026-09-24.** v5 remains
the lineage baseline. The 159 checkpoints are in `checkpoints/archive/v10_run/`, every one
evaluated. The league has been restored to v5's end again (§3), and the league v10 left is kept
as `*.before_restore_202609240948`.

### The 400M criterion

Pooled 375–425M (15 results) against the v5 400M reference:

| condition | v10 | needed | |
|---|---|---|---|
| 1. head to head vs v5 | −1.68 ± 0.86 (−1.95 SE) | at or above −2 SE | pass, barely |
| 2. Necto / Nexto goal difference | −29.0 ± 1.3 / **−33.6 ± 0.5** | better than −26.2 or −23.0 | **FAIL** |
| 3. touches / min | **5.93 ± 0.19** vs 7.39 | not clearly worse | **FAIL** (z −6.3) |
| 3. retreat conceded | **39.2% ± 2.2** vs 25.0% | not clearly worse | **FAIL** (z +3.5) |
| 3. kickoff goals against / Necto goals for | 8.9 / 3.6 vs 6.5 / 2.8 | not clearly worse | hold |

### The run against v7, 50M windows

| | Necto GD v10 / v7 | Nexto GD v10 / v7 | retreat goal-side v10 / v7 | kickoff GA v10 / v7 | touches v10 / v7 | h2h v10 / v7 |
|---|---|---|---|---|---|---|
| 0–50M | −27.8 / −29.9 | −31.2 / −28.6 | 100.0 / 97.3 | 7.5 / 9.0 | 6.8 / 7.2 | −0.5 / −5.9 |
| 100–150M | −29.2 / −29.7 | −31.8 / −31.2 | 92.5 / 91.7 | 7.9 / 7.6 | 6.5 / 6.4 | +1.9 / +3.0 |
| 200–250M | −27.7 / −27.7 | −35.3 / −32.9 | 99.2 / 94.2 | 6.6 / 10.2 | 5.9 / 6.7 | −1.5 / −0.5 |
| 250–300M | −30.1 / **−33.3** | −33.0 / −32.4 | 97.2 / 93.8 | 11.3 / 11.3 | 6.9 / 6.6 | −1.2 / −0.3 |
| 300–350M | −29.5 / **−32.2** | −33.5 / −33.5 | 97.5 / **85.4** | 10.6 / **11.4** | 6.5 / 6.5 | −3.9 / +0.4 |
| 350–400M | −27.0 / **−35.3** | −32.7 / −35.6 | 90.4 / **83.3** | 8.8 / **13.9** | 6.1 / 6.2 | −1.6 / +1.5 |
| 400–450M | −27.8 / — | −33.1 / — | 94.4 / — | 8.9 / — | 5.9 / — | −3.1 / — |
| 450–500M | −27.3 / — | −33.5 / — | 96.1 / — | 7.3 / — | 6.2 / — | −5.1 / — |

Training was healthy throughout: entropy −0.364 to −0.372, explained variance 0.81–0.82, value
loss 0.070–0.084. Mean training reward slid from 0.71 (150–200M) to 0.63–0.65 (400–550M).

### Findings

1. **The tag mix caused v7's regression, the part that was v7's alone.** Over 250–400M, v7 slid
   to −32 to −35 against Necto, 83–85% retreat goal-side and 11–14 kickoff goals against. v10 held
   −27 to −30, 90–97% and 7–11 over the same steps. §4's first outcome held: the natural mix is the
   lineage default, and v7's forced weighting is retired.
2. **v10 did not improve either.** Nexto, the held-out opponent, stayed at −31 to −35 for all
   523M, against the reference's −27. Touches fell from 6.8 to about 6.0 (reference 7.4). First
   touch fell on bounces (33% → 21–28%) and, after 400M, on dropped balls (17–24% → 7.5–10%). The
   head to head against v5 went steadily negative: −3.9, −1.6, −3.1 and −5.1 over 300–500M. After
   500M steps v10 loses to the checkpoint it started from.
3. **Three continuations of v5's reward, one plateau.** v6 (T6 inert, so v5's optimum), v7
   (forced starts) and v10 (natural starts) have trained v5's reward for ~1.4B steps past v5's
   end. None beat v5 against a fixed opponent, and v7 and v10 both lost touches and shot quality.
   With v8's and v9's reward terms, nothing tried since v5 has moved the plateau. The start mix
   explains why v7 got *worse* than the others, not why none got *better*.
4. **A structural cause has since been measured: the steer and throttle pre-activations sit far
   past full lock.** First measured 2026-09-23 in a separate session (project memory
   `tanh-mean-saturation`). Reproduced 2026-09-24 with `scripts/action_saturation.py` (~4–6k
   grounded states per checkpoint from two 3000-step matches against Necto; results in
   `logs/saturation_baseline_20260924.json`):

   | | pretrained | v3 198k | v5 222000 | v10 229000 | v10 253800 |
   |---|---|---|---|---|---|
   | median d(mean)/d(pre), steer / throttle | 0.96 / 0.89 | 0.057 / 0.043 | 0.031 / 0.019 | 0.038 / 0.051 | 0.024 / 0.083 |
   | median \|pre\|, steer / throttle | 0.18 / 0.33 | 4.3 / 3.6 | 4.3 / 4.0 | 4.6 / 3.5 | 4.7 / 2.9 |
   | each pass past 0.95, steer / throttle | 1% / 19% | 70% / 72% | 75% / 77% | 72% / 71% | 72% / 65% |
   | averaged mean past 0.95, steer / throttle | 0.5% / 19% | 20% / 54% | 32% / 63% | 28% / 55% | 35% / 46% |
   | braking (throttle mean < −0.1) | 29.5% | 0.8% | 1.3% | 2.0% | 5.0% |

   The action mean is `0.5 · (tanh(raw pass) + tanh(mirrored pass) · sign)` (`agent/models.py:525`),
   and full lock (0.95) starts at a pre-activation of 1.83. The first session's "76–85%" was the
   per-pass figure. The averaged mean reads lower because of **steer cancellation**: in 24–28% of
   grounded states both passes sit at full lock in *opposite* directions, so steer reads ~0 (median
   0.002) as +1 and −1 cancelling, with no gradient. A small observation change that pulls either
   pass off its rail swings steer by up to 0.5, which is a direct mechanism for the steer dither.
   Throttle's log_std floor is −2.5, sigma 0.082 (`agent/models.py:58`), too narrow to explore
   braking. That is a direct mechanism for findings 2 and 3: further training cannot change ground
   control in most ground states, and the regressing metrics (touches, whiffs, first touch on drops)
   are ground control. It predates every version in this lineage, which is why no reward or start
   change moved it; 500M steps of v10 moved throttle a little (63% → 46%, braking 1.3% → 5%) and
   steer not at all.
5. **The next version should target the action head, not the reward or the starts.** Reward
   and starts stay at v10's, which is now the natural-mix reference. Candidates from the
   saturation finding: a penalty on pre-tanh magnitude, a clip-aware log-probability, or
   tanh-squashed sampling. The first of these leaves the parameterisation alone and is the smallest
   single variable, and it has to act on each pass: a clip-aware log-probability would not undo
   opposed cancellation, where the averaged mean is already at 0. Re-measure saturation before claiming any fix. The metric has to be the
   saturation itself, plus touches and first touch; per v8's lesson 3, not a quantity the change
   pays for.

