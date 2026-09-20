# Run v5 — v3's reward at a lower learning rate

Status: **running, 2026-09-20** — 250M steps in, progress in §4a. This is a **training run, not a reward version**.
The reward is v3's, unchanged and still frozen (`config/reward_versions/v3.json`, code
`9243332689bd95fa`, settings `646c9927726c7f82`); `config/default_config.yaml` goes back to
`reward_version: v3`. The one change is `learning_rate: 3e-4 → 1.5e-4`.

The rules of `docs/reward_v3_spec.md` §2 (R1–R7) apply. R1 (frozen per run) covers the reward;
this run additionally freezes the learning rate, so it is a single-variable experiment like the
reward runs before it.

## 1. Why the learning rate, and not another reward term

v4 is rejected (`docs/reward_v4_spec.md` §6) and left three findings. Two of them rule out more
reward work on the same problem: a permanent pull toward the ball has now failed twice, and paying
for the loose-ball race for 500M steps did not change the first-touch rate, so the deficit is
mechanical rather than motivational. The third points here:

| Evidence | Value | Expected |
|---|---|---|
| `approx_kl` per update, steady over 300M+ steps | **0.023** | ~0.01 |
| `clip_fraction` | **15%** | a few % |
| Head-to-head spread between adjacent checkpoints | **up to 25 goals / 10 min** | within eval noise (~3) |
| `explained_variance` | 0.84 | healthy, and it is |
| Mean reward, entropy, value loss across the same window | flat | — |

Each update moves the policy about twice as far as a PPO trust region intends, and neighbouring
checkpoints play very differently while every loss curve stays flat. That is churn, not learning:
the run keeps changing without going anywhere. The learning rate has been 3e-4 for the whole
project — 3.2B steps and three reward versions — and was never revisited as the bot improved.

Diagnosis order (R7): (a) sim and opponents are unchanged and correct; (b) the scenario mix is
v3's, which produced the king; so (c) — and the change is a hyperparameter, not a reward term.

## 2. The change

`hyperparameters.learning_rate: 1.5e-4` in `config/default_config.yaml`, and the same value in
`config/live_config.json`, which overrides the yaml at runtime. Halving is deliberate: enough to
show up in KL and in checkpoint-to-checkpoint spread, not so much that learning stalls. Everything
else is untouched — `ent_coef` 0.008, `clip_range` 0.2, batch 16384, 4 epochs, γ 0.995, the v3
reward and its scenario mix, the league ratios.

## 3. The run

- **Start:** `checkpoints/baselines/v3_iter198000.pt`, the v3 king, which is also what v4 started
  from. So this run and v4 are two branches from one point, differing in one variable each.
- **This continues the v3 run rather than starting a new one.** The start checkpoint is stamped v3
  and the config now names v3, so the trainer sees no version change: no return-normaliser reset,
  no fresh optimizer, no critic warm-up, and the anneal clocks restore from the checkpoint. The
  king is 399M steps into v3's clock, past the 300M closeness horizon, so T4 stays at 0 — the same
  reward the king was trained under for its last 100M steps. That is what we want: the learning
  rate is then the only thing that differs from the run that produced the king, with the critic
  already fitted to this reward.
- **Consequences of continuing that clock.** History records carry v3's run key, so the dashboard's
  "current run" curves include the original v3 run before this point (the KL panel is empty over
  that older stretch). Eval results must be named `--name lr15_<steps>M`: the automatic name would
  be `v3_<steps into v3's clock>M`, which collides with the original run's results.
- **Checkpoint numbering** restarts from 198000 and would overwrite v4's files, so the v4 run's
  checkpoints move to `checkpoints/archive/v4_run/` first, with league state and TrueSkill ratings
  repointed (as was done for v3's).
- **Nothing is changed during the run.**

## 3a. False start (2026-09-20)

The first attempt trained ~1,000 iterations (16M steps) at **3e-4**, not 1.5e-4, while every
history record and the dashboard reported 1.5e-4. `optimizer.load_state_dict` restores the
learning rate saved inside the checkpoint, overwriting the rate the optimizer was constructed
with, and the live-config path that would have corrected it only fires when `live_config.json`
changes *after* startup. The eval at iteration 199000 (head to head −3.75) measures 3e-4 training
and is void.

Fixed in `agent/ppo.py`: `apply_learning_rate()` re-asserts the configured rate after any optimizer
state is loaded, the history record now logs the optimizer's actual rate rather than the config's,
and `test_learning_rate_resume.py` fails if that call is removed. The run restarts from the king,
and the ~1,000 iterations already trained are discarded rather than resumed, so the run is clean.

## 4. How it is judged

Baseline: `evals/baselines/v3_iter198000.json`. Results named `lr15_<steps>M` (`--name`), to keep
them apart from the original v3 run's `v3_<steps>M` results.

**Primary:** head to head against the v3 king, goal difference per 10 min, **pooled over three
adjacent checkpoints** (nine seeds) at each decision point. Single-checkpoint readings are not
decisions (v4 §6, finding 3).

**Secondary, and the direct test of the hypothesis:** the spread of those three checkpoints. If the
churn is the learning rate, the spread shrinks. `approx_kl` should fall toward 0.01 and
`clip_fraction` into the single digits within the first few million steps; if KL does not move,
the learning rate was not the constraint and the run stops early.

**Necto guardrail (added 2026-09-20, after the 50M evals).** Goal difference per 10 min against
Necto must not be clearly worse than the king's −28.0, i.e. the seed ranges must overlap. A run is
not adopted on the head-to-head alone.

*Why.* v4 and this run both beat the king head to head while getting worse against Necto: v4 at
iteration 210000 (+11.9 h2h, −33 vs Necto) and this run at 201000 (+7.5 h2h, −36 to −38 vs Necto).
Training opponents are self-play plus a league of the king's own descendants, so a run drifts
toward beating its own lineage, and the head-to-head scores exactly that drift. Necto is the only
opponent that never changes and never trains, so it is the check on whether the bot is improving
or merely diverging. Nexto is reported beside it but is noisier (one seed).

**Guardrails:** touches per min, kickoff goals against, retreat scenario conceded, shot conversion.

**Scenario metrics are deterministic.** The drop/bounce/wall/retreat trials are seeded, so two runs
of the same checkpoint return identical scenario numbers (verified on the 50M pair). A repeat eval
buys precision on match metrics only, and a scenario difference between checkpoints is exact, never
noise. Match seeds carry about ±5 goals per 10 min on a single checkpoint, which is the bar any
checkpoint-to-checkpoint spread has to clear.

**Decision points:**
- **~5M steps:** KL and clip fraction have moved as expected, or stop.
- **150M:** pooled head to head at least level with the king, and the three-checkpoint spread
  smaller than v4's at a comparable point. If both fail, the learning rate is not the answer and
  the run stops.
- **400M:** adopt if the pooled head to head is clearly positive (every seed above zero) **and**
  the Necto guardrail holds. The v3 king was crowned on +15 against v2 by this measure, with Necto
  improving at the same time (−44.5 → −28.0), which is what a genuine gain looks like.

**Checkpoint hygiene:** any checkpoint that evals well is copied into `checkpoints/baselines/`
the same day. The league prunes everything else, and v4 lost its best checkpoint that way.

## 4a. Progress (through 250M steps, 2026-09-20)

Three adjacent checkpoints pooled at each point, against `evals/baselines/v3_iter198000.json`.

| Point | Head to head, pooled | Seeds positive | Checkpoint means | Spread | vs Necto |
|---|---|---|---|---|---|
| ~50M | +7.50 | 6/6 | 7.5 | — | −37.0 |
| ~100M | +10.62 | 11/12 | 10.0, 4.2, 18.2 | 14.0 | −32.3 |
| ~150M | +8.17 | 8/9 | 8.5, 3.8, 12.2 | 8.5 | −33.2 |
| ~200M | +11.00 | 9/9 | 16.5, 5.5, 11.0 | 11.0 | −30.8 |
| ~250M | +6.08 | 7/9 | 12.8, 5.0, 0.5 | 12.2 | −34.8 |

The king's Necto figure is −28.0 (seeds −30.0 to −25.5).

- **The 150M gate passes on both halves.** Pooled head to head is well above level, and the spread
  (8.5–14.0) is below v4's 25.2 at a comparable point. The run continues.
- **The mechanism worked; the effect plateaued.** `approx_kl` fell to 0.0105 and `clip_fraction` to
  9% within ~2,000 iterations and has been flat since (0.0148/10.9% → 0.0110/9.5% across the run).
  Explained variance 0.85, entropy and mean reward flat. Checkpoint spread halved and then stopped
  halving: 8–14 goals between checkpoints 200 iterations apart is still above the ±5 seed noise.
  The head-to-head jumped at 50M and has oscillated between +6 and +11 for 200M steps since.
- **Mid-run scenario dips were transient.** Retreat reached-goal-side went 87.5% (king) → 29.2% →
  15.6% → 88.9% → 83.3% → 93.1%, and drop first touch 16.7% → 1.0% → 20.8% → 13.9%. The 100M
  readings, which looked like a systematic collapse at the time, were not. Retreat conceded is still
  worse than the king (31.9% vs 16.7%).
- **The Necto guardrail is the weak point.** It passes only marginally at 250M (range −41.2 to
  −28.5 against the king's −30.0 to −25.5) and shows no trend toward improving over 250M steps.
  The attacking metrics behind it are all down on the king: goals for per 10 min 0.8 (king 4.2),
  touches per min 5.9 (7.0), shot conversion 21.7% (47.6%), kickoff goals against 9.1 (4.0),
  back-wall climbs 38 per 100 touches (11.8). This is v4's pattern in milder form: clearly better
  than its own ancestor, not better against the one opponent that never trains.
- **Pinned:** `checkpoints/baselines/lr15_iter210000.pt` (+16.5 over three seeds) and
  `lr15_iter213000.pt` (+12.75).

On present evidence the 400M bar — every seed above zero **and** the Necto guardrail holding — is
failed on both counts. If 300M and 350M look like 250M, the run is called at 400M rather than
extended, and the next lever is 0.8e-4 or the mechanical approach-quality problem from v4 §6.

## 5. If this works

A lower learning rate applies to every later run, and the reward question from v4 stays open:
loose-ball first touch is a mechanical deficit, so the next reward version (a real v5) should aim
at approach quality — arriving with speed retained and under control — or at nothing at all, if
v3's reward at a stable learning rate is already improving.
