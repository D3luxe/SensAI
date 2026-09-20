# Run v5 — v3's reward at a lower learning rate

Status: **draft, 2026-09-20** — not started. This is a **training run, not a reward version**.
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

**Guardrails:** touches per min, kickoff goals against, retreat scenario conceded, shot conversion.

**Decision points:**
- **~5M steps:** KL and clip fraction have moved as expected, or stop.
- **150M:** pooled head to head at least level with the king, and the three-checkpoint spread
  smaller than v4's at a comparable point. If both fail, the learning rate is not the answer and
  the run stops.
- **400M:** adopt if the pooled head to head is clearly positive (every seed above zero). The v3
  king was crowned on +15 against v2 by this measure.

**Checkpoint hygiene:** any checkpoint that evals well is copied into `checkpoints/baselines/`
the same day. The league prunes everything else, and v4 lost its best checkpoint that way.

## 5. If this works

A lower learning rate applies to every later run, and the reward question from v4 stays open:
loose-ball first touch is a mechanical deficit, so the next reward version (a real v5) should aim
at approach quality — arriving with speed retained and under control — or at nothing at all, if
v3's reward at a stable learning rate is already improving.
