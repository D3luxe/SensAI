# Reward v4 — specification

Status: **agreed and implemented 2026-09-19**; frozen as `config/reward_versions/v4.json` (code
`05332866aaf14755`, settings `354c613fc62e7ee9`). v3 is frozen
(`config/reward_versions/v3.json`) and its outcome is in `docs/reward_v3_spec.md` §9. The rules of
v3 §2 (R1–R7) apply unchanged.

## 1. What v3 left, and the one thing v4 changes

v3 is adopted: its king (iteration 198000) beats the v2 king +15 goals per 10 min head to head, and
it fixed the retreat and back-wall-climb problems. What it did not do, measured on the eval suite:

| Metric | v2 | v3 king (399M) | v3 at 500M |
|---|---|---|---|
| Drop scenario: SensAI first touch | 58% | 17% | 21% |
| Bounce scenario: SensAI first touch | 29% | 21% | 12.5% |
| Wall scenario: SensAI first touch | 100% | 75–83% | 75–79% |
| Touches per min vs Necto | 6.6 | 7.0 | 4.5–5.7 |
| Kickoff first touch | 3% | 13–16% | 0% |
| Retreat scenario: conceded | 46% | 17–25% | 42–46% |

The loose-ball regression began at ~150M, while T4 (closeness) was still at half weight, and grew
into general passivity after T4 reached zero at 300M. v3 has no term that values getting to the
ball before the opponent: T2 pays only once the ball moves, T3 only on contact, and T4 was
absolute (distance to the ball regardless of the opponent) and temporary by design.

Diagnosis order (R7): (a) the sim and the opponent are unchanged since v3, which played them
correctly; (b) a scenario already exists — `bounce_drop` at 0.04 through all of v3 — and the
regression happened with it in the mix. So (c): one reward term.

**v4 = v3 + one term, T6 (ball race).** Everything else — T1, T2, T3, T5, their weights, the
scenario mix, γ — is v3's, unchanged. T4 stays where v3 ended it: weight 0, no schedule. The v3
king was trained with T4 at zero for its last 100M steps, and T4 was an exploration aid for a
policy that no longer needs one; restarting its 1.0 → 0 anneal would make it a second change
running beside T6, and the eval could not say which one did what.

## 2. T6 — Ball race

`Φ_race(s) = (|b − o| − |b − c|) / 12000`, where `c` is the car, `o` the nearest opponent car and
`b` the ball (3D positions, the same normalisation as T4). Pays `w₆·(γΦ' − Φ)` with `w₆ = 1.0`,
`Φ(terminal) = 0` on goal steps, **no anneal**.

*Purpose:* value being nearer the ball than the opponent. Unlike T4 it is relative: sitting back is
free when the opponent is also away from the ball, and it costs only when the opponent is winning
the race to a ball SensAI could contest — exactly the drop/bounce/wall situations above. In 1v1 it is
zero-sum (the two cars' potentials are negatives of each other), so it cannot be farmed jointly in
self-play, and as a potential it nets zero over any loop (R4). It reads no actions (R2) and has no
gates (R6).

*Scale:* closing 1,200 uu on the opponent earns 0.1, the same rate T4 had at the start of v3; a
full race from 6,000 uu behind to level is 0.5. Goals (±10/−7.5) and T2 (±10 per end-to-end
play) still dominate.

*Known risks, and how the eval would show them:*
- **Double-commits and overextension.** Being nearer the ball is not always right (e.g. when a
  teammate has it, or when challenging leaves the net open). In 1v1 there is no teammate; the
  eval's caught-upfield and beaten-goal-side shares of goals against will show over-challenging.
- **Kickoffs.** Both cars start equidistant, so T6 is 0 at kickoff and pays for arriving first;
  kickoff first touch should rise, and kickoff goals against must not.

## 3. The run

- **Start:** `checkpoints/baselines/v3_iter198000.pt` (v3 king, stamped v3 `9243332689bd95fa` /
  `646c9927726c7f82`). Same version-change handling as v3: return normaliser reset, fresh
  optimizer, 50 iterations of critic warm-up. No anneal schedules in v4.
- **No control run. Nothing is changed during the run.**

## 4. How v4 is judged

Baseline: `evals/baselines/v3_iter198000.json`. The eval suite every ~50M steps, named
`v4_<steps>M`, run on a numbered checkpoint (`checkpoint_iter_*.pt`), never `latest_model.pt`.

**Primary:** head to head against the v3 king (`--reference checkpoints/baselines/v3_iter198000.pt`),
goal difference per 10 min. v3 showed it separates checkpoints that the Necto match cannot (399M vs
500M were ~5 apart against Necto and ~18 apart against v2 head to head). Goal difference vs Necto
is reported beside it.
**Guardrails:** touches per min, kickoff goals against, shot conversion, retreat scenario conceded.
**Where v4 must earn its keep:** SensAI first touch in the drop, bounce and wall scenarios;
kickoff first touch.

**Decision points:** 250M — stop and review if head to head vs the v3 king is clearly negative
(all seeds below zero) and falling. 500M — adopt v4 if head to head vs the v3 king is at least
level within the seed spread and the loose-ball first-touch metrics are better.

## 5. Implementation

- `env/rewards_v4.py`: `RewardV4` = v3's terms plus T6. `env/rewards_v3.py` is not edited (its hash
  is under test).
- `config/reward_versions/v4.json`: v3's settings with `closeness_weight: 0`, annealing disabled,
  plus `race_weight: 1.0`; `start_checkpoint` and
  `baseline_eval` as above. Registry entry in `env/reward_registry.py`.
- Tests (`test_rewards_v4.py`, `test_reward_versions_frozen.py`): T6 formula, zero-sum in 1v1,
  nearest opponent, v4 pays exactly v3 with T6 off, telescoping and `Φ(terminal)=0`, action
  independence, v3 → v4 version change, v4 frozen.
- Checkpoint numbering continues from the start checkpoint, so the v3 run's checkpoints after
  iteration 198000 were moved to `checkpoints/archive/v3_run/` (league state and TrueSkill ratings
  repointed) before the v4 run could overwrite them.
