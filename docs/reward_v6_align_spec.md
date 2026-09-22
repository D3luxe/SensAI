# Reward v6 — v5 plus Align Ball Goal

Status: **closed, not adopted, 2026-09-22** — ran to ~507M steps; outcome in §6. Was agreed and
implemented 2026-09-21; frozen as `config/reward_versions/v6.json` (code
`8f6de804eeb8e03f`, settings `88f455c4cdf9cae8`). v5's checkpoints are archived in
`checkpoints/archive/v5_run/` (`scripts/archive_run.py`). v5 is adopted
(`docs/reward_v5_gamma_spec.md` §7) and its king is `checkpoints/baselines/v5_iter222000.pt`. The
rules of `docs/reward_v3_spec.md` §2 (R1–R7) apply unchanged.

## 1. What v5 left, and the one thing v6 changes

v5 (v3's terms at a 20 s horizon) beat the v3 king on every pooled measure except one. Pooled over
the four 400M checkpoints (222000, 222200, 222400, 222600) against the pooled v3 reference (the
king and its two archived neighbours):

| | v5 at 400M | v3 pooled |
|---|---|---|
| Nexto goal difference /10 min | **−27.4** | −33.0 |
| Necto goal difference /10 min | **−29.4** | −35.2 |
| Necto goals for /10 min | **2.94** | 1.83 |
| on target /100 touches | **6.5** | 4.8 |
| kickoff goals against | **6.4** | 10.6 |
| back-wall climbs /100 touches | 18.2 | 19.4 |
| **retreat scenario: conceded** | **27.1%** | **20.8%** |

And where v5 still concedes to Necto (32.3 goals per 10 min, shares of goals against):

| cause | share |
|---|---|
| **caught upfield** | **43.8%** |
| back-wall climb | 22.0% |
| beaten while goal-side | 11.2% |
| goal-side but too far | 5.5% |

Two thirds of what Necto scores comes from SensAI being in the wrong place — upfield of the ball,
or up the back wall — rather than from being beaten in a contest it was positioned for. No v5 term
values position relative to the play: T2 values where the *ball* is, T5 values boost, and T4
(distance to the ball, retired at 0) had no tactical direction by design.

Diagnosis order (R7): (a) the sim and opponents are unchanged since v5, which played them
correctly; (b) the scenario mix already contains `retreat` situations, and v5 reaches goal-side
in 97% of them — so the scenario is common enough, and the deficit is not in *getting* back but in
the positioning that decides whether it has to; (c) one reward term.

**v6 = v5 + one term, T6 (align ball goal).** T1, T2, T3, T5, their weights, the scenario mix and
γ 0.9977 are v5's, unchanged. T4 stays at weight 0 with no schedule. v4's race term is not
revived.

## 2. T6 — Align ball goal

From Seer (Ma/Neville/Walo, eq. 3.7), with the attacking half corrected:

```
Φ_align(s) = 0.5 · cos(b − c, c − own) + 0.5 · cos(b − c, opp − c)
```

`c` the car, `b` the ball, `own`/`opp` the goal centres `(0, ∓5120, 321.4)` for blue (mirrored for
orange); 3D positions; `cos(u, v) = u·v / (|u|·|v|)`. Pays `w₆·(γΦ' − Φ)` with `w₆ = 1.0`,
`Φ(terminal) = 0` on goal steps, **no anneal**. Range [−1, 1].

- The **defending half** is +1 when the car sits on the line from its own net to the ball, between
  them — goal-side — and −1 when the ball is between the car and its own net.
- The **attacking half** is +1 when the ball is on the car's line to the opponent's net — the car is
  behind the ball, lined up to shoot — and −1 when the car is past the ball.
- Along the length of the pitch the halves agree: behind the ball is ≈ +1 on both (0.99), caught
  upfield is ≈ −1 on both (−1.00). That is the axis v5 concedes most along. Off it they separate:
  level with the ball out wide scores 0.

Values at canonical positions (3D, blue; computed, and pinned by the tests):

| position | defending | attacking | Φ |
|---|---|---|---|
| behind the ball on the axis, (0, −3000) with ball at centre | 0.99 | 1.00 | **0.99** |
| caught upfield, (0, 2000) with ball at (0, −1000) | −1.00 | −0.99 | **−1.00** |
| centre kickoff | 0.85 | 1.00 | 0.93 |
| off-centre kickoff | 0.94 | 1.00 | 0.97 |
| diagonal kickoff | 0.22 | 0.92 | 0.57 |
| level with the ball, out wide | 0.00 | 0.00 | 0.00 |
| in the net mouth, ball in the corner | 0.12 | 0.14 | 0.13 |

The last row is a real limitation, not a bug: the defending half measures being on the line from
the net to the ball, so shadowing the ball out toward the corner scores +1 on that half while
standing in the net mouth scores little. That is the positioning v5 lacks (it is caught upfield, not caught in the
net), but a policy that learns to leave the net too readily would show it in the beaten-goal-side
share of goals against.

**The correction.** The paper prints the attacking half as `cos(c − b, opp − c)`. With car, ball
and opponent net in a line in that order, that is −1: it would penalise the ideal attacking
position, contradicting the paper's own description ("a higher reward if it is behind the ball and
the ball is between the car and the opponent's goal"). rlgym-tools' `AlignBallGoal` uses
`cos(b − c, opp − c)`, which matches the description; v6 uses that. The unit tests pin the sign
with the positions in the table above.

**Degenerate points.** `cos` is undefined when the car is exactly on the ball or on a goal centre.
The implementation returns 0 for a half whose vectors have length below 1 uu. That is a numerical
guard, not a gate: at 1 uu the half is already meaningless, and the case never arises in play
(the car's centre cannot reach the ball's centre or the goal centre).

**Why potential-based, unlike Seer.** Seer pays this every step. Here it pays `γΦ' − Φ` (R4), so it
cannot be farmed: holding an aligned position earns nothing (slightly less than nothing, (γ−1)Φ),
and any loop nets zero. That matters more for this term than most. It is **not zero-sum** in 1v1:
both cars can be aligned at once (one goal-side, one behind the ball), so a per-step version could
be collected jointly in self-play by two cars politely holding their shapes — the "excessive
patience" pathology v5 §5 warned about.

The consequence has to be said plainly. Potential shaping does not change which policy is optimal;
it changes how quickly the critic learns what T1 already implies — that being caught upfield
concedes goals. v5 showed the horizon was the binding constraint on that learning. T6 is a bet that
positioning is still under-credited even at a 20 s horizon, because a positioning mistake costs a
goal many seconds later and the critic has to learn the whole chain. If the retreat and
caught-upfield numbers do not move, the conclusion is that the gap is not a credit-assignment
problem, and the next change is a scenario (R7 b), not another term.

*Scale:* moving from caught upfield (Φ ≈ −1) to goal-side and behind the ball (Φ ≈ +1) earns 2.0
over the move — the same size as a quarter of an end-to-end T2 play, and a fifth of a goal. Goals
(+10 / −7.5) and T2 (±10 per end-to-end play) still dominate. Weight 1.0 matches the two previous
single-term additions (v3's T4, v4's T6), so the dose is comparable.

*Known risks, and how the eval would show them:*
- **Passivity.** Goal-side is rewarded; challenging from goal-side is not. If touches per min falls
  below the v5 reference (7.5) and goals for vs Necto falls with it, T6 is making the policy wait.
- **Ball-chasing from behind.** The attacking half pays for lining up behind the ball, which can
  mean turning away from a contest to circle round. The eval's first-touch rates in the drop,
  bounce and wall scenarios (v5 king: 20.8%, 33.3%, 75.0%) would fall.
- **Kickoffs.** Both cars start at mirrored positions, so Φ is equal for both (0.57 to 0.97
  depending on the spawn), so T6 gives neither car an edge at the start. It should not change
  kickoffs, and kickoff goals against must not rise.

## 3. The run

- **Start:** `checkpoints/baselines/v5_iter222000.pt`, the v5 king (stamped v5,
  code `9243332689bd95fa` / settings `d911cde2868fbdda`). Version change v5 → v6: return
  normaliser reset, fresh optimizer, 50 iterations of critic warm-up. γ is unchanged, so the
  critic's scale is too; `explained_variance` should recover within the warm-up.
- **League.** Ratings are the batch fit pinned on the v3 king (`utils/rating_fit.py`), which does
  not care that the population is from a new run. The v5 checkpoints stay in the fit as ordinary
  players.
- **Checkpoint numbering** continues from 222000 and would overwrite v5's own files after it, so
  v5's checkpoints move to `checkpoints/archive/v5_run/` first. The batch fit is keyed by path, so
  the results log and leaderboard entries for moved files are repointed in the same step (as was
  done for v3, v4 and the learning-rate run).
- **Nothing is changed during the run.** Learning rate 1.5e-4, as v5.

## 4. How v6 is judged

Baseline: `evals/baselines/v5_iter222000.json`. Results named `v6_<steps>M`, on numbered
checkpoints only.

Everything is pooled over **all** checkpoints evaluated at a decision point (at least three,
adjacent). v5 taught this twice: a single checkpoint's Necto reading varies from one eval to the
next (222000 read 4.25 and then 1.00 goals for), and a king chosen partly for evaluating well is a
winner's-curse reference.

**Reference: the v5 400M group, pooled** — the table in §1. Not the v5 king's own readings.

**Primary:** head to head against the v5 king (`--reference checkpoints/baselines/v5_iter222000.pt`),
goal difference per 10 min. The per-seed sd at v5's 400M was 6.7.

**Guardrails** (not clearly worse than the reference, where "clearly" means outside the spread of
the reference's own four checkpoints): Necto goals for (2.94), touches per min (7.5), kickoff goals
against (6.4), Nexto goal difference (−27.4).

**Where v6 must earn its keep:** retreat scenario conceded (27.1% → toward v3's 20.8%), the
caught-upfield share of goals against (43.8%), back-wall climbs (18.2), and Necto goal difference
(−29.4), which is what those three add up to.

**Decision points:**
- **150M:** stop and review if the pooled head to head is clearly negative, or if touches per min
  and goals for vs Necto are both below the reference (the passivity failure).
- **400M:** adopt if the pooled head to head is positive by more than two standard errors of the
  pooled seeds, **and** the guardrails hold, **and** at least two of the three positioning metrics
  are better than the reference by more than its own checkpoint spread.

These gates are written before the run. v5's were revised twice during its run, both times for
good reasons, both times before the relevant evals — v6 should not need to.

**Checkpoint hygiene:** any checkpoint that evals well is copied into `checkpoints/baselines/` the
same day.

## 5. Implementation

- `env/rewards_v6.py`: `RewardV6` = v3's terms plus T6. `env/rewards_v3.py` is not edited (its
  hash is under test, and v5 shares it).
- `config/reward_versions/v6.json`: v5's settings plus `align_weight: 1.0`; `start_checkpoint`
  and `baseline_eval` as above. Registry entry in `env/reward_registry.py`.
- Tests (`test_rewards_v6.py`, `test_reward_versions_frozen.py`):
  - T6 at the canonical positions in §2's table, to two decimals;
  - the attacking half's sign against the paper's printed form, so the correction cannot be
    silently undone;
  - mirrored for orange; degenerate points return 0 for that half;
  - v6 pays exactly v5 with `align_weight: 0`;
  - telescoping and `Φ(terminal) = 0`; action independence (R2);
  - v5 → v6 version change; v6 frozen once started.
- Six terms counting the retired T4, which is R5's limit. v7 cannot add a term without removing one.

## 6. Outcome (run stopped at ~507M steps, 2026-09-22)

**Not adopted.** `checkpoints/baselines/v5_iter222000.pt` stays the king, and v7 is built on v5's
reward, not v6's (`docs/reward_v7_replay_spec.md`).

**The 400M gate.** Pooled over 246200, 246400 and 246600 against the v5 400M reference:

| | v6 400M | v5 reference | gate |
|---|---|---|---|
| head to head vs v5 king | +2.50 ± 1.29 (7/9 seeds) | — | needs > 2 SE: **1.94, missed** |
| Necto goals for | 4.9 | 2.9 | guardrail holds |
| touches /min | 7.2 | 7.5 | holds |
| kickoff goals against | 7.8 | 6.4 | holds (range 2.7–11.3) |
| Nexto goal difference | −29.0 | −27.4 | holds (range −33.0 to −23.2) |
| retreat conceded | 27.8% | 27.1% | not better |
| caught-upfield goals /10 min | 13.0 | 14.4 | not better (range 8.7–18.4) |
| back-wall climbs /100 touches | 16.5 | 18.1 | not better (range 10.9–25.4) |

Positioning improved on none of the three, where two were required.

**The run, pooled at each decision point:**

| | h2h vs v5 king | Necto GF | Necto GD | touches | kickoff GA | retreat conceded |
|---|---|---|---|---|---|---|
| 100M | +7.25 | 1.6 | −30.3 | 5.3 | 9.2 | 33.3% |
| 150M | +3.50 | 1.4 | −31.0 | 5.9 | 2.7 | 27.8% |
| 200M | −2.58 | 2.8 | −29.3 | 7.5 | 13.4 | 26.4% |
| 250M | +1.75 | 2.0 | −26.8 | 6.3 | 2.9 | 34.7% |
| 300M | +2.75 | 1.7 | −33.3 | 6.8 | 11.4 | 36.1% |
| 350M | +4.67 | **3.8** | **−23.9** | **7.8** | 6.4 | 30.6% |
| 400M | +2.50 | **4.9** | **−25.2** | 7.2 | 7.8 | 27.8% |
| 450M | +1.58 | 1.6 | −27.2 | 6.4 | 4.0 | 26.4% |
| 500M | +5.69 | 2.2 | −29.8 | 6.8 | 13.5 | 44.8% |

**Findings, for later versions:**

1. **T6 moved the failure around rather than removing it.** At 100M goals conceded while goal-side
   doubled (3.6 → 7.5 per 10 min) as the bot got into position and lost the duel from there; by
   150M caught-upfield goals had jumped to 20 per 10 min instead; by 400M both were back at v5's
   level. Positioning is not a credit-assignment problem a potential can fix at this horizon, which
   is the conclusion §2 said this outcome would mean. The next change is a scenario (R7 b): v7.
2. **350–400M was the best stretch against Necto of any run** (goal difference −23.9 and −25.2
   pooled; single checkpoints 243600 at −17.8 and 246600 at −21.2, both pinned with 243400). It did
   not hold: by 450–500M the Necto numbers were back at v5's. Those checkpoints are not v7's start,
   because they were chosen for evaluating well.
3. **The last 100M were lineage drift again.** Head to head peaked at 500M (+5.69, 10/12 seeds) while
   kickoff goals against doubled and the retreat scenario went from 27.8% to 44.8% conceded. The
   league crowned 253000 (+5.75 head to head, 0.5 Necto goals for, 58.3% retreat conceded). Third
   run in a row where head to head alone would have adopted the wrong checkpoint.
4. **Boost is a collection problem.** From 193M the eval reports boost economy: ~1 big pad a
   minute, 1–4% of boost spent while already supersonic, 54–79% of retreats begun with under 12
   boost, and time on empty rising to 50% by 500M. The bot does not waste boost; it does not collect
   it. That is the target for a later version's boost change, not spending.
5. **The 150M passivity gate compared against v5's 400M numbers,** which v5 itself would have
   failed at 150M. v7's gate compares against v5 at the same step count.
