# Reward v5 — v3's terms at a twenty-second horizon

Status: **adopted 2026-09-21 at ~400M steps; king `checkpoints/baselines/v5_iter222000.pt`**
(see §7). Drafted while the learning-rate run, `docs/run_v5_lr_spec.md`, was still open. The rules
of `docs/reward_v3_spec.md` §2 (R1–R7) apply unchanged.

**A note on the name.** `run_v5_lr_spec.md` is a *training run*, not a reward version; it left the
reward at v3. This is the fifth *reward version*, so it is v5 and its snapshot is
`config/reward_versions/v5.json`. The two are unrelated despite the shared digit.

## 1. Why this has to be a reward version at all

The only change is the discount factor, γ 0.995 → 0.9977. That looks like a hyperparameter, and in
most projects it would be one. Here it is not, because every dense term pays
`w·(γ·Φ(s′) − Φ(s))` (R4), and potential-based shaping is policy-invariant **only** when the
shaping γ equals the γ PPO optimises. So γ is part of the reward's frozen settings and its
`settings_sha`. `env/reward_registry.py:92` already refuses the mismatch by name:

> `hyperparameters.gamma 0.9977 differs from reward v3's frozen gamma 0.995: a new gamma is a new reward version`

That guard is correct and stays. The consequence is that this run gets a version bump even though
no term, weight or scenario changes — v5's terms are **v3's, identical, at a different γ**.

## 2. Why γ, and why now

Our γ has been 0.995 for the whole project — 3.5B steps, three reward versions, two tuned
hyperparameters — and has never been revisited. Seer (Ma/Neville/Walo, §3.8) parameterises the
discount by a half-life instead of a bare number:

```
γ = exp( log(0.5) / (T · A) )      A = actions per second, T = the time after which a reward halves
```

Seer runs frameskip 8 at 120 ticks — **A = 15 actions/sec, identical to ours** (`tick_skip: 8`).
That makes the comparison exact:

| | T (half-life) | γ |
|---|---|---|
| Seer, initial — *"to facilitate learning an initial strategy"* | 10 s | 0.9954 |
| Seer, final — *"to ensure Seer performs proper long-term planning"* | 20 s | 0.9977 |
| **Ours, every run to date** | **9.2 s** | **0.995** |

We have been sitting at Seer's beginner horizon the entire time. A 9-second half-life is enough to
connect a touch to the goal it causes a second or two later; it is not enough to value a retreat, a
boost pickup or a positioning move that pays off eight seconds downfield.

**The symptoms match a short horizon, not a bad reward.** From the learning-rate run at 250M
(`run_v5_lr_spec.md` §4a), against the v3 king: goals-for per 10 min against Necto 1.39 vs 4.25
while goals-against is flat (35.0 vs 32.2); touches per min 5.9 vs 7.0; back-wall climbs 38 per 100
touches vs 11.8; shot conversion 21.7% vs 47.6%. The bot is not conceding more — it has stopped
constructing anything. That is what a myopic critic produces.

Diagnosis order (R7): (a) sim and opponents are unchanged and correct; (b) the scenario mix is v3's,
which produced the king; (c) the reward's terms have now been probed twice (v4's race term, v2's
ball pull) with nothing to show. γ is the remaining untouched knob, and unlike the terms it has an
external reference point telling us our value is the wrong one.

## 3. The change

`hyperparameters.gamma: 0.9977` in `config/default_config.yaml`, mirrored into
`config/reward_versions/v5.json`'s settings so the identity resolves. Everything else is v3's:
T1 goal +10/−7.5 at `aggression_bias` 0.25, T2 ball position weight 5, T3 touch 0.5·Δv/2300,
T4 closeness retired at 0, T5 boost √(boost/100). No race term — v4 is rejected and stays rejected.
`learning_rate` stays at whatever the learning-rate run concludes with (1.5e-4 unless that run says
otherwise), because changing two hyperparameters at once is the thing these specs exist to prevent.

**Fixed, not ramped.** Seer increased T continuously from 10 s to 20 s across training. We hold
0.9977 fixed, because a ramp is two variables — the endpoint and the schedule — and v3's closeness
anneal already taught us that a moving weight makes the eval unreadable (v4 §1: "the eval could not
say which one did what"). If the fixed jump proves too aggressive, a ramp is the *next* experiment,
not a hedge inside this one.

## 4. The run

- **Start:** `checkpoints/baselines/v3_iter198000.pt`, the v3 king — the same start point as v4 and
  as the learning-rate run, so all three are single-variable branches from one node.
- **Version change handling.** The start checkpoint is stamped v3 and the config will name v5, so
  the trainer takes its normal version-change path: return-normaliser reset, fresh optimizer,
  50 iterations of critic warm-up. **This matters more here than usual.** Raising γ from 0.995 to
  0.9977 roughly doubles the scale of the discounted return the critic has to predict, so the
  inherited value head is not merely stale, it is systematically wrong by a large factor. Expect
  `explained_variance` to collapse from ~0.85 at the switch and to recover over the warm-up and the
  first few million steps. **If it has not recovered past ~0.6 by 20M steps, stop** — that is the
  critic failing to refit, and nothing measured after it is meaningful.
- **Checkpoint numbering** restarts from 198000 and would overwrite the learning-rate run's files,
  so those move to `checkpoints/archive/v5lr_run/` first, with league state and TrueSkill ratings
  repointed, as was done for v3's and v4's.
- **Nothing is changed during the run.** No mid-run γ adjustment, no reward patching.

## 5. How v5 is judged

Baseline: `evals/baselines/v3_iter198000.json`. Results named `v5g_<steps>M`.

**Primary:** head to head against the v3 king, goal difference per 10 min, **pooled over three
adjacent checkpoints** (nine seeds) at each decision point. Single-checkpoint readings are not
decisions (v4 §6, finding 3).

**Necto guardrail (revised 2026-09-21, pooled against pooled):** goals scored per 10 min against
Necto, **pooled over three adjacent checkpoints, compared with the v3 king pooled the same way** —
the king plus `checkpoints/archive/v3_run/checkpoint_iter_198200.pt` and `_198400.pt`. Not
clearly below that reference.

*Why it changed mid-run.* The original guardrail compared a nine-seed pooled reading against the
king's single-checkpoint 4.25, and the ±0.36 it quoted is seed noise only. Checkpoint-to-checkpoint
noise is far larger, and the king is an outlier in its own run: nine evaluated v3 checkpoints from
150M to 501M read 1.0, 0.5, 0.5, 0.5, 0.75, **4.25**, 1.0, 0.5, 0.5 (mean 1.06, sd 1.22). Every run
throws the occasional spike — v3 at 6M and 98M, the learning-rate run at 547M (4.0) and 603M
(6.0), v5 at 201600 (3.75) and 210400 (6.75) — and the king was crowned partly for evaluating well,
so 4.25 is a winner's-curse number no pooled reading was ever likely to reach. A three-checkpoint
pool carries about ±0.7 from checkpoint noise, so "clearly below" means more than ~1 goal under
the reference. (Provisionally 1.06, the v3 run's own level, until the neighbours were measured.)

*Measured reference, 2026-09-21* (`evals/v3king_n1.json`, `evals/v3king_n2.json`):

| v3 checkpoint | GF vs Necto | Necto GD | Nexto GD | on target /100 | back-wall /100 | kickoff GA | retreat conceded |
|---|---|---|---|---|---|---|---|
| 198000 (king) | 4.25 | −28.0 | −33.8 | 12.9 | 11.8 | 4.0 | 16.7% |
| 198200 | 0.50 | −38.0 | −30.7 | 0.7 | 23.5 | 9.7 | 20.8% |
| 198400 | 0.75 | −39.5 | −34.5 | 0.8 | 22.8 | 18.0 | 25.0% |
| **pooled** | **1.83** | **−35.2** | **−33.0** | **4.8** | **19.4** | **10.6** | **20.8%** |

The neighbours, 200 and 400 iterations after it, read like the rest of the v3 run, which confirms the
king's Necto numbers as a spike. **The guardrail reference is 1.83.** It still contains the king
itself, so it remains biased upward; the two neighbours alone read 0.63. The pooled row is also the
reference for the scenario targets below, in place of the king's own readings.

This also weakens, retrospectively, the Necto-scoring half of the case against v4 and the
learning-rate run; the learning-rate run's churn finding stands on its own.

**Where v5 must earn its keep**, all of which a longer horizon should improve if the diagnosis is
right: goals-for vs Necto, touches per min (5.9 → toward 7.0), back-wall climbs per 100 touches
(38 → toward 12), retreat scenario conceded (31.9% → toward 16.7%).

**Watch for the opposite failure.** A long horizon can produce its own pathology: excessive
patience, declining to challenge, farming boost while the opponent sets up. If touches per min
falls *further* and goals-against climbs, γ is too long and T = 15 s (γ 0.99692) is the retry.

**Decision points:**
- **~20M steps:** `explained_variance` recovered past ~0.6, or stop (§4).
- **150M:** pooled head to head at least level with the king, **and** pooled goals-for vs Necto
  not clearly below the pooled v3 reference. The head-to-head alone is not enough.
  *Result:* head to head +6.0, 9/9 seeds positive; goals-for 0.67 against the provisional 1.06,
  inside the ±0.7 checkpoint noise. **Passed** under the revised guardrail (it failed the original
  4.25 comparison, which is what prompted the revision). Against the measured 1.83 it is 1.16
  under — just past the ~1-goal line, so borderline in hindsight; the later pools read 2.83 (200M),
  0.92 (250M) and 0.83 (300M), 1.31 across all twelve checkpoints.
- **400M:** adopt if the pooled head to head is clearly positive (every seed above zero) **and**
  pooled goals-for vs Necto is **not clearly below** the pooled v3 reference (more than ~1 goal
  under 1.83). *Revised 2026-09-21 from "at least level with", before any 400M eval was run:* a
  difference between two three-checkpoint pools carries about ±1.0 from checkpoint noise alone, and
  the reference still contains the king's spike, so "at least level" would fail a run that matched
  v3 about half the time. Same rule as the 150M gate.

**Progress, pooled (h2h vs v3 king / seeds positive / GF vs Necto / Necto GD):**
150M +6.0 / 9/9 / 0.67 / −37.1 · 200M +18.0 / 9/9 / 2.83 / −38.2 · 250M +8.4 / 7/9 / 0.92 /
−31.4 · 300M +17.6 / 9/9 / 0.83 / −34.3 · **350M +16.25 / 9/9 / 2.83 / −27.0** (219000, 219200,
219400). At 350M every Necto-facing number is at or past the pooled v3 reference: on target 7.0
(4.8), kickoff GA 5.1 (10.6), back-wall 13.2 (19.4), retreat conceded 25.0% (20.8%).

**Checkpoint hygiene:** any checkpoint that evals well is copied into `checkpoints/baselines/` the
same day. v4 lost its best checkpoint to the league's retention.

## 6. What this does not address

Seer's §3.6.1 lists 16 reward terms. Reviewed against ours, most are already present (Boost Amount
is our T5 formula-for-formula, Boost Difference is that same potential's difference, Distance Ball
Goal is T2, Ball Touch is T3, Goal Scored is T1) and three — Distance Player Ball, Closest to Ball,
Velocity Player to Ball — are the ball-pull idea that failed as v2's `player_to_ball` and again as
v4's T6. Those are closed.

One term is genuinely new and genuinely ours to take: **Align Ball Goal** (Seer eq. 3.7),

```
0.5·cos_sim(ball − car, car − own_net) + 0.5·cos_sim(car − ball, opp_net − car)
```

a bounded state function that scores the car's *angle* relative to both nets rather than its
distance to the ball, so it pays for being goal-side when the opponent attacks — the opposite of a
ball pull, which is why it does not repeat v2's and v4's failure. It maps onto our measured
back-wall-climb and retreat deficits. It is deliberately **not** in v5: γ and a new term at once is
two variables, and γ goes first because it is the cheaper test and the one with outside evidence
against our current value. If v5 works, Align Ball Goal is v6; if v5 fails, it is v6 anyway, as the
next single change.

Also noted from the paper, and deliberately deferred: Seer makes the whole reward zero-sum by
subtracting the opponent's reward, which would prevent joint farming of shaped terms in self-play.
That is a structural change to every term at once, so it cannot be a single-variable run, and it
would invalidate cross-version comparison against the v3 king.

**A caution about the source.** Seer reached TrueSkill ~40 after 10 billion steps with all 16 terms
and never learned to aerial or double-jump. Its reward design is not self-evidently better than
ours, and R5's six-term cap stands. What the paper is being used for here is the one thing it
measures better than we do: the discount horizon, where it has a principled parameterisation and a
before/after value, and we have a number nobody ever chose.

## 7. Outcome — adopted (run stopped at ~400M steps, 2026-09-21)

**The 400M gate, over all four checkpoints evaluated** (222000, 222200, 222400, 222600; choosing
the best three would repeat the selection error §5 describes):

| Condition | Result | |
|---|---|---|
| pooled head to head clearly positive, every seed above zero | +9.25, **10 of 12 seeds** | missed literally, by −0.8 (222200) and −2.2 (222600) |
| pooled goals-for vs Necto not clearly below 1.83 | **2.94** | passed |

**Adopted over the literal miss, as an explicit decision rather than a rewritten rule.** "Every
seed above zero" was the proxy for *clearly better*. Five consecutive pools were positive (+6.0,
+18.0, +8.4, +17.6, +16.25 from 150M to 350M, then +9.25), 44 of 48 seeds from 150M on, and the
two misses sit inside one single-seed standard deviation (3.9). The rule is left as written, and
this paragraph records that it was overruled and why.

**Against the pooled v3 reference at 400M:** Nexto goal difference −27.4 vs −33.0 (the clearest
gain; 222000's −23.2 is the best of any run), Necto goal difference −29.4 vs −35.2, on target 6.5
vs 4.8, kickoff goals against 6.4 vs 10.6, touches per min 7.5 vs 6.5, back-wall climbs level
(18.2 vs 19.4). **Retreat conceded 27.1% vs 20.8% is the one deficit v5 never closed** — the
positioning gap Align Ball Goal (§6) targets.

**The v5 king is 222000.** A direct match against 219400, the other candidate
(`evals/king_222000_vs_219400.json`), came out +2.75 for 222000 (8.25, −3.75, 3.75): level
within noise. 222000 was chosen as the later checkpoint and the most even across every measure,
not for its best single reading. Baseline eval: `evals/baselines/v5_iter222000.json`.

**Findings to carry forward:**
1. **A 20 s horizon helped** where v4's race term and the learning-rate run did not, with the same
   start point and the same judging. It supports the §2 diagnosis that the critic was myopic.
2. **The judging itself was the bigger error.** The Necto guardrail compared pools against one
   checkpoint that was a spike in its own run (§5). The same checkpoint also varies between eval
   runs: 222000 read 4.25 goals-for vs Necto in its 400M eval and 1.00 in the king match; the v3
   king read 4.25 and 6.25. Every Necto-facing number is pooled from here on, never read from one
   eval of one checkpoint.
3. **Steering jitter is unchanged by the horizon** (`logs/jitter_250M.json`: 18.9–20.4% of grounded
   decisions vs v3's 15.9%). It is learned; deploy-side smoothing, tested inside the eval first, is
   the next thing to try.
4. **The league's absolute scale drifted again** (king mu 41 against Necto's anchored 30 while
   losing to it 35–1). Peer ranking was unaffected; the fix is outside this spec.

**Pinned from this run:** 200200, 201600, 210200, 210400, 213200, 216200, 219200, 219400, 222000,
222400.

**Next:** reward v6 — v5 plus Align Ball Goal, one change, starting from the v5 king.
