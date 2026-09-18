# Reward v3 — specification

Status: **agreed 2026-09-18** (weights, aggression_bias, scenario mix, start point); not yet implemented. v2 is frozen (git tag `reward-v2`,
`config/reward_versions/v2.json`, `test_reward_v2_frozen.py`) and stays runnable beside v3.

## 1. Why a v3, and what went wrong twice before

v2 has 16 terms, ~4,300 lines, 80 smooth gates, 74 clamps and seven terms that read the
controller. Every behaviour problem found in the last month came from terms *interacting*, not
from one term being wrong (e.g. retreat flip-chaining: `retreat_flip` pays on the dodge,
`air_roll_recovery` exempts the landing, `boost` charges the alternative). The traps:

1. **Symptom patching.** Each observed behaviour got a new term or gate, checked against the one
   situation it targeted and never against everything else.
2. **Paying actions and events instead of outcomes.** Every such bonus produced an exploit, and
   every exploit a gate.
3. **Changing rewards in place, mid-run.** No record of which reward produced which checkpoint.
4. **Judging from anecdotes.** Single logs and 30-minute runs whose spread (3/54 vs 9/36) is larger
   than the effects being judged.
5. **Environment bugs read as reward problems.** The simulated Necto could not flip from the
   ground and did not play its scripted kickoff for the whole history of the project; rewards were
   tuned against an opponent that did not exist.

v3 is not "v2 with fewer terms". It is a small reward that obeys the rules below, run once,
end to end, and judged on the evaluation suite.

## 2. Rules (apply to v3 and every later version)

R1. **Frozen per run.** Terms, weights, anneal schedules and the scenario mix are fixed before the
    run and never edited during it. Any change is a new version and a new run. The checkpoint
    records the version (`reward_identity`).
R2. **State and outcome only.** No term reads the action vector. Enforced by a test that the
    reward is identical for any two action vectors given the same state transition.
R3. **No payment for performing an action or event** (a dodge, a jump, a powerslide, a pad). An
    event may be *measured* only through its effect on the game state.
R4. **Every dense term is potential-based** — `F = γ·Φ(s') − Φ(s)` with `Φ(terminal) = 0` on goal
    steps — or carries an anneal schedule declared in this spec. Potential terms cannot be farmed:
    any loop sums to zero.
R5. **At most 6 terms.** Each has one paragraph saying what it is for and unit tests.
R6. **No gates without a measured reason.** A term's formula is written here in full; if it needs a
    special case, that is a design smell to resolve in the next version, not a patch.
R7. **Diagnose in order**: (a) sim/opponent fidelity, (b) a scenario/state setter that makes the
    situation common, (c) only then a reward change — in the next version.

## 3. Terms

All distances in uu, speeds in uu/s. `own`/`opp` goal centres are `(0, ∓5120)` for blue (mirrored
for orange). `γ = 0.995` (the trainer's discount). Potentials use `Φ(terminal) = 0` on goal steps;
on truncations (timeouts, resolved scenarios) `Φ(s')` is used as normal and the trainer bootstraps.

| # | Term | Form | Weight | Anneal |
|---|------|------|--------|--------|
| T1 | Goal | sparse event | +10 / −7.5 (aggression_bias 0.25) | none |
| T2 | Ball position | potential | 5 | none |
| T3 | Touch | outcome event | 0.5 | none |
| T4 | Closeness to ball | potential | 1.0 | 1.0 → 0 over the first 300M steps |
| T5 | Boost held | potential | 1.0 | none |

**T1 — Goal.** `goal_reward = +10` to every car on the scoring team and
`concede_reward = −goal_reward · (1 − aggression_bias)` to every car on the conceding team, on the
goal step, with `aggression_bias = 0.25` (so −7.5). No speed, placement or save multipliers: how
good a shot or save was shows up in T2 on the way there. *Purpose:* the objective; everything else
must stay small beside it.

*aggression_bias* is the share of the concede penalty removed to promote attacking play: a
symmetric ±10 tends to produce a passive player, while at 0.25 a 1–1 trade nets +2.5, so the policy
accepts riskier attacking play. It touches T1 only; T2 stays zero-sum. *Watch:* the v2 baseline is
over-committed rather than passive (caught upfield 34%, back-wall climbs 24% of goals against), so
a cheaper concede could also weaken the push to defend. If goals against per 10 min and the
caught-upfield share both worsen without goals for rising, v4 lowers it.

**T2 — Ball position.** `Φ_ball(s) = (|b − own| − |b − opp|) / 10240`, ranging from −1 (on our
goal line) to +1 (on theirs), 3D ball position `b`. Pays `5·(γΦ' − Φ)`. Moving the ball the length
of the pitch toward their goal earns about +10 over the play; the opponent doing the same costs the
same. Saves and clears are simply the ball moving away from our goal. *Purpose:* dense, zero-sum
credit for moving the ball the right way, without a velocity term that can be farmed by juggling.

**T3 — Touch.** On a step where the car's touch count increases: `0.5 · min(1, |Δv_ball| / 2300)`,
with `Δv_ball` the change in ball velocity over that step. Direction is not judged here — T2 does
that. *Purpose:* makes contact with the ball worth something before it has learned where to send
it; a hard contact is worth more than a tap (a 2000 uu/s hit ~0.43, a 100 uu/s tap ~0.02).
*Known risk:* many taps add up — 30 taps earn ~0.65, more than one strike's T3. Pure tap-farming
still earns far less than the T2 a dribble moving goalward collects, and the eval's touches/min
metric will show it if it happens. If it does, the v4 fix is to scale T3 down (or anneal it), not
to add a freshness gate.

**T4 — Closeness to ball.** `Φ_car(s) = −|b − c| / 12000` (3D car-to-ball distance, normalised by
roughly the pitch diagonal). Pays `w₄(t)·(γΦ' − Φ)` with `w₄` falling linearly from 1.0 to 0 over
the first 300M environment steps of the run, on the run's own clock. *Purpose:* an exploration aid
that points at the ball early; it is potential-based, so driving past the ball and back nets zero,
and it fades out so it cannot shape late-game positioning. It deliberately has no tactical target.

**T5 — Boost held.** `Φ_boost(s) = √(boost / 100)`. Pays `1.0·(γΦ' − Φ)`. Picking up boost is
worth more when low than when full; spending it costs what refilling refunds. No situational
penalties. *Purpose:* makes boost a resource; whether to spend it is decided by T1/T2 outcomes.

Scale check (per 10 minutes of play, rough): T1 +10 per goal scored, −7.5 per goal conceded; T2 sums to ±10 per end-to-end
play; T3 up to ~0.5 per hard touch (~5–15 per 10 min); T4 ≤ 1 per approach, zero net over loops;
T5 ≤ 1 per tank. Goals dominate, as they must. The trainer normalises returns
(`normalize_returns: true`), so only these ratios matter.

**Not in v3** (removed with no replacement term): jump bridge, retreat flip, air-roll recovery,
powerslide, overshoot, spin/lateral-slip/slide-waste costs, jump cost, time cost, own-goal threat,
save bonus, the tactical pursuit target and everything in it (intercepts, shadowing, challenge
sigmoid, hold/standoff), boost pathing and waste penalties. Their intended behaviours are left to
T1/T2 outcomes plus the scenario mix below.

## 4. Scenario mix (training starts; fixed for the run)

State setting, not reward terms, is how v3 makes hard situations common enough to learn.

| Scenario | v2 | v3 | Notes |
|---|---|---|---|
| replay (2M frames, real match states) | 0.50 | 0.45 | the backbone: realistic states |
| kickoff | 0.09 | 0.15 | the real Necto now contests kickoffs; SensAI loses most |
| aerial | 0.08 | 0.08 | |
| goalie_save | 0.05 | 0.06 | |
| wall_play | 0.05 | 0.05 | |
| wall_rebound | 0.05 | 0.05 | |
| turnaround | 0.05 | 0.05 | |
| dribble_flick | 0.05 | 0.03 | |
| custom | 0.08 | 0.00 | user presets are edited over time, which breaks R1 |
| **bounce_drop** (new) | – | 0.04 | ball falling or bouncing near the car (the whiff problem) |
| **retreat** (new) | – | 0.04 | ball heading for our net, car upfield of it (the retreat problem) |

The two new setters draw from wider ranges than the eval suite's `drop`/`bounce`/`retreat`
generators and randomise which team is in the situation, so training does not memorise the eval
starts. Training opponent mix is unchanged (self-play 0.5, king 0.25, pool 0.25, Necto 0.1875).

## 5. The run

- **Start:** policy weights from `checkpoints/baselines/v2_iter173600.pt` — the v2 king when v2
  training was stopped (iteration 173,600, 2.84B steps, stamped v2 `7c5d7a2b15e057d7` /
  `20d320a8ec343cca`). Its eval, `evals/v2_iter173600.json`, is the baseline v3 is judged against,
  so the baseline is exactly the policy v3 starts from. The value function was
  trained on v2's returns and is wrong for v3: run the existing critic warm-up
  (`critic_warmup_iterations`, policy frozen) and reset the return normaliser before the first
  policy update. Fresh optimizer state. v3's anneal clock starts at 0.
- **No control run.** If v2's habits look like they are fighting the new reward, a from-scratch v3
  run can be started separately and compared on the eval suite.
- **Nothing is changed during the run.** If something looks wrong, it is noted for v4.

## 6. How v3 is judged

The eval suite (`scripts/eval_suite.py`, suite v1) is run on:
- the baseline: the pinned v2 start checkpoint → `evals/v2_iter173600.json`
  (`evals/v2_iter172000.json` was taken when the suite was built and is kept for reference)
- v3 every 50M steps (~2 h at current throughput), named `v3_<steps>M`

and compared with `--compare`. Differences inside the seed spread are not differences.

**Primary** (decides the run): goal difference per 10 min against Necto.
**Guardrails** (must not collapse): touches/min, kickoff first-touch %, shot conversion.
**The problems v2 had** (should improve, and are where v3 is expected to earn its keep):
retreat boosting %, retreat-dodge rate and landing quality, `ran_under_ball_pct` and first touch in
the drop/bounce scenarios, conceded % in the retreat scenario, goals against caused by back-wall
climbs and being caught upfield.

**Decision points:**
- **250M steps:** if the primary metric is clearly worse than v2 (non-overlapping seed ranges) *and*
  falling, stop and review. No mid-run fixes either way.
- **500M steps:** adopt v3 if the primary metric is at least v2's within the noise and the
  v2-problem metrics are better. Otherwise its results inform v4 — a single, measured addition
  (one term or one scenario) chosen from the eval, not from a log.

## 7. Implementation (after this spec is agreed)

- `env/rewards_v3.py`: the five terms and a combiner, target < 300 lines. v2's `env/rewards.py` is
  not edited (its hash is under test). Shared helpers v3 needs are imported read-only or copied.
- `env/reward_registry.py`: picks v2 or v3 from a new `reward_version` config key; `RewardManager`
  construction goes through it. `env/reward_version.py` hashes the active version's files.
- `config/reward_versions/v3.json`: the frozen settings above; the run's config points at it.
- Two training setters, `BounceDropSetter` and `RetreatSetter`, in `env/state_setters.py`.
- Trainer: return-normaliser reset + critic warm-up on a version change; anneal clock for T4.
- Tests: each term's formula; potential telescoping (any closed loop sums to ~0); `Φ(terminal)=0`
  on goals; action-independence (R2); v2 still frozen.

## 8. Parking lot (candidates for v4, only if the v3 eval shows the need)

- Landing quality / speed retained after a flight, as a potential on horizontal speed toward the
  ball (never paid on the dodge).
- A save-race potential: time margin between the car and the ball reaching our goal.
- A height potential near an airborne ball, if aerial touches regress.

Each would enter alone, in its own version, justified by an eval metric.
