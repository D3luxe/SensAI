# Reward v7 — v5 with a cleaned, situation-weighted replay pool

Status: **agreed and implemented 2026-09-21; rebased on v5 2026-09-22** when v6 closed without being
adopted (`docs/reward_v6_align_spec.md` §6). Frozen as `config/reward_versions/v7.json` (code
`6f7bbc1ec86ffaae`, settings `e9d24c1d25cf5da0`); the replay pool is frozen into it before the run
starts (§4). Drafted at v6 ≈100M steps, so §1 describes v6 as it stood then. **v7's reward is v5's,
unchanged** (v3's terms at γ 0.9977; the align term is not carried over), and it starts from the v5
king. The one variable is **which replay frames training starts from**. The rules of
`docs/reward_v3_spec.md` §2 (R1–R7) apply. This is a reward *version* rather than a bare training
run because the scenario mix is part of the frozen settings (`SETTINGS_SECTIONS`), and the replay
pool now becomes part of the frozen identity too (§4).

## 1. What v6 showed, and why the starts

At ~100M steps v6 is ahead of v5 at the same point on every measure, and head to head +7.25 ± 2.0
(9/9 seeds) against the v5 king. What it has not fixed shows up in the goals-against causes, pooled
over its three ~100M checkpoints (goals per 10 min vs Necto, not shares, since total conceded moved):

| cause | v6 ~100M | v5 400M reference |
|---|---|---|
| caught upfield | 13.2 | 14.1 |
| back-wall climb | 2.7 | 7.1 |
| **beaten while goal-side** | **7.5** | **3.6** |

T6 moved SensAI into position; it now loses the duel from there. Watching SensAI against Necto, the
pattern is specific: kickoffs are even, then Necto takes control, and every challenge either meets
a flick out of a dribble or a pop over the car. On-target shots per 100 touches (2.3, reference
6.5) fit the same picture: SensAI's touches are scrambles, not possession.

That is a mechanical deficit, and v4 already showed a reward term does not fix one (paying for the
loose-ball race for 500M steps did not move first-touch rates). R7's order puts scenarios before
terms. The starts are where the time goes, and they are not what they look like:

| start (v6 mix) | share of all episodes |
|---|---|
| kickoff scenario | 15% |
| **replay frame that is a kickoff countdown** | **8.6%** |
| **replay frame taken after a goal** | **4.1%** |
| replay frame from live play | 32% |
| synthetic scenarios (aerial 8, save 6, turnaround 5, wall 5, wall rebound 5, bounce-drop 4, retreat 4, dribble-flick 3) | 40% |

Every post-goal frame puts the ball in the goal mouth moving into the net (the setter clamps y to
±5050): the episode ends in a goal within a tick or two and no decision is made. And within live
play, the moment that decides the Necto games is nearly absent:

| situation in live replay frames | share | distinct moments |
|---|---|---|
| **carry under challenge** (ball on a car, other car within 1000) | **0.5%** | ~2,100 |
| carry, uncontested | 2.1% | ~7,600 |
| aerial (ball above 300, a car airborne within 800) | 22.9% | ~7,500 |
| challenge (both cars within 1000 of the ball) | 19.6% | ~15,300 |
| goal threat (ball in a final third, moving at that goal) | 13.2% | ~8,900 |
| open play (none of the above) | 41.7% | ~20,800 |

Tags are exclusive and take the first match in that order. A "moment" is a run of consecutive
tagged frames in one replay; neighbouring frames at 30 Hz are near duplicates, so moments, not
frames, are the real sample size.

## 2. The change

**v7 = v5's reward, unchanged, with three changes to where episodes start.** (Drafted as v6's
reward; v6 was not adopted, so v7 changes the starts of the version that was.) (a) is a data fix;
(b) is the experiment; (c) makes (b) usable.

### (a) Prune dead frames — when the pool is loaded

A frame is never a start if:

| rule | frames removed |
|---|---|
| ball at centre and still (kickoff countdown) | 19.2% |
| ball beyond a goal line, \|y\| > 5120 (post-goal) | 9.1% |
| either car's last position update more than 0.5 s old | 12.1% |
| a car at the spawn point (demolished, or not yet replicated) | 0.2% |
| **total (overlapping)** | **29.9%** — 1.40M of 2.00M frames kept |

*Pruned at load, not at ingest (changed from the draft).* The pool also feeds behaviour-cloning
pretraining (`agent/pretrainer.py`), which pairs consecutive frames into transitions; cutting frames
out at ingest would put gaps inside its segments. So the pool keeps every frame, and v7 builds an
index of the kept ones when it loads, cached beside the pool (§3). The effect on training starts is
identical.

The replay parser rejects a replay whose header `TeamSize` is above 1. Every replay in the pool is
1v1 today, but the parser keeps "the first car of each team" and would silently produce a teammate
pair from a team replay.

**Kickoff exposure is held constant.** The countdown frames were kickoffs in all but name, so
removing them would cut kickoff practice from ~24% of episodes to 15%, a second variable.
`kickoff_prob` goes 0.15 → **0.24** and `replay_prob` 0.45 → **0.36**; the replay share that
remains is all live play (it was 0.32 live + 0.04 post-goal). The synthetic scenarios keep their v5
weights exactly.

### (b) Sample replay starts by situation

At load, every kept frame gets one tag (§1's table, same rules). A replay start picks a tag by
weight, then a frame uniformly within it.

| tag | pool share | **v7 weight** |
|---|---|---|
| carry under challenge | 0.5% | **5%** |
| carry | 2.1% | **7.5%** |
| aerial | 22.9% | **17.5%** |
| challenge | 19.6% | **30%** |
| goal threat | 13.2% | **20%** |
| open play | 41.7% | **20%** |

The weights move replay starts toward contested, decisive play without starving the rest. The
carry-under-challenge weight is a ten-fold boost but deliberately capped: at 5% of 36% it is ~1.8%
of all episodes, ~18,000 draws per million episodes spread over ~8,400 variants (~2,100 moments ×4
with (c)'s mirroring), so each variant recurs a couple of times per million episodes. Pushing it higher would replay the same few thousand situations; if it
proves too thin the answer is more replays (§5), not a larger weight.

### (c) Mirror at sample time

Each replay start is independently:
- **flipped left–right** (x → −x; yaw → π − yaw, roll → −roll) with probability ½, and
- **team-swapped** (rotated 180° about the vertical, x → −x and y → −y, yaw → yaw + π, and the two
  cars exchanged) with probability ½.

The team swap matters beyond variety. Against the league and Necto the learner is always the blue
car, so today it only ever plays the blue human's side of a replay frame. With the swap, a
carry-under-challenge frame puts SensAI on either end of it: carrying, or defending the carry,
which is the drill the Necto games call for.

Both transforms are exact symmetries of the pitch, so they cannot create an impossible state.

## 3. Memory, and how much room there is for more replays

**Every arena loads its own copy of the pool.** `RocketSimArena` builds a `WeightedScenarioSetter`,
which builds a `ReplayParser`, which decompresses the whole pool into its own arrays: 298 MB, ×128
arenas ≈ **38 GB**. That is most of the 70 GB the training workers have committed (16 workers at
~4.1 GB private each, for 8 arenas each), and the machine is at 122.5 GB committed of a 158.9 GB
limit. Under the current design, each additional million frames costs ~19 GB of commit, so the pool
could grow by only ~1.5M frames before training hits the commit limit.

v7 fixes this before it adds anything:
- The pool is stored as uncompressed `.npy` arrays (`data/replays/replays_pool.store/<size>_<mtime>/`),
  built from the `.npz` the first time it is loaded and opened with `mmap_mode="r"`. The operating
  system keeps **one** copy in its page cache, shared by every worker and arena. An ingest writes a
  new `.npz`, so the next load builds a new store beside the old one (Windows will not replace a file
  a running trainer has mapped); old stores are removed once nothing holds them.
- One pool per process (a module-level cache keyed by path, mtime and size), and v7's index is
  cached in the store directory and mapped too.
- Pools under 32 MB on disk are read into memory as before: a store is not worth writing for them,
  and it keeps the unit tests' throwaway pools deletable on Windows.

Measured on the real pool: eight parsers (one worker's arenas) with their v7 samplers add **0 MB**
of private memory, against ~298 MB per parser before. The first load builds the store and index
in ~4 s; later loads take milliseconds.

After the fix the pool costs its own size once: 156 bytes per frame. `MAX_POOL_FRAMES` goes 2M →
**14M** frames (~2.2 GB), which is ~10M kept after pruning and ~1,300 replays at ~11,000 frames
each. The cap counts all frames, since pruning happens at load.

Replays are ingested **before** a run starts, never during one: the arenas read the pool once, at
startup, so an ingest changes nothing until the next restart, and then it changes the starts
mid-run (R1). For v7 the trainer enforces this (§4).

## 4. What is frozen

- `config/reward_versions/v7.json`: v5's rewards and γ, the scenario weights of §2(a), and a new
  `replay_sampling` block: the tag weights, the mirroring probabilities, and the pruning thresholds.
- **The pool is part of the identity.** The snapshot's `replay_pool` records the frame count, the
  kept frames per tag, and a hash over every frame's state and the index. The trainer checks it at
  startup, before any worker starts, and refuses to run v7 on any other pool, or on none frozen.
  Replays are added **before** v7 starts, then frozen with `scripts/freeze_replay_pool.py --version
  v7`, which prints the per-tag counts as the check that the pool is the one meant.
- Code: v7's `CODE_FILES` are v5's (`env/rewards_v3.py`, `env/scenarios_v3.py`) plus
  `env/replay_sampling_v7.py` (pruning, tagging, weighted sampling, mirroring).
  `env/state_setters.py` and `utils/replay_parser.py` gain a switch and the memory-mapped store, but
  the uniform path every earlier version uses samples exactly as before, bit for bit, and a test
  pins that.

## 5. The run

- **Start:** the v5 king, `checkpoints/baselines/v5_iter222000.pt`. v6 was not adopted, and its best
  stretch (350–400M) is not used as a start: those checkpoints were picked for evaluating well, the
  winner's-curse reference v5's judging rules out. They stay pinned.
- **Before starting:** `scripts/archive_run.py` moves v6's checkpoints out of the way (numbering
  restarts from 222000); ingest any new replays (1v1; higher rank is better, since carries and
  flicks are more common there); then freeze the pool.
- Version change v5 → v7: return normaliser reset, fresh optimizer, 50 iterations of critic
  warm-up. The reward and γ are v5's, so explained variance should recover within the warm-up.
- **Nothing is changed during the run.**

## 6. How v7 is judged

Pooled over all checkpoints evaluated at a decision point (at least three, adjacent), as v6.
Results named `v7_<steps>M`, `--reference checkpoints/baselines/v5_iter222000.pt`.

**Reference: the v5 400M group, pooled** (222000, 222200, 222400, 222600 — the table v6 was judged
against), with its own checkpoint range, which is what "clearly" means below:

| metric (vs Necto unless named) | reference | checkpoint range |
|---|---|---|
| goals for /10 min | 2.9 | 1.2 – 4.2 |
| goals against while goal-side /10 min | 3.6 | 2.0 – 5.6 |
| on target /100 touches | 6.5 | 5.1 – 8.3 |
| touches /min | 7.5 | 7.1 – 7.8 |
| kickoff goals against | 6.4 | 2.7 – 11.3 |
| caught-upfield goals against /10 min | 14.4 | 8.7 – 18.4 |
| Nexto goal difference /10 min | −27.4 | −33.0 – −23.2 |
| first touch: drop / bounce / wall scenario | 26% / 36% / 78% | |

**Primary:** head to head against the v5 king. Per-seed sd at v5's 400M was 6.7.

**Where v7 must earn its keep** (what the new starts are for):
- goals conceded while goal-side, and Necto goals for (the duel, and possession);
- on target per 100 touches;
- first-touch rate in the drop, bounce and wall scenarios.

**Side metric, boost economy (added 2026-09-21).** SensAI has run on empty for 40–48% of every
game against Necto since v3, spending ~280 boost a minute while supersonic only ~2.5% of the time.
The eval now reports, vs Necto: boost collected, small and big pads per minute, the share of boost
spent while already supersonic and while airborne, and the share of retreats begun with under 12
boost (the "Boost economy" cards). Not a gate. v7's replay starts carry the human players' boost
(the pool averages 42), so if time on empty falls under v7 the bot misuses boost it has; if it does
not, the gap is collection. Either way it decides what a later version's boost change targets. v5's
evals predate these metrics; v6's carry them from 193M on (at 400M: 42% on empty, ~1 big pad a
minute, 60% of retreats begun under 12 boost), which is the nearest comparison.

**Guardrails** (must not fall outside the reference's range on the wrong side): kickoff goals
against (kickoff exposure is held constant, so it should not move), touches per min, Nexto goal
difference, caught-upfield goals against.

**Decision points:**
- **150M:** stop and review if the pooled head to head is clearly negative (below −2 standard errors),
  or a guardrail is clearly broken. Passivity is judged against **v5 at the same step count**, not
  its 400M numbers: v5 at 150M read 5.5 touches/min and 0.7 goals for, and went on to be adopted (the
  v6 run showed the 400M comparison stops a healthy run early).
- **400M:** adopt on the criterion below.

The head to head alone cannot adopt (v4, the learning-rate run and v6's last 100M all beat their
ancestor while getting no better against Necto). These gates are written before the run.

### The 400M criterion (changed 2026-09-22 at ~273M steps, before any eval past that point)

*What changed and why.* The gate as first written required the head to head to be **positive** by
more than two standard errors. Two things since say that is the wrong test. First, every eval of
this run is now automated (`scripts/auto_eval.py`), so a decision point pools ~20 adjacent
checkpoints and ~60 seeds instead of three checkpoints and nine: the head-to-head readings that
drove the 150M (+7.17) and 200M (−8.92) reviews were the extremes of that sampling, and the pooled
series shows head to head flat near zero from 50M on. Second, the head to head measures how well a
checkpoint beats **its own ancestor**, and v4, the learning-rate run and v6's last 100M each did
that while getting worse at the game. Necto and Nexto never change and never train, so they are the
only measure of the game itself (`docs/run_v5_lr_spec.md` §4).

So a run may be adopted while level with its ancestor, provided it is **better against the fixed
opponents**. All three conditions must hold, pooled over every checkpoint within ±25M of 400M:

1. **Not clearly worse than its ancestor.** Pooled head to head at or above −2 standard errors of
   its own seeds.
2. **Clearly better against a fixed opponent.** Necto **or** Nexto goal difference better than the
   v5 400M reference by more than twice the combined standard error. With the reference's 12 seeds
   and ~60 of v7's, that is **Necto better than −26.2** or **Nexto better than −23.0** (reference:
   −29.4 and −27.4).
3. **No guardrail clearly worse** by the same test: Necto goals for (2.94), touches per min (7.49),
   kickoff goals against (6.42), retreat scenario conceded (27.1%).

*Where the run stood when this was written,* pooled over the 7 checkpoints from 248M to 272M, in
combined standard errors against the reference: Necto goal difference −33.4 (**−2.3**), Nexto −32.4
(**−2.1**), Necto goals for 1.39 (−2.8), touches 6.55 (−3.7), kickoff goals against 12.4 (**+3.9
worse**), retreat conceded 42.9% (+2.6 worse). On that window v7 fails conditions 2 and 3, and the
criterion is recorded here **before** the evals that will decide it, exactly so that it cannot be
fitted to them afterwards.

## 7. Implementation

Built in a separate git worktree while v6 trained, so no file the live run imported was touched,
and merged when v6 stopped.

- `env/replay_sampling_v7.py`: `prune_mask`, `tag_frames`, `build_index` and its cache,
  `mirror_x`, `swap_teams`, `ReplaySampler`, `pool_fingerprint`, `check_replay_pool`.
- `utils/replay_parser.py`: the `TeamSize` guard; the memory-mapped store and per-process cache;
  `MAX_POOL_FRAMES` 14M; `fit_car_count` shared with the sampler.
- `env/state_setters.py`: `ReplayStateSetter.configure()` switches to the sampler;
  `WeightedScenarioSetter.update_weights` passes it the `replay_sampling` block.
- `env/reward_registry.py`: the v7 entry, `OPTIONAL_SETTINGS_SECTIONS` (hashed only when present, so
  v2–v6's identities are untouched), and `scenario_payload`, which the trainer and
  `scripts/policy_health.py` send the environments. `agent/ppo.py` checks the frozen pool.
- `config/reward_versions/v7.json`; `config/default_config.yaml` names v7.
- `scripts/freeze_replay_pool.py`.
- Tests:
  - pruning rules on synthetic frames, one per rule;
  - tags on hand-built frames, and exclusivity in priority order;
  - sampled tag frequencies match the weights (each within 5 standard errors at a fixed seed);
  - each mirror is an exact symmetry: a mirrored state stepped in RocketSim matches the mirrored
    trajectory of the original;
  - the uniform path (v2–v6) returns the same frames as before for a fixed seed;
  - many arenas in one process share one buffer; the `.npy` store round-trips the `.npz`;
  - a missing or changed frozen pool refuses to run;
  - team replays are rejected at ingest;
  - v7's rewards, γ, code and synthetic scenario weights are v5's, it starts from the v5 king, and
    only v7's scenario payload carries replay sampling;
  - the eval's boost-economy counters (`test_eval_boost_economy.py`).
  (`test_replay_sampling_v7.py`, 22 tests; full suite 807 passing.)

## 8. Outcome

*(to be written when the run closes)*
