# Behavioral cloning replay pipeline: defects to fix before the next clone

Status, branch `fix/bc-replay-pipeline`:

| defect | status |
|---|---|
| E. Pool sampled at 3 Hz, solver told 30 Hz | **fixed** (full-rate parsing, real per-car dt; legacy pools get 10/30 s) |
| F. Quaternion to Euler conversion mirrored pitch and roll | **fixed** |
| G. Pretrainer observations built with a mirrored right vector | **fixed in the pretrainer**; the shared helpers are a separate task |
| D. Flip-cancel heuristic hard-writes pitch on wall frames | **fixed** (deleted) |
| A. Euler differencing used as angular velocity | **fixed** |
| C. Height-only contact test | not fixed |
| B. Ground steer computed about world Z | partly: steer now reads `omega_body · up_car` as a side effect of A, but wall frames still never reach the ground branch until C lands |

None of this affects a running policy. It only takes effect on a re-clone, so it should be done
before the next fresh start rather than mid-run. **The existing `replays_pool.npz` carries E and
F baked in and must be re-ingested**, not just re-cloned.

Net effect so far on imputed pitch (legacy pool as the pretrainer used it, against a full-rate
pool re-parsed from the same 225 replays with E, F, D and A fixed):

| car state | before: mean abs pitch, exactly +1.0 | after: mean abs pitch, exactly +1.0 |
|---|---|---|
| floor, z under 25 | 0.011, 0.3% | 0.004, 0.0% |
| z 25 to 200 | 0.557, 20.1% | 0.258, 5.0% |
| wall, z over 200 | 0.796, 40.3% | **0.200, 4.1%** |

The remaining wall and ramp mass is mostly C: those frames are still routed to the airborne
branch.

Context: Rocket League replays store physics, not inputs. Of the eight actions the game
takes, replays provide only throttle, steer, handbrake, jump and boost. Pitch, yaw and roll
are absent and must be imputed, and whether the wheels are touching a surface is not recorded
either. Seer's authors hit exactly this and solved the contact problem by training a
histogram-based gradient-boosting classifier on manually collected driving data, then imputed
the missing rotational actions on top of carball's approximations.

We took a cheaper route on both and it is wrong in several independent ways. All figures below
were measured, not estimated.

Correction to the premise: replays do record **angular velocity**. Every car `RigidBody` update
carries `angular_velocity`, in world axes, directionally exact against the rotation stream
(median cosine 1.000 over 12,642 pairs). Its scale is a raw network unit, roughly x80 to x100
with a noisy fit, so it is stored in the pool as `car_ang_vel` but not yet consumed by the solver.

E, F and G were found while fixing D and A, and each outweighs D.

---

## E. The pool was sampled at 3 Hz and the solver was told 30 Hz

**The largest single defect.** `_extract_replay_binary` kept every 10th network frame. Replays
record at 30 Hz (`RecordFPS` 30, per-frame `delta` 0.0333 s), so pool frames were 0.333 s apart
(position-vs-velocity estimate: median 0.325 s). The pretrainer passed `dt=1/30`, inflating every
imputed rate and acceleration about 10x.

Consequences beyond the rate scale:

- 0.33 s is too coarse to impute stick inputs at all: jumps, dodges and flip cancels merge.
- The 250 uu continuity filter at 0.33 s spacing admits only cars slower than about 760 uu/s, so
  fast play was silently dropped.
- Even at 30 Hz, a car's physics is only replicated every 2 to 3 frames (median gap 0.067 s, 90th
  percentile 0.1 s). Consecutive pool frames are often a carried-over duplicate, so a fixed `dt`
  is wrong even at full rate.
- The old 100,000-frame cap held about 18 minutes of play at full rate.

Pitch on the legacy pool, D and A applied, before and after passing the true spacing:

| car state | dt 1/30: mean abs pitch, exactly +1.0 | dt 1/3: mean abs pitch, exactly +1.0 |
|---|---|---|
| floor | 0.011, 0.3% | 0.002, 0.0% |
| z 25 to 200 | 0.595, 18.2% | 0.259, 9.4% |
| wall | 0.721, 40.3% | 0.437, 22.1% |

**Fix (done):** parse every frame (`ReplayParser.frame_stride = 1`) and store `frame_time`,
per-car `car_time` of the last physics update, `car_ang_vel` and `segment_id`. The pretrainer and
the audit only label frames where the car was actually replicated (`is_fresh_car_update`) and
pair each with that car's next update in the same recording (`find_car_transition`), using the
real gap, rejecting gaps over 0.15 s and teleports. Legacy pools without timing fall back to the
old 250 uu filter with `dt = 10/30`. Pool cap raised to 2,000,000 frames; the 225 replays produce
2.46M, so the oldest 19% are currently dropped.

---

## F. The quaternion to Euler conversion mirrored pitch and roll

`_quat_to_euler` used the aerospace ZYX formulas, which do not match the Euler convention that
RocketSim's `rsim.Angle`, the solver and `CarState` all rebuild orientation from. Yaw survived
(forward matched velocity on the floor), but pitch and roll came out mirrored: on side walls the
rebuilt up vector pointed **into** the wall (median -1.00, where the quaternion gives +1.00), and
rotation-derived rates disagreed with the quaternion path by 173%.

This corrupted `car_rot` for every consumer, including `ReplayStateSetter`, which feeds pool
angles to `rsim.Angle(...).as_rot_mat()` to spawn training episodes.

**Fix (done):** read pitch, yaw and roll off the quaternion's rotation matrix: `pitch =
asin(fwd_z)`, `yaw = atan2(fwd_y, fwd_x)`, `roll = atan2(-right_z, up_z)`. Verified: rebuilding
the basis from the angles matches the quaternion matrix to 3e-5, wall up vectors point away from
the wall (+1.00 over 804 frames), and the basis matches `rsim.Angle` exactly.

---

## G. Pretrainer observations used a mirrored right vector

RocketSim's right vector is `up x forward`. `CarState.get_right_vector()` and
`bot.rotation_to_rot_mat` both compute `forward x up`, the negative. In training the env builds
cars from RocketSim's `rot_mat`, but the pretrainer built replay cars from `rot` alone, so every
lateral observation during cloning was mirrored relative to what the policy sees in RL.

**Fix (done in the pretrainer):** replay cars are given `rot_mat` rows in RocketSim's convention.
The two shared helpers are left unchanged here because the live bot and rewards also call them;
correcting them needs a caller-by-caller audit of which signs were tuned against the mirror.

---

## D. The flip-cancel heuristic hard-writes full pitch on wall frames

**Cheapest to fix, but smaller than first thought.** Deleting it alone moved wall frames at
exactly +1.0 pitch from 40.3% to 34.5%; most of the saturation came from E.

`utils/inverse_dynamics.py`, in the airborne branch:

```python
if up[2] < 0.2 and abs(pitch_rate) < 2.5 and (a_fwd < -200.0 or speed_fwd < -300.0
                                              or (speed_fwd > 300.0 and abs(fwd[2]) < 0.5)):
    pitch_act = 1.0
```

Its stated purpose is detecting a player holding the stick to cancel a backflip. A car driving
horizontally along a vertical wall satisfies it by accident: the up vector points into the
arena so `up[2]` is near 0, `abs(fwd[2]) < 0.5` holds, and any forward speed above 300 closes
the condition. It then overrides whatever the measured rate said.

Measured over 30,000 replay frames, running the real solver:

| car state | mean abs pitch | above 0.9 | exactly +1.0 |
|---|---|---|---|
| floor, z under 25 | 0.011 | 0.5% | 0.3% |
| z 25 to 200 | 0.557 | 41.3% | 20.1% |
| wall, z over 200 | 0.796 | 70.2% | **40.3%** |

So cloning teaches "car is high off the floor, hold full pitch." For comparison the trained
policy currently sits at 0.72 mean abs pitch with 37% saturated, so this is not a small
contributor to the tumbling.

**Fix:** delete the heuristic rather than repair it. A rule that infers a specific human stick
input from a coincidence of geometry cannot be made reliable, and correct angular velocity
(see A) makes the real flip-cancel case visible without guessing.

**Caution on the aggregate.** The dataset-wide pitch average is 0.254 with 11.7% saturated,
which looks harmless and is misleading: it is dominated by floor frames where the ground
branch writes exactly zero. Always condition on car height when auditing this.

---

## A. Euler differencing is invalid everywhere, not just where the contact test is wrong

`measured_omega = (rot_next - rot_t) / dt` treats Euler angle differences as body angular
velocity. They are not the same quantity, and the construction degenerates near pitch = ±90°.

This is **not** only a wall problem. A genuinely airborne car nose-up for an aerial, a ceiling
approach, or a wall-to-air takeoff sits at the same singularity.

Measured pitch rate from this construction, against a 5.5 rad/s physics cap:

| frames | mean | 95th percentile |
|---|---|---|
| floor | 0.22 | 0.23 |
| z 25 to 200 | 7.68 | 26.45 |
| wall, z over 200 | 10.34 | 28.85 |
| near gimbal, abs(euler pitch) over 1.3 rad | 13.18 | 39.93 |

Rates up to seven times what the game permits, which then divide by the torque constant and
clip to ±1.

**Fix:** build the relative rotation `R_rel = R_t^T · R_t+1`, convert to axis-angle, and divide
by dt. Singularity-free everywhere in the arena. The body-frame components drop straight onto
the pitch, yaw and roll axes.

---

## B. The ground branch computes steer in the world frame

```python
yaw_rate = float(measured_omega[1])                      # rotation about world Z
steer_val = -(yaw_rate * 500.0) / max(200.0, abs(speed_fwd))
```

On the floor this is fine. On a side wall the steering axis is the wall normal (world X), not
world Z, so steer comes out near zero when turning up or down the wall and unstable when
turning across it.

This matters because it **invalidates the obvious fix for the contact problem**. Simply routing
wall frames into the ground branch so pitch is zeroed would trade a bad pitch label for a bad
steer label.

**Fix:** project body angular velocity onto the car's own up vector, `omega_body · up_car`,
rather than reading the world Euler yaw rate.

---

## C. No flat-plane distance threshold can detect surface contact

Current test, `agent/pretrainer.py`:

```python
on_ground=(car_p[2] < 25.0)
```

This labels **50.4%** of replay frames airborne. Real 1v1 air time is nearer 25%. Of the
mislabelled frames, 18.4% are cars driving on walls (z over 200) and 32.0% sit in the
25-to-200 ramp band.

The natural repair, distance to the nearest flat plane under some threshold, also fails.
Tested against RocketSim's own contact flag across 470 genuinely grounded steps while
climbing a wall:

| flat-plane threshold | truly grounded steps wrongly called airborne |
|---|---|
| 25 uu | 17.2% |
| 40 uu | 14.9% |
| 60 uu | 12.8% |
| 80 uu | 10.6% |
| 120 uu | 8.1% |

Mean distance to the nearest flat plane **while genuinely grounded** is 43.9 uu, 90th
percentile 88.5, maximum 562.2. Raising the threshold does not rescue it; it just trades false
negatives for false positives. The floor-to-wall fillet radius here is nearer 256 uu than the
1000 sometimes quoted, but the conclusion does not depend on that number.

**Fix options, in order of preference:**
1. Model the actual collision geometry including the floor/wall fillets and the 45° corner
   bevels (`CORNER_OFFSET = 1152`, `CORNER_LIMIT = 8064` in `env/physics_engine.py`), and test
   suspension or wheel-contact distance.
2. Seer's approach: train a small classifier on labelled driving data.
3. Weakest acceptable fallback: combine a generous surface distance with the constraint that
   velocity along the surface normal is near zero, which rules out most true flight.

---

## Remaining order

E, F, G, D and A are done. C is next and is the largest remaining piece; B lands with it. After
that, consider consuming the recorded `car_ang_vel` directly once its scale is pinned down.

Re-audit after each step by conditioning the imputed action distribution on car height. The
floor row should stay near zero, and the wall row should keep coming down.

## Reproducing the measurements

`python scripts/audit_imputed_actions.py --pool <pool>` reproduces the height-bucketed tables.
It uses the pretrainer's own pairing (`find_car_transition`), so timed and legacy pools are both
handled; `--dt` overrides the spacing. The first tables in D and A were measured on the legacy
pool with `dt=1/30`, matching what the pretrainer did at the time.

To rebuild a full-rate pool from the replay folder (about a minute for 225 replays):

```python
from utils.replay_parser import ReplayParser
p = ReplayParser(pool_path="data/replays/replays_pool_30hz.npz",
                 demo_dir=r"C:/Users/coryf/OneDrive/Documents/RL_ML_Training/replays")
p.clear_pool()
p.ingest_directory(max_replays=0, sort="oldest")
```

`rrrocket.exe` is auto-downloaded into `bin/` on first use. The contact ground truth in C comes
from driving a scripted car into a wall in `RocketLeagueEnv` and reading `car.on_ground`, which
is RocketSim's own flag.

Note when writing probes here: assigning to `car.pos` or `ball.pos` does nothing, because
RocketSim owns the state and overwrites it on the next step. Scripted scenarios must be
reached through actions, or through the scenario setters.
