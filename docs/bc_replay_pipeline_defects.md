# Behavioral cloning replay pipeline: four defects to fix before the next clone

Status: **not fixed**. None of this affects a running policy. It only takes effect on a
re-clone, so it should be done before the next fresh start rather than mid-run.

Context: Rocket League replays store physics, not inputs. Of the eight actions the game
takes, replays provide only throttle, steer, handbrake, jump and boost. Pitch, yaw and roll
are absent and must be imputed, and whether the wheels are touching a surface is not recorded
either. Seer's authors hit exactly this and solved the contact problem by training a
histogram-based gradient-boosting classifier on manually collected driving data, then imputed
the missing rotational actions on top of carball's approximations.

We took a cheaper route on both and it is wrong in four independent ways. All figures below
were measured, not estimated.

---

## D. The flip-cancel heuristic hard-writes full pitch on wall frames

**Worst of the four, and the cheapest to fix.**

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

## Suggested order

D first, on its own, since it is a deletion and the effect is large and isolated. Then A,
which is self-contained and fixes both the wall and the genuine-aerial cases. Then C, the
largest piece of work. B only matters once C routes wall frames into the ground branch, so it
lands with C.

Re-audit after each step by conditioning the imputed action distribution on car height. The
floor row should stay near zero, and the wall row should come down from 0.796 toward it.

## Reproducing the measurements

The probes used for all of the above lived in the session scratchpad and were not kept. Each
is short: load the replay pool with `ReplayParser.load_pool()`, sample consecutive frame pairs
with the same 250 uu continuity filter `agent/pretrainer.py` uses, call
`InverseDynamicsSolver.solve_car_action` directly, and bucket the output by `car_pos[2]`. The
contact ground truth in C comes from driving a scripted car into a wall in
`RocketLeagueEnv` and reading `car.on_ground`, which is RocketSim's own flag.

Note when writing probes here: assigning to `car.pos` or `ball.pos` does nothing, because
RocketSim owns the state and overwrites it on the next step. Scripted scenarios must be
reached through actions, or through the scenario setters.
