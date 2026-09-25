# Rebuild plan: SensAI on Prometheus

Status: **agreed 2026-09-24; Phases 0, 1 and 2's gate passed 2026-09-24; Phase 2 complete; Phase 3's run 1 ended 2026-09-25 without adoption (a fresh policy jumps half the time, so ground controls never trained); a grounded start fixed that in `headcheck2_1v1`, but no continuous run (run 1, head checks 1–4) learned to steer toward the ball; next is `discrete_1v1`, a discrete teacher for the continuous head (user's call 2026-09-25).** SenseiBot's trainer is retired after v11
(`docs/reward_v11_pretanh_spec.md` §8). Training moves to a new workspace, **`C:\Users\coryf\antigravity\SensAI`**,
built from Prometheus (https://github.com/mitige/prometheus, reviewed at commit `e4d097d`; upstream HEAD
re-checked 2026-09-24 and unchanged). SensAI Studio (`ui/`), the evaluation suite and the probes move to
SensAI with it; Prometheus's own ImGui GUI is not built. This repo becomes the archive.

## 1. Why rebuild

v5 was adopted at ~3.6B steps. v6 through v11 then spent ~2B more steps on single variables against
v5's architecture and trainer, and none was adopted:

| run | steps | what it established |
|---|---|---|
| v6 | ~507M | Potential terms cannot change the optimal policy; v6 was v5 trained for longer, and plateaued. |
| v7 | ~413M | The start mix moves behaviour; forced contested starts caused v7's second-half regression. |
| v8 | ~227M | A reward on acquiring boost is cycled (collect → dump → collect). |
| v9 | ~156M | A reward on holding boost is hoarded; the bot stopped spending where it mattered. |
| v10 | ~523M | The clean control: v5's reward trained longer does not beat v5, under either start mix. |
| v11 | ~162M | **The first change to move the plateau, and it was structural:** fixing the steer head's dead gradient cut whiffs 57 → 35 per 100 touches and gave the best Necto GD since v5 (−23.7). |

Five of six runs changed rewards or starts; the one structural change was the one that clearly helped.
Meanwhile three outside references point at structure and scale, not reward detail:

| | Seer (thesis, 2022) | GigaLearn (default) | Prometheus | SenseiBot |
|---|---|---|---|---|
| actions | discrete (1,800) | discrete table (~90) | **continuous, tanh-squashed Gaussian** | continuous, Gaussian around tanh |
| network | MLP + **LSTM**, 2.0M params | MLP 256-256 + 256×3, ~0.3M | **attention** (256 dims, 4 heads, 2 blocks) + MLPs 1024×3 | MLP 256-256-128, 0.26M |
| previous action in obs | yes | yes | yes | **no** |
| physics | game | RocketSim CPU (C++) | **RocketSim on GPU**, 1,024 arenas | RocketSim CPU (Python), 128 envs |
| batch / minibatch | 27,648 / — | — | **200,000 / 20,000** | 16,384 / 1,024 |
| scale quoted | ~10B steps (≈ Platinum I) | — | teacher checkpoint at ~28.2B steps | ~5.6B total |

Prometheus is the closest match to what SensAI wants: continuous actions without the saturation
failure, attention, previous action in the observation, large batches, GPU physics, and a built-in
teacher→student transfer path. Its GPU physics looked like the step change in throughput on an RTX
5090. Phase 0 measured otherwise: C++ RocketSim on the CPU (24 cores) is as fast or faster for our
setup, so SensAI trains on CPU physics and uses the 5090 for the network (§6, 2026-09-24).

## 2. What carries over

**Lessons that bind the new reward design** (each is a measured result, not a preference):

1. **No potential-based terms for behaviour.** They telescope to a start-state constant and cannot change
   the optimal policy (v6 correction, v7 §8 finding 2).
2. **Never price a change in one's own boost.** Any term that pays the rise can be cycled (v8), and a
   discounted fall only scales the exploit (v8 §7 finding 4).
3. **Never price holding a resource.** A level term is hoarded, and the paid-for metric passes while
   the game gets worse (v9 §6).
4. **A dense always-on non-negative bonus is a survival bonus** unless zero-sum (v9 §2). Size is a
   statement about the optimum, not about what the learner finds first (v9 §6 finding 2).
5. **Ball position is not a good-play signal.** In 781 minutes of human 1v1, an advanced ball predicts
   *conceding* 4–12 s later (AUC 0.386); possession and territory score at chance (v7 §8 finding 5).
6. **No reward reads the action** (standing rule R2).

**Method that carries over unchanged:**

- One variable per run; criteria written before the run; a clean control when comparisons get murky.
- Pool evals over ±25M of checkpoints; "clearly better" means more than twice the combined standard
  error.
- Probe readings drive decisions only when pooled over several checkpoints and at least six seeds
  (v11 §7, ~105M).
- Point every canary at the quantity the change prices, and never accept the paid-for metric as
  evidence it worked (v8 lessons 2 and 3).
- Nexto is the held-out opponent; Necto has been a training opponent (18.75%) and reads optimistic.
- Archive each run and reset the league at every transition so runs never train against another
  run's checkpoints (v9 §6 finding 7).

**Assets that move to SensAI** (Phase 1 step 1): SensAI Studio (`ui/`, `utils/process_manager.py`,
`utils/config.py`, `utils/run_history.py`, `utils/diagnostics.py`), the eval suite (`scripts/eval_suite.py`,
Necto/Nexto matches plus the scenario suite), the auto-eval watcher, the TrueSkill evaluator and rating fit,
the probes (`action_saturation.py`, `steering_jitter_probe.py`, `defence_challenge_probe.py`,
`retreat_comparison.py`, `policy_health.py`, `entropy_report.py`), the replay parser, and the pinned
checkpoints (`checkpoints/baselines/`, including v5 222000 and v11 228000/230400) as a reference ladder.
The eval suite and probes run matches through SenseiBot's Python environment, so its runtime moves with
them as evaluation infrastructure: `env/rocket_env.py`, `env/observations.py` (the 108-dim observation),
`env/physics_engine.py`, `env/baseline_agent.py` (the heuristic chaser), `agent/checkpoint.py` (loads the
v3–v11 models), `utils/eval_results.py`, and the parts of `utils/config.py` and `env/reward_registry.py`
that `policy_health.py` reads. Trace the imports when moving (`graft callers <file> --depth all`), so the
ladder keeps working.

**Stays here as the archive:** the reward specs (`docs/reward_v*_spec.md`), run history and logs, and the
781-minute human replay study.

**Retired** (no longer used for training; they stay in this repo): `agent/ppo.py`, `train.py`, the
training side of `env/` (reward terms, reward versions, training scenario mix, replay sampling),
`utils/league_manager.py`, behavioural cloning, the auto-eval watcher's training coupling, and
`config/reward_versions/`.

## 3. Review of Prometheus (2026-09-24)

- **Safety:** no network access beyond an optional local Weights & Biases metrics receiver
  (`GigaLearnCPP/python_scripts/metric_receiver.py`, off unless enabled). Build and run scripts only set
  paths. `deceive_config.json` is an alternative training config. `AGENTS.md` carries instructions
  for AI coding tools about a code-graph utility; treat it as documentation.
- **Licence:** the Prometheus code declares no licence, which means all rights reserved. Private use
  is one thing; publishing SensAI's code or entering it in tournaments needs permission from the author
  (mitige). RocketSim, RLBotCPP, `rlbot/`, pybind11, nlohmann json and thread_pool carry LICENSE files;
  GigaLearnCPP and RLGymCPP have none at their top level, so they need checking upstream as well.
- **Running it:** Claude reviews code, writes config and changes, and analyses results. **The user
  builds and runs** the executables.
- **Its continuous head** (`GigaLearnCPP/src/private/GigaLearnCPP/PPO/PPOLearner.cpp:225-372`):
  samples in pre-tanh space and then squashes, with the log-probability corrected for the tanh; the
  mean is clamped to ±4; the spread is predicted per state within [`varMin`, `varMax`] = [0.1, 1.0];
  the entropy bonus includes the squash correction, which discourages saturated means. The policy
  gradient on the mean never passes through tanh's slope, which removes the dead-gradient failure
  measured in every SenseiBot checkpoint since v3.
- **Transfer learning** (`PPO/TransferLearnConfig.h`): student imitates a teacher's actions (MSE and
  log-likelihood), buttons (BCE), spread and KL, with half the rollouts driven by the student and aerial
  states weighted 8×. It supports a discrete teacher with a continuous student.
- **Driving it without its GUI.** The ImGui app (`src/gui/GuiApp.cpp`) and the command-line
  `GigaLearnBot.exe` (`src/ExampleMain.cpp`) are both thin callers of `GGL::Learner`, whose public members
  give an external front end everything it needs: `stopRequested` and `saveRequested` (atomics checked by
  the loop), `iterationCallback(Learner*, const Report&)` (called synchronously on the training thread
  each iteration, `Learner.cpp:1380`), `ppo->SetLearningRates()`, `ppo->config.entropyScale`, and
  `config.ppo.gaeGamma` (read fresh each iteration). The GUI is excluded with `-DPROMETHEUS_BUILD_GUI=OFF`.
  With `sendMetrics` off, no Python interpreter is started at run time; Python is linked at build time only.
- **No architecture-specific GPU code.** A search of every CUDA and C++ source for `__CUDA_ARCH__`
  branches, inline assembly, legacy warp intrinsics, texture/surface references and device-capability
  checks found none. Blackwell needs only the build flags, the toolkit and LibTorch (Phase 0).

## 4. Phases

Each phase ends at a gate. Nothing moves on until the gate passes.

### Phase 0: build and verify on the RTX 5090 (no training)

1. **Toolchain.** The 5090 is Blackwell (compute capability 12.0); older CUDA and LibTorch builds cannot
   run on it. State of this machine on 2026-09-24:

   | requirement | status |
   |---|---|
   | RTX 5090, driver | ✅ compute capability 12.0, driver 610.88 |
   | CUDA Toolkit ≥ 12.8 | ✅ 12.8 (V12.8.61), `CUDA_PATH` set |
   | VS 2022 C++ build tools, CMake ≥ 3.18 | ✅ Build Tools 17.14, MSVC 14.44, bundled CMake 3.31 |
   | **CUDA's Visual Studio integration** | ❌ `CUDA 12.8.props/.targets` are in neither VS 2022 nor VS 2026. Re-run the CUDA installer with *Visual Studio Integration*, or copy `CUDA\v12.8\extras\visual_studio_integration\MSBuildExtensions\*` into `…\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations`. Without it CMake stops with "No CUDA toolset found". |
   | **Python 3.11 x64** | ❌ only 3.14 is installed. The bundled pybind11 is 2.12, which predates 3.13/3.14 support, so 3.14 cannot be used. Install 3.11 alongside 3.14 and set `PROMETHEUS_PYTHON_HOME` to it; otherwise `tools/prometheus_env.bat` silently picks whatever `python` is on PATH. Studio and the eval suite keep running on RLBotGUIX's own Python 3.11 (see Phase 1 step 1). |
   | **LibTorch, CUDA 12.8 build (2.7 or later), Release** | ❌ not installed. Extract to a short path such as `C:\tools\libtorch` and set `LIBTORCH_DIR`. The installed PyTorch 2.10 is CPU-only and does not count. |
   | Shell | CMake is not on PATH, and VS 2026 Build Tools is also installed. Build from **Developer PowerShell for VS 2022** so its CMake and toolset are used. |
   | Long paths | Windows `LongPathsEnabled` is 0. The longest repo path under `SensAI\` is 161 characters, which fits, but clone with `-c core.longpaths=true` and consider enabling long paths (a system setting, done by the user). |

2. **Architecture flags.** `RocketSimCuda/CMakeLists.txt` compiles for `CUDA_ARCHITECTURES "75;86;89"`
   in 8 places (lines 56, 73, 80, 90, 107, 122, 136, 151). Add `120` to every one. Without it the build
   succeeds and every kernel fails at runtime with "no kernel image is available". This is the only
   Blackwell-specific change (§3).
3. **Fork into `SensAI`.** `git clone -c core.longpaths=true https://github.com/mitige/prometheus
   C:\Users\coryf\antigravity\SensAI`, keep the original commit hash in the first commit message, and make
   the architecture change the second commit.
4. **Build**, both in Release:
   - the root project with `build.bat`. This builds the learner and, from `RocketSimCuda/`, only
     `RocketSimCudaTest`, `RocketSimCudaGaeParity` and `RocketSimCudaBenchmark1024`;
   - `RocketSimCuda/` configured as its own project (`cmake -S RocketSimCuda -B build_rscuda`, same
     `CMAKE_PREFIX_PATH`, then `cmake --build build_rscuda --config Release`). The CPU↔GPU comparison,
     game-state bridge, env-pipeline benchmark and divergence tracer are only defined when that folder is
     the top-level project (`CMakeLists.txt:93`).
5. **Physics parity.** On the 5090: `RocketSimCpuCudaCompare` (compare_cpu_cuda),
   `RocketSimCudaGameStateBridgeTest`, `RocketSimCudaTest` and `RocketSimCudaGaeParity`. The GPU physics
   must match CPU RocketSim on this card; a physics mismatch teaches the policy the wrong game.
   `RocketSimCudaTraceDivergence` is the tool if the comparison fails. Separately,
   `bash RocketSimCuda/tests/run_cpu_parity.sh` runs the CPU-only harness: it compiles the kernels as
   plain C++ and checks them against a float64 reference, so it checks the physics code, not this card.
6. **Throughput.** Run `RocketSimCudaBenchmark1024` and `RocketSimEnvPipelineBenchmark`, then a few
   minutes of `run.bat train` in CUDA mode and in `cpu` mode. Record env-steps/s for each. (This uses
   upstream's command-line trainer and 2v2 defaults; it measures the hardware, not a run.)

**Gate:** every parity test passes, and CUDA throughput is measured. The throughput number sets the
budget for Phase 3 (at SenseiBot's ~8k steps/s, 5B steps took ~7 days).

**Phase 2 complete 2026-09-24* (all items above). *Gate passed 2026-09-24.** Parity passes with three documented exceptions, and throughput is
measured (§6). Result: **train on CPU physics**, ~123k steps/s in 1v1 at 2048 games (~150k in 2v2),
which puts 5B steps at ~10–12 h.

### Phase 1: headless trainer, Studio, and the 1v1 config

SensAI Studio stays the front end. Prometheus's ImGui GUI is not built (`-DPROMETHEUS_BUILD_GUI=OFF`);
the upstream GUI stays in the tree, unmodified, as reference. `ExampleMain.cpp` (`GigaLearnBot.exe`)
stays as the benchmark tool, with the Phase 0 flags (`cpu`, `cpu-obs`, `unpadded`, `gpu-rewards`,
`players=`, `games=`); runs go through `SensAITrainer`.

1. **Move Studio and the evaluation into SensAI.** Copy the assets listed in §2 into a Python folder in
   SensAI (e.g. `studio/`, beside the C++ tree), with their history noted in the commit message, and
   commit before changing anything. Studio and the eval suite keep running on RLBotGUIX's Python 3.11
   (`%LOCALAPPDATA%\RLBotGUIX\Python311`, torch 2.0.1 CPU, gradio 6.25), which `start.bat` uses.
   - *Done 2026-09-24* as SensAI `studio/`, copied from SenseiBot `06973cc` with `git archive`
     (committed contents, not working-tree run state). The import closure is larger than §2's list:
     `env/rocket_env.py` imports the reward modules, `utils/league_manager.py` and the BC pretrainer,
     and `agent/__init__.py` imports `agent/ppo.py`, so the retired trainer comes along as a
     dependency (60 modules). Also copied: 73 tests whose imports stay inside that set, `config/`,
     `data/`, `evals/baselines/`, `checkpoints/baselines/` plus the Necto/Nexto/pretrained models,
     `bin/rrrocket.exe`, `collision_meshes/`, `requirements.txt`, `start.bat`. `studio/` is the
     working directory for Studio and, later, for `SensAITrainer`. Pruning the retired modules is step
     3's job, once nothing imports them.
2. **Write the headless trainer**, `src/SensAIMain.cpp` → `SensAITrainer.exe`, a new CMake target
   beside `GigaLearnBot`. It:
   - reads one **profile JSON** (`--config <path>`): mode and `playersPerTeam`, reward list and weights
     (by name, through a small reward factory), state-setter mix, network and PPO settings, and the
     horizon schedule. This is the config-profile gating described below;
   - trains on **CPU physics** (`EnvPhysicsBackend::ROCKETSIM_CPU`, `cudaNoCpuWorldState` off) with the
     learner on the GPU (`LearnerDeviceType::GPU_CUDA`). Start at 2048 games in 1v1, which keeps a
     ~49-step segment per player at 200k steps per iteration; games and steps per iteration are
     explicit, frozen run-1 settings, since together they set the segment length GAE sees. The backend
     stays a profile field so GPU physics can be revisited, but it is not an option for run 1;
   - writes `logs/train.pid` at start and removes it on exit;
   - in `iterationCallback`, appends one line to `logs/history.jsonl` and rewrites `logs/metrics.json`
     in the shapes Studio reads today (`utils/run_history.py`, `utils/process_manager.py`), mapping
     Prometheus's report keys (`Policy Entropy`, `SB3 Clip Fraction`, `Rewards/*`, …) to Studio's. Each
     history line also carries the unmapped report under `raw`, so no metric is lost to a mapping gap;
   - also in `iterationCallback`, polls `config/live_config.json` and applies: `paused` (wait inside
     the callback), `save_checkpoint_requested` → `saveRequested`, a new `stop_requested` →
     `stopRequested` then a final `Save()`, `learning_rate` → `ppo->SetLearningRates()`, `ent_coef` →
     `ppo->config.entropyScale`. Rewards and the start mix are frozen for the run (R1) and not live;
   - sets `config.ppo.gaeGamma` from the horizon schedule each iteration and logs T and gamma.
   - *Done 2026-09-24* as SensAI `7a38e2e` (`src/SensAIMain.cpp`, profile layout in `docs/profile.md`).
     Smoke-tested with `smoke_1v1.json` (256 games, 8.2M steps over two sessions): profile checks,
     saves kept and stamped, weights-only archive every 500k, resume under the same profile, pause,
     manual save while paused, live learning rate and entropy, and a clean `stop_requested` stop with a
     final save all work. Explained variance was added to the Learner's report, and
     `SetLearningRates`/`SetEntropyScale` exported, since `PPOLearner` is not exported from the DLL.
     Studio's live keys stay as they are (`learning_rate` sets both learning rates, `ent_coef` sets the
     entropy scale); `stop_requested` is new.
3. **Wire Studio to it.**
   - `process_manager.start_training` launches `build\Release\SensAITrainer.exe --config <profile>`
     with `tools/prometheus_env.bat`'s PATH (for the LibTorch and Python DLLs), instead of
     `python train.py`.
   - `stop_training` sets `stop_requested` and waits for a clean exit before falling back to killing the
     process tree. Killing it loses everything since the last save.
   - The liveness checks look for `SensAITrainer.exe`, not `python`, in `tasklist`
     (`process_manager.is_running`, `scripts/fresh_run.py:training_is_live`).
   - Training plots and the reward card read the mapped keys.
   - The Training config tab writes the profile JSON instead of the yaml.
   - The League tab becomes a read-only view of Prometheus's old-version pool
     (`trainAgainstOldVersions`, `tsPerVersion`, `maxOldVersions`) or is hidden. Fixed training opponents
     go with the old league.
   - Custom scenarios wait for a C++ state setter that reads the scenario JSON (not needed for run 1).
   - Replays & pretraining is removed, since behavioural cloning is retired; transfer learning is the
     replacement if a teacher is ever used. (2026-09-25: it will be; the teacher is `discrete_1v1`.)
   - Tabs that load checkpoints (Evaluation, Behaviour, Watch a match, Health) wait for Phase 2's bridge.
   - *Done 2026-09-24* as SensAI `02fcb35`, `99bc9e3`, `5366a4a`:
     - Start runs `SensAITrainer` on a profile picked in the header.
     - Stop requests a clean stop, and kills only after a 60 s timeout.
     - "Running" means the pid file's process is a live `SensAITrainer.exe`. The old test needed fresh
       metrics, so it showed a trainer paused for more than 25 s as stopped.
     - A profile card replaces the reward card, and the plot titles follow the records.
     - The Training config tab is a profile editor: a profile is read-only once its run has saves, and
       *Save as new profile* can set `start_from` to the source run's latest save.
     - The League tab, league manager, behavioural-cloning pretrainer and SenseiBot's PPO trainer are
       removed (user's call). `agent/__init__.py` no longer imports the old trainer.
     - Kept: the replay pool, parser and tools (the "Replays" tab), as the source for replay start
       states or pretraining in a later run, and `bot.py` to play SenseiBot's reference models in RLBot.
     - Custom scenarios stays hidden until a C++ state setter reads its scenarios.
     - 66 test modules, 721 tests pass, apart from the two tests in `test_reward_audit_fixes` that fail
       at random in SenseiBot too.
4. **Smoke test:** Start, Pause, Save, change LR, and Stop from Studio against a short 1v1 run. Check
   that the plots update, the checkpoint lands, and the process exits cleanly.
   - *Done 2026-09-24* on `smoke_1v1.json` (user). Phase 1's gate now needs only the frozen run-1
     profile (`run1_1v1.draft.json` has no open decisions left) and run 1's written spec.

**Configuration for run 1:**

- **Start with 1v1, keep everything 2v2.** SensAI's goal includes 2v2, so nothing team-related is
  removed from the code. Modes are **gated by config profile** instead: a 1v1 profile
  (`playersPerTeam: 1`) and a 2v2 profile, each with its own reward list and weights. 2v2-only terms
  (`Kickoff2v2SimpleReward`, `TeamSpacingReward`, `TeammateBumpPenaltyReward`, the `teamSpirit`
  settings) stay in the code and in the 2v2 profile, and are simply not enabled in the 1v1 profile.
- **One network for both modes.** `AdvancedObsPadded` pads the observation to 2 teammates and 3
  opponents with zeros (`MAX_TEAMMATES = 2`, `MAX_OPPONENTS = 3`, 29 features per player) and includes
  the previous action. A 1v1 policy therefore sees the same input layout as a 2v2 or 3v3 one, and can be
  trained on into 2v2 without changing its inputs.
  - **Checked in Phase 1 (2026-09-24): there is no attention over players.** `AttentionModel::Forward`
    feeds the whole flat observation as one token, as both query and key (`AttentionModel.cpp:94–99`).
    Attention over a single token has nothing to weigh, so the "attention head" works as a residual
    MLP of ~7M parameters, and its padding-mask argument is never used. The padded layout still keeps
    the input size fixed across modes, but the network sees fixed slots, like an MLP. §1's table is
    right about the configuration and wrong to imply attention over players.
  - Empty slots are zero-filled. That is unambiguous (a real car's forward and up vectors are unit
    length, a zero slot's are zero), but in 1v1 the empty-slot inputs are always zero, so their
    first-layer weights never receive a gradient and stay at their random initial values until 2v2.
    **Decided: zero-fill for run 1** (user's call, 2026-09-24). At the move to 2v2, zero the
    empty-slot first-layer weights so new cars start with no effect. Real attention over players
    (per-player tokens with a mask) is a later architecture run, compared against run 1.
- Keep: continuous head with `varMin`/`varMax`, the attention-head network (a residual MLP in practice), `AdvancedObsPadded`,
  `trainAgainstOldVersions` at 20%.
- **Reward set, judged against §2's lessons.** Prometheus's defaults include terms our own runs showed
  fail:

| Prometheus term (weight) | what it pays | verdict for SensAI |
|---|---|---|
| `GoalReward` (150) | goals | keep; the dominant term |
| `TouchBallReward` (3.2, zero-sum) | touches | keep |
| `GoalDirectedTouchSpeedReward` (0.4, zero-sum) | touch speed toward goal | keep |
| `VelocityPlayerToBallReward` (0.28) | speed toward the ball | candidate; Seer uses it too. Canary: touches and time to ball, not speed |
| `VelocityBallToGoalReward` (5.0) | ball velocity toward goal | candidate, zero-sum; watch for overcommitting (lesson 5) |
| **`PickupBoostReward` (20)** | **each rise in boost, plus 1 per pickup** | **drop.** This is v8's T7 ratchet, which was cycled |
| **`BoostSpendPenaltyReward` (0.1)** | minus each fall in boost | **drop.** A discounted fall does not stop cycling (lesson 2) |
| **`SaveBoostReward` (0.2, not zero-sum)** | sqrt of boost held, every step | **drop.** This is v9's T8 without the zero-sum, so a hoarding and survival bonus (lessons 3–4) |
| **`MKHKickoffSpeedflipReward` (0.12)** | a speedflip detected from **pitch/yaw inputs** | **drop.** It reads the action (R2). Replace with a state-based kickoff term, e.g. Seer's velocity toward the ball while the ball is at centre |
| `FreeGoalEventPenaltyReward` (0.42) | on conceding, penalty scaled by the closest defender's distance from goal | **off in the 1v1 profile.** With one defender it pays for staying home, which is the passivity already seen. Reconsider for 2v2, where "someone stays back" is a real rotation rule |
| `Kickoff2v2SimpleReward`, `TeamSpacingReward`, `TeammateBumpPenaltyReward` | 2v2 kickoff roles, spacing, not bumping a teammate | **2v2 profile only**; kept in code, off in 1v1 |
| `WavedashKickoff50Reward`, `StagedFlickReward`, `WavedashRecoveryReward`, `AirDribbleReward`, `AerialTouchHeightReward`, `IsBallVelocityOnTargetReward` | mechanics shaping | read each before use: check R2, and check nothing pays for a state a canary does not watch |

- **Horizon: Seer's annealed schedule** (user's call). Express the horizon as a half-life T, the time
  after which a reward is worth half, and derive gamma from it at 15 steps/s:
  `gamma = exp(ln(0.5) / (15 · T))`. **Start at T = 10 s (gamma 0.99539) and anneal to T = 20 s
  (gamma 0.99769)**, as Seer did (thesis §3.8, Fig. 3.8). A short horizon makes early credit
  assignment easier; a long one supports the planning a strong policy needs. The end point equals
  SenseiBot's adopted v5 gamma (0.9977). Prometheus's default 0.993 is a 6.6 s half-life, shorter than
  both.
  - **Schedule:** hold at 10 s until the run's early mechanism check has passed, then move T (not gamma)
    linearly to 20 s. Choose the length of the ramp from Phase 0's throughput (~123k steps/s in 1v1, so
    ~440M steps per hour), and freeze start, end,
    hold and ramp in run 1's spec. The schedule is part of the run's identity, so changing it means a
    new run.
  - **Implementation:** GigaLearn reads `config.ppo.gaeGamma` fresh on every iteration
    (`GigaLearnCPP/src/public/GigaLearnCPP/Learner.cpp:1307`), so the anneal is a per-iteration update
    of that value from total timesteps, made in `SensAITrainer`'s `iterationCallback` (step 2 above).
    Upstream's `ExampleMain.cpp:294` hardcodes 0.993 and is not used for runs. Log the current T and
    gamma with every iteration.
  - **Reading results during the ramp:** return standardization (`standardizeReturns`) absorbs the
    growing return scale, but the critic has to catch up at each step up. Watch explained variance
    through the ramp, and don't attribute behaviour changes during it to anything else without a
    same-step control.
- **Write the first run's spec before it starts**, in the SensAI repo, in the same format as
  `docs/reward_v*_spec.md`: identity, reward table, decision points, 400M-equivalent criterion.

**Gate:** Studio drives `SensAITrainer` end to end (step 4), a frozen 1v1 profile, and a written spec
for run 1.

**Gate passed 2026-09-24** (SensAI `be4e6fa`): Studio drives the trainer (step 4), the profile is
frozen as `studio/config/profiles/run1_1v1.json`, and run 1's spec is `docs/run1_spec.md`.

### Phase 2: bring the evaluation across

- **Python inference bridge.** Port `AdvancedObsPadded`, the attention model and the squashed head to
  Python/PyTorch so SensAI checkpoints load into the eval suite and every probe (now in SensAI), running
  in RocketSim from Python. Studio's CPU PyTorch (2.0.1, RLBotGUIX Python 3.11) is enough for matches
  and probes; the parity test must hold across it and the trainer's LibTorch 2.10. This also unlocks
  Studio's Evaluation, Behaviour, Watch a match and Health tabs.
- **Parity test.** For the same game states, the C++ and Python observations must match to float
  tolerance, and the deterministic actions must match. No eval number is trusted until this passes.
  - *Passed 2026-09-24* (SensAI `af5fc1a`: `studio/sensai_bridge`, `src/SensAIParityDump.cpp`,
    `studio/scripts/sensai_parity.py`, `parity_sensai.bat`). On 200 states (100 1v1, 100 2v2; 600
    player observations, half orange, 84% airborne, pads on cooldown in 64 states): observations
    bit-identical to C++, deterministic actions within 7.8e-7, no button disagreements. Not covered: a
    demolished car (`is_demoed` was never set).
  - Found on the way: upstream's `GetBoostPadTimers(inverted)` returns the other side's timers, so
    an empty pad whose mirror is available reads as available. The bridge keeps it (the model trained
    on it); 214 of 600 observations would differ without it. Recorded in run 1's spec as a candidate
    fix for a later run.
  - *Arena adapter passed 2026-09-24* (SensAI `27e8f65`). `sensai_bridge/arena.py` reads the eval
    suite's pip RocketSim 2.2.1 arena (RocketSim's own car state; pads matched by position), and
    `sensai_bridge/bot.py` (`SensAIBot`) plays with the trainer's timing: 7 ticks of the previous action,
    then the new one, as an 8-tick block, with the previous action reset at each episode start. Bots
    that take raw RocketSim controls are now picked by `wants_raw_controls()` (Necto, Nexto, SensAI),
    not by class. On the same 600 dumped states the pip binding's `has_flip_or_jump()` agrees with
    C++ on every player, and observations through the adapter match C++ within 2.4e-7.
- **Eval suite on SensAI checkpoints.** *Done 2026-09-24* (SensAI `e5648b5`). The checkpoint being
  evaluated can be a SensAI save folder (`<run>/<step>` or `<run>/archive/<step>`):
  `scripts/eval_policy.py` plays it as blue through `SensAIBot`, and `RocketLeagueEnv.step(agent_ticks=...)`
  runs its per-tick block without the jump sequencer. Results are stamped with the profile as the
  version and named `<profile>_<M>M`. `shot_quality.py` and `drop_ball_scenario.py`, which the suite
  imports, came across from SenseiBot. SenseiBot `.pt` files still evaluate as before.
- **Reference ladder.** Heuristic chaser, v3 198000, v5 222000, v11 228000 and 230400, Necto, Nexto.
  SenseiBot sits near the floor against Necto (−29 goals per 10 min), where real differences look like
  noise. A ladder spanning weaker opponents gives resolution from the first checkpoint.
  - *Done 2026-09-24* (SensAI `4c25eea`, readings `b153a30`). Every suite run now also plays the heuristic,
    v3 198000, v5 222000, v11 228000 and v11 230400 on the three match seeds (groups `ladder_<rung>`); Necto
    and Nexto stay the top rungs. Each reference was run through the suite as the evaluated side (the
    heuristic, Necto and Nexto can now play blue) into `studio/evals/baselines/`, and
    `eval_suite.py --ladder-table <result>` prints a result beside them. Goal difference per 10 min:

    | evaluated \ vs | heur | v3 | v5 | v11 228k | v11 230k | Necto | Nexto |
    |---|---|---|---|---|---|---|---|
    | heuristic | 1.0 | −7.2 | −16.0 | −15.8 | −26.8 | −38.8 | −39.8 |
    | v3 198000 | 14.8 | −2.8 | −8.2 | −7.0 | −19.5 | −23.0 | −39.0 |
    | v5 222000 | 15.0 | 12.0 | −2.5 | −3.8 | 2.0 | −26.2 | −27.8 |
    | v11 228000 | 15.2 | 12.0 | 10.0 | −3.8 | 2.2 | −18.2 | −30.0 |
    | v11 230400 | 26.0 | 29.5 | −2.0 | 3.0 | 2.0 | −16.2 | −31.5 |
    | Necto | 39.5 | 26.2 | 29.5 | 27.5 | 24.8 | 9.2 | −8.2 |
    | Nexto | 42.8 | 34.2 | 26.2 | 28.0 | 35.8 | 10.8 | 3.8 |

    The heuristic rung is where an early checkpoint gets resolution; the SenseiBot rungs sit within
    about ±10 of one another, so they resolve the middle only coarsely.
  - Found on the way: **a seed does not fix the game.** The same seed replays differently from run to
    run (v5 vs Necto, seed 11, three runs: 0–13, 2–7, 0–10), so seeds are independent samples. With 3
    seeds, "every seed on one side of zero" happens 25% of the time by chance. The source of the
    nondeterminism is not yet found (np and torch are seeded).
  - Found on the way: **the blue seat is worth little, and one run is noisy.** The evaluated checkpoint
    always plays blue. The first run suggested a large seat bias (Necto beat itself from blue 163–96
    over 9 seeds); a second run of all seven references (SensAI `6e56ec4`) took Necto's mirror from
    +9.2 to −0.2. Pooled over 12 seeds, blue Necto leads 204–138 (about +4 per 10 min); Nexto is even
    (101–109); the SenseiBot mirrors are inconsistent. Identical runs moved 3-seed means by up to ~9
    goals per 10 min (v11 230400 vs Necto −16.2 then −25.0), so one suite run is a rough reading and
    run 1's pooling over ±50M steps is what calls a change. The table above is the first run; the
    committed files are the second.
- **Probe updates.** `action_saturation.py` must read the new head: pre-tanh means against the ±4
  clamp and the state-dependent spread. Braking share and steer reversals stay as-is.
  - *Done 2026-09-24* (SensAI `768033d`). GigaLearn takes the log-probability on atanh(action), so the
    mean's gradient carries no tanh derivative: v3–v11's failure (mean past full lock, gradient ~0)
    cannot happen through tanh. The wall is the ±4 clamp on the raw mean, which passes no gradient.
    For a SensAI checkpoint `action_saturation.py` reports per channel (throttle and steer grounded,
    pitch, yaw, roll airborne) deterministic saturation, share at and near the clamp, the raw mean's
    size, the per-state spread (median, p10, p90, share at the floor) and braking share. The smoke
    model: nothing saturated or clamped, spread near its 1.0 ceiling, 77% braking or reversing.
  - Kickoffs: the suite now reports `kickoff_dodge_pct` (dodged before first contact),
    `kickoff_opp_dodge_pct`, `kickoff_contact_s` and `kickoff_approach_speed` (the step before
    contact), shown in Studio as a Kickoffs section. Against Necto: every SenseiBot checkpoint and the
    heuristic 0% dodges at ~1980–2070 uu/s; Necto and Nexto 100% at ~2200. The references carry them.
  - Not ported: `defence_challenge_probe.py`, `retreat_comparison.py`, `steering_jitter_probe.py` and
    `policy_health.py` still load SenseiBot `.pt` files only. Port one when a run needs it.
- **Auto-eval.** Point the watcher at SensAI's checkpoint folder and Prometheus's checkpoint layout
  (one folder per timestep count, plus `RUNNING_STATS.json`).
  - *Done 2026-09-24* (SensAI `35fc3df`). `scripts/auto_eval.py` follows `checkpoints/<run>/archive/`
    and evaluates the first complete archive at or past each 25M steps from the run's own start,
    oldest first, reporting a backlog when it falls behind. Studio's "Follow the run" takes the
    interval in millions of steps and a ladder switch; the Evaluation tab lists SensAI saves and
    archives, pools by run and compares with v5 222000's reading by default. Checked end to end on
    the smoke run at a 2M interval.
  - **Cost, open for run 1:** a full suite with the ladder is ~12 CPU-minutes (2.9 min on 4 idle
    workers). Run 1's spec assumed ~46 s, which was the pre-ladder suite's wall time on 4 workers. At
    ~123k steps/s, 25M steps is ~3.4 min, so the watcher barely keeps pace on an idle machine and
    falls behind beside the trainer, which also loses the cores it takes. Options: a longer
    interval, `--no-ladder` on most evals, or accept the lag and let pooling catch up after.

**Gate:** parity passes, and one checkpoint runs through the full suite end to end.

*Gate passed 2026-09-24.* Parity as above; the full suite (3 Necto seeds of 12000 steps, Nexto, 4
scenarios of 24 trials) ran end to end on the smoke run's 6062848 save in 0.8 min on 8 workers, and
its result lists in Studio as `smoke_1v1 +6M`. The smoke model barely plays (no touches against Necto;
its training log shows 0.3-0.5 touches per player-minute), so this proves the plumbing, not a baseline.
The ladder, probes and auto-eval followed; Phase 2 is complete.

### Phase 3: first run

- **From scratch.** Prometheus's 28B-step teacher is not public. Using v5 as a teacher would mean
  porting SenseiBot's 108-dim observation and model into C++. Do that only if Phase 0's throughput is
  too low to train from scratch in reasonable time. It is not: at ~123k steps/s, v5's ~3.6B steps take
  ~8 h, so run 1 starts from scratch.
- **Milestones written before the run**: an early mechanism check (saturation, spread, braking), a
  passivity check (head to head and guardrails), and an adoption criterion (clearly better than v5 on
  the ladder and on Nexto).
- **First comparison target:** v5 222000, then v11 230400.

*Run 1 ran 2026-09-24 23:20 to 2026-09-25, stopped at ~4.16B steps (SensAI `docs/run1_spec.md` §7–8).*
- **Not adopted.** In the final window it was −39.3 against v5 head to head, −40.8 against Necto and
  −44.8 against Nexto. It was level with the heuristic chaser from ~1.5B on, and the plateau check
  fired at 3.5B. The one learned behaviour is kickoff flipping (0% → ~90% dodges).
- **Why:** throttle, steer, pitch and roll never learned. Their means stayed at ~0 with the spread
  pinned at 1.0 through 4B steps; only yaw learned. So the policy moved by boosting and steered by
  jumping and yawing (81–88% airborne).
- **First hypothesis, not sufficient:** GigaLearn's `ComputeContinuousEntropy` adds
  `log(1 − tanh²(mean))`, which pulls the means to 0. `headcheck_1v1` switched it off
  (`actions.entropy_squash_correction: false`) and tracked run 1 step for step to 200M.
- **Revised cause:** a fresh head's button logits are ~0, so jump and handbrake are pressed half the
  time. Both runs were 88% airborne from the first iteration, and jump p on the ground was still
  0.44 at 200M (0.3 at 4B). Throttle and steer do nothing in the air, so they got almost no signal
  while the entropy bonus pushed their spread to the ceiling, where `tanh` leaves it almost no
  gradient.
- **`headcheck2_1v1`** (`actions.init_button_bias: [-3, 0, -3]`, `init_spread: 0.3`, stopped at
  ~158M): the car trained ~40% on the ground. Steer and pitch learned, and touches per player-minute
  reached 1.84 at 150M, against ~0.8 for run 1 at 130M.
- **What it left:** with weak signal, handbrake and boost drifted toward p 0.5, yaw and roll sat at the
  spread ceiling, and the throttle spread rose again after 100M. The entropy scale (0.025) outweighs
  the policy gradient on those channels.
- **Next:** `headcheck3_1v1` is `headcheck2_1v1` with entropy scale 0.01 (SensAI `docs/run1_spec.md`
  §8).
- **`headcheck3_1v1`** (entropy 0.01, stopped at ~222M) was worse on every outcome: 0 touches
  against Necto, and 70–84% airborne.
- **The finding behind it:** no SensAI checkpoint steers toward the ball.
  - `headcheck3` turns hard one way in every state.
  - `headcheck2` turns one way until the ball is ahead.
  - Run 1 never steered.
- **Likely source:** every run's first iterations take huge updates on an untrained critic (KL
  0.5–4, clip 0.3–0.8). `headcheck3`'s steer locked in its first 3 iterations.
- **Next:** `headcheck4_1v1` is `headcheck2` plus `ppo.policy_warmup` (policy frozen 5M, ramped over
  10M). The hard pass is steer following the ball's side both ways by 50–100M
  (`action_saturation.py` now prints it). If it fails, the fallback is discrete actions with
  continuous control distilled later (user's call).
- **`headcheck4_1v1`** (policy warmup, stopped at ~115M) failed. The warmup fixed the first updates
  (KL 0.005–0.015). But the one-way steer (−0.5 in every state) was already there at 5.3M, before the
  policy had learned anything: it comes from the fresh output layer, and PPO never learned away from
  it. Touches trailed `headcheck2` at every step (0.74 against 1.29 per player-minute at 100M), and so
  did evals (Necto −128 to −136 against −68 to −88).
- **Discrete next (user's call, 2026-09-25).** Behavioural cloning would help here: a cloned start
  already steers to the ball. It takes the form of GigaLearn's transfer learning, not SenseiBot's
  replay cloning, which had imputed actions and the old observation. Transfer learning needs a teacher
  in GigaLearn's format, so `discrete_1v1` makes one: `headcheck2`'s setup with GigaLearn's 90-action
  table, to be cloned into the continuous head later. Pass: steer follows the ball by 50–100M, and
  touches above `headcheck2`'s. If discrete fails too, the rewards and starts are next, not the head.
  SensAI `docs/run1_spec.md` §8.
- **Found on the way:** Studio's writes to `live_config.json` replayed SenseiBot's stale ent 0.008 /
  LR 1.5e-4 onto one iteration per run (too little to matter). Fixed in the trainer and in Studio's
  Start.
- **Run 1b** (horizon ramp from run 1) is dropped.
- **Process miss:** the 50M saturation check in the spec was never run. It would have caught this in
  minutes. The watcher should run the probe itself.

## 5. Risks and open questions

| risk | what would show it | response |
|---|---|---|
| GPU physics differs from CPU on Blackwell | Phase 0 parity tests | **resolved:** parity passes with three documented exceptions, and runs use CPU physics |
| throughput gain smaller than hoped | Phase 0 benchmarks | **resolved:** CPU physics ~123k steps/s in 1v1, ~15× SenseiBot; no teacher needed |
| CPU-bound collection | Studio, or anything else heavy, running on the same 24 cores during a run | keep the machine quiet during runs; collection, not learning, is the limit |
| games and steps per iteration set the segment length | too many games per iteration leaves short segments and a critic-heavy GAE | freeze both in each run's spec; revisit GPU physics only together with larger iterations |
| eval bridge mismatch | Phase 2 parity test | fix before trusting any number |
| Studio misreads the trainer | plots or cards blank or wrong after the key mapping | Phase 1 smoke test; the trainer writes the raw report too, so a mapping bug never loses data |
| hard stop loses progress | Studio kills the process tree | graceful `stop_requested` first; kill only as a fallback |
| upstream changes after the fork | new Prometheus commits | the fork is pinned at `e4d097d`; `SensAIMain.cpp` is a separate file, so pulling upstream fixes stays a clean merge |
| Prometheus's hyperparameters are tuned for 2v2 | early instability, entropy collapse | treat batch, lr, entropy and gamma as explicit run-1 choices |
| unknown strength of Prometheus itself | no published results | judge only on our own ladder |
| licence | publishing or tournament entry | ask mitige before either |
| retreat regression (v11, cause unknown) | retreat scenario in the suite | keep the retreat metrics as a guardrail |
| kickoff: SenseiBot never flipped | kickoff first touch, a kickoff dodge metric | state-based kickoff reward in run 1; add a kickoff-dodge count to the suite |
| upstream action-head defaults untested for 1v1 continuous control | spread pinned at `var_max`, means stuck at 0 (run 1) | **found in run 1:** a fresh policy's buttons are pressed half the time, so the car is airborne from the start and ground controls never train; `actions.init_button_bias` / `init_spread` (checked by `headcheck2_1v1`). The entropy's tanh term, which pulls the means to 0, is switched off too (`entropy_squash_correction`) but was not the cause. Run the saturation probe at 50M on every run |
| no critic warmup in GigaLearn | the first iterations update the policy on a critic that explains nothing: from scratch (KL 0.5–4 in run 1 and the head checks), and in a `start_from` run whose critic fits old returns | **available:** `ppo.policy_warmup` (policy LR 0 for N steps, then a ramp; the shared head trains with the critic meanwhile), counted from the run's start. `headcheck4_1v1` showed that it works (first KL 0.005–0.015), but it was not what blocked steering. Use it for the transfer and continuation runs |
| the continuous head does not learn from scratch | steer ignores the ball's side (run 1, head checks 1–4) | **open:** train a discrete teacher (`discrete_1v1`), then clone it into the continuous head with GigaLearn's transfer learning. A fresh discrete policy puts ~45% of its grounded probability on jumps, so check ground time from iteration 1 |
| moving to 2v2 later | a 2v2 run from a 1v1 checkpoint | same network and observation carry over; the eval suite and probes are 1v1-only and need 2v2 versions (Necto and Nexto both play team modes) before a 2v2 run is judged |

## 6. Decisions log

- 2026-09-24: retire SenseiBot's trainer after v11; rebuild on Prometheus in `SensAI`.
- 2026-09-24: keep continuous actions (user's call); the squashed-Gaussian head replaces SenseiBot's.
- 2026-09-24: use GPU physics on the RTX 5090. *Reversed later the same day, see below.*
- 2026-09-24: horizon follows Seer: half-life 10 s annealed to 20 s (gamma 0.99539 → 0.99769), with the
  schedule frozen per run (user's call).
- 2026-09-24: SensAI's goal includes 2v2. Start in 1v1; keep all 2v2 code, gated by config profile
  rather than removed (user's call).
- 2026-09-24: keep SensAI Studio as the front end, driving Prometheus's `Learner` through a headless
  trainer; do not use Prometheus's ImGui GUI (user's call).
- 2026-09-24: Studio, the eval suite and the probes move into the SensAI repo; SenseiBot becomes the
  archive (user's call).
- 2026-09-24: Phase 0 checked on this machine: GPU, CUDA 12.8 and VS 2022 ready; Python 3.11, LibTorch
  cu128 and CUDA's Visual Studio integration still to install. Phase 0's test list corrected for the
  standalone `RocketSimCuda` build.
- 2026-09-24: SensAI forked (`e4d097d` + sm_120 flags `77ec197` + `build_rscuda.bat` and
  `tools/phase0_checks.ps1` `b30d879`); both builds succeed with LibTorch 2.10.0+cu128 and Python 3.11.
- 2026-09-24: **Phase 0 physics parity passes on the 5090**, with three documented exceptions
  (logs in SensAI `run_logs/phase0/`). GAE parity passes; CPU↔GPU comparison passes every scenario but
  one.
  1. `RocketSimCudaTest` ball drop (5 checks): the test is wrong, not the physics. It sets the ball
     with exactly zero velocity, which the physics treats as asleep, copying RocketSim
     (`Collision.cuh:1439`). The comparison's ball drops match CPU at 1, 120 and 1,440 ticks.
  2. Game-state bridge, ball position after a goal (1 of 79 checks): the goal-interior geometry gap
     upstream already documents. The goal event and ball velocity match, and the episode has ended by
     then.
  3. `front_flip_60`, heading 13.6° off (3° allowed): `RocketSimCudaTraceDivergence` shows CPU and GPU
     identical through the jump, the flip and the flight (ticks 0–28, within 0.002 UU). They split at
     the first ground contact (tick 29), where the car lands nose-down on the exact kickoff diagonal and
     the two simulations break the symmetry in opposite directions. This is the exact tie upstream
     documents (`compare_cpu_cuda.cpp:491`), not a flip-physics difference.
- 2026-09-24: **Phase 0 throughput measured** (SensAI `tools/phase0_throughput.ps1`: fresh weights
  per setup, no old-version games, 4 min each, median after 3 warm-up iterations; 200k steps per
  iteration; Core Ultra 9 285K, 24 cores). Upstream's command-line `train` mode could not start with
  its own padded observation; SensAI `76307d1`/`a88ff53`/`78376c6` add `cpu-obs`, `unpadded`,
  `gpu-rewards` and `players=`.

  | setup | overall steps/s | collection | segment per iteration |
  |---|---|---|---|
  | round 1, 2v2, upstream rewards, CPU physics, 1024 games | **150,671** | 253,131 | ~49 steps |
  | round 1, 2v2, GPU physics (any obs), 2048–4096 games | 89–100k | 116–126k | |
  | round 2, 1v1, GPU-supported rewards, CPU physics, 1024 games | 110,085 | 180,829 | ~98 steps |
  | round 2, 1v1, CPU physics, 2048 games | **122,953** | 207,570 | ~49 steps |
  | round 2, 1v1, GPU physics, padded obs built on CPU, 4096 games | 82,901 | 114,290 | ~24 steps |
  | round 2, 1v1, fully on GPU (unpadded obs, GPU rewards), 4096 games | 93,035 | 160,436 | ~24 steps |
  | round 2, 1v1, fully on GPU, 8192 games | 127,841 | 187,963 | ~12 steps |
  | round 2, 1v1, fully on GPU, 16384 games | fails: 32,768 truncation points > 20,000-step minibatch | | ~6 steps |

  Findings:
  1. Learning (PPO) runs at 280–650k steps/s; collection is the limit everywhere.
  2. **CPU physics is the fastest usable setup.** The GPU only catches up (128k vs 123k) when
     *everything* is on the GPU, which needs the unpadded observation and GPU-supported rewards, and
     8,192 games. At a fixed 200k steps per iteration, 16,384 players leave each player a ~12-step
     segment per iteration, so most of GAE's horizon (λ 0.975 ≈ 40 steps; a 10 s half-life ≈ 150
     steps) is bootstrapped from the critic. Matching CPU's segment length would mean ~4× larger
     iterations, a learning-setup change rather than a free speed-up.
  3. Keeping the state copy (padded observation, or any reward without a GPU version) makes GPU
     physics slower than CPU physics.
  4. ~120–150k steps/s is ~15–19× SenseiBot's ~8k: 1B steps ≈ 2–2.5 h, 5B ≈ 10–12 h.
- 2026-09-24: **train on CPU physics; the 5090 runs the network** (user's call, reversing the GPU
  physics decision above). This keeps `AdvancedObsPadded`, so one network carries from 1v1 to 2v2, and
  lets rewards be any C++ term, not only those with a RocketSimCuda version. Extending the GPU observation
  code to build the padded layout is no longer needed. GPU physics stays built and parity-tested, available if a later run raises steps
  per iteration enough to keep segments long at 8k+ games. Phase 0's gate is passed.
- 2026-09-24: run 1 profile choices (user's calls): horizon held at T = 10 s until improvement
  plateaus, then reassessed as a continuation run; `GoalReward` concede scale stays −0.8; empty player
  slots zero-filled. Kickoff: `KickoffProximityReward` (state-based, zero-sum in 1v1, not
  potential-based) at a proposed 0.1. Layout in SensAI `docs/profile.md`.
- 2026-09-24: **Phase 1 passed.** Run 1's spec (SensAI `docs/run1_spec.md`) sets evaluation every 25M
  steps pooled over ±50M, milestones at 50M, 250M and 1B, a plateau check every 500M (the latest window
  not clearly better than the window 1B earlier on head to head vs v5, Nexto GD or Necto GD), adoption
  checks from 2B and a 5B cap (user's calls). Whether adoption must also match v11 is deferred until
  run 1 has started.
- 2026-09-25: **run 1 stopped at ~4.16B steps, not adopted** (user's call on the evidence in SensAI
  `docs/run1_spec.md` §8). Run 1b dropped. The next run is `headcheck_1v1`: run 1 with Gaussian-only
  analog entropy, to confirm the cause before anything else changes.
- 2026-09-25: critic warmup for continuation runs recorded as an open question (risks table); it has
  no effect on from-scratch runs.
- 2026-09-25: `headcheck_1v1` stopped at ~201M: switching off the entropy's squash term did not
  unstick throttle and steer. The revised cause is the fresh policy's 50% jump and handbrake, which
  keeps the car airborne from the start. Next is `headcheck2_1v1` (grounded start: button logit
  biases and a low starting spread).
- 2026-09-25: `headcheck2_1v1` stopped at ~158M: the grounded start unlocked throttle and steer, but
  entropy scale 0.025 pulls every weak-signal channel back to maximum entropy. `headcheck3_1v1`
  (scale 0.01, nothing else changed) is next. The live-config replay bug is fixed.
- 2026-09-25: `headcheck3_1v1` stopped at ~222M: a lower entropy scale did not help, and it exposed
  that no SensAI checkpoint steers toward the ball. `headcheck4_1v1` adds a policy warmup (also the
  critic warmup for continuation runs raised on 2026-09-24). If steering still fails, decide between
  continuous and discrete actions.
- 2026-09-25: `headcheck4_1v1` stopped at ~115M. The warmup fixed the first updates, not the steering;
  the one-way steer is the fresh output layer's and was never learned away. **Discrete actions next
  (user's call)**, as the teacher for the continuous head through GigaLearn's transfer learning, the
  form of the behavioural cloning cut in the rebuild. This amends the 2026-09-24 "keep continuous"
  decision: continuous stays the end goal, but not as the from-scratch learner.
