# 🏎️⚽ SenseiBot - Rocket League ML Bot Studio & Training Environment

A complete Reinforcement Learning training environment and management dashboard for Rocket League bots, featuring **Vectorized Proximal Policy Optimization (PPO)**, a headless **3D Rocket League physics simulator**, **real-time dynamic parameter adjustment**, and an interactive **Gradio GUI**.

---

## 🌟 Key Features

* **3D Headless Arena Simulation (`env/physics_engine.py`)**:
  * Realistic car kinematics: throttle, steering curve, air pitch/yaw/roll, jumping, flipping/dodging, powersliding/drifting, and boost acceleration.
  * Ball aerodynamics with bounce restitution, drag, and goal net scoring detection.
  * 34 standard boost pads (6 Big Pads + 28 Small Pads) with active timers and collection radii.
  * Multi-agent support (1v1, 2v2, 3v3) with symmetric coordinate inversion.
* **High-Throughput Vectorized PPO (`agent/ppo.py`)**:
  * Multi-environment parallel rollout collection.
  * Generalized Advantage Estimation (GAE-$\lambda$).
  * Actor-Critic PyTorch networks with orthogonal initialization.
  * Integrated TensorBoard and JSON streaming for metrics.
* **Live Dynamic Parameter Tuning (No Restarts Required)**:
  * Adjust reward weights, learning rate, and entropy coefficients via IPC while training is actively executing.
* **Interactive Gradio Dashboard (`ui/app.py` / `app.py`)**:
  * **Controls**: Start, Pause, Resume, Stop, and Trigger Checkpoints without freezing the UI.
  * **Live Reward Sliders**: Real-time tuning for Ball Touches, Speed to Ball, Face Ball Alignment, Scoring/Conceding, Saves, Boost Management, and Aerials.
  * **Real-Time KPIs & Graphs**: Live plots of Mean Reward, Policy/Value Loss, Ball Touches, and Policy Entropy.
  * **Console Stream**: Real-time stdout/stderr log viewer.
  * **2D Pitch Match Visualizer**: Simulate trained checkpoint models against opponents and view 2D trajectories and scoring breakdowns.

---

## 📁 Repository Structure

```
SenseiBot/
├── config/
│   ├── default_config.yaml     # Baseline hyperparameters, rewards, and env settings
│   └── live_config.json        # Shared IPC file for runtime parameter modification
├── env/
│   ├── __init__.py
│   ├── physics_engine.py       # 3D headless Rocket League physics simulator
│   ├── rocket_env.py           # Gymnasium-compatible vectorized environment
│   ├── rewards.py              # Modular and weighted reward functions
│   ├── observations.py         # Symmetric observation builders & local frame transforms
│   └── actions.py              # Continuous and discrete action parsers
├── agent/
│   ├── __init__.py
│   ├── models.py               # Actor-Critic PyTorch neural networks
│   └── ppo.py                  # Vectorized PPO trainer with live config reload
├── utils/
│   ├── __init__.py
│   ├── process_manager.py      # Non-blocking training subprocess & IPC manager
│   └── visualizer.py           # 2D pitch visualizer and match replay generator
├── ui/
│   ├── __init__.py
│   └── app.py                  # Gradio management interface
├── checkpoints/                # Saved model weights (.pt)
├── logs/                       # TensorBoard and JSON training logs
├── train.py                    # Standalone CLI training script
├── evaluate.py                 # Evaluation & match benchmarking script
├── app.py                      # Main Gradio application launcher
└── README.md                   # System documentation
```

---

## 🚀 Quickstart Guide

### 1. Launch the Gradio Studio UI
Start the web dashboard by running:
```bash
python app.py
```
Then open your browser at **`http://127.0.0.1:7860`**.

### 2. Start Training via CLI (Optional)
If you prefer running training directly from the command line:
```bash
# Standard training run
python train.py

# Specify custom config or max iterations
python train.py --config config/default_config.yaml --iterations 100

# Resume from a checkpoint
python train.py --checkpoint checkpoints/latest_model.pt
```

### 3. Evaluate a Trained Bot
Benchmark your trained policy across 10 matches and save the visual field trajectory:
```bash
python evaluate.py --model checkpoints/latest_model.pt --episodes 10 --save-plot logs/match_eval.png
```

### 4. Launch TensorBoard
To view detailed scalar curves in TensorBoard:
```bash
tensorboard --logdir=logs --port=6006
```

---

## 🎛️ Reward Function Tuning Guide

| Reward Component | Default Weight | Recommended Range | Purpose |
| :--- | :--- | :--- | :--- |
| **`behind_ball_weight`** | `1.5` | `0.5 - 4.0` | **Goal-Side Positioning**: Rewards staying between ball and own net. Stops own-goals and bad backward hits |
| **`possession_weight`** | `2.0` | `0.5 - 5.0` | **Ball Control & Dribbling**: Rewards carrying and matching ball speed (<350 distance) instead of booming |
| **`defensive_position_weight`**| `1.0` | `0.5 - 3.0` | **Goalkeeping**: Rewards positioning on the line between ball and net when defending |
| **`demo_bump_weight`** | `2.0` | `0.5 - 5.0` | **Physical Play**: Rewards supersonic bumps & demolitions against opponents |
| **`boost_steal_weight`** | `1.0` | `0.5 - 3.0` | **Boost Starvation**: Rewards collecting 100-pads on opponent's half |
| **`touch_ball_weight`** | `5.0` | `2.0 - 15.0` | Primary incentive for establishing contact with the ball |
| **`speed_toward_ball_weight`**| `1.5` | `0.5 - 5.0` | Encourages aggressive closing speed toward the ball |
| **`face_ball_weight`** | `0.5` | `0.1 - 2.0` | Directs the car nose toward the ball |
| **`aligned_shot_weight`** | `2.0` | `1.0 - 5.0` | Incentivizes hitting the ball directly toward the goal |
| **`aerial_height_weight`** | `0.8` | `0.2 - 3.0` | Encourages jumping and aerial maneuvers |
| **`goal_weight`** | `20.0` | `10.0 - 50.0` | Large reward for scoring in the opponent's net |
| **`goal_speed_multi`** | `2.0` | `0.0 - 5.0` | **Power Shot Multiplier**: Scales goal reward by shot velocity (up to (1 + multi)x for supersonic goals) |
| **`concede_weight`** | `-20.0` | `-50.0 - -10.0`| Large penalty when the opponent scores |
| **`save_weight`** | `10.0` | `5.0 - 25.0` | Reward for clearing the ball from the defensive net |
| **`boost_management_weight`** | `0.2` | `0.05 - 1.0` | Encourages boost pad collection and boost conservation |
| **`velocity_weight`** | `0.1` | `0.05 - 1.0` | Encourages maintaining high general speed |
