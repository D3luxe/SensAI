"""
Behaviour diagnostics from the trainer's per-iteration telemetry (logs/history.jsonl).

Everything here reads the current reward run only (utils/run_history.py): mean reward and losses
change scale with the reward version, and behaviour under v2 says nothing about v3's policy.

  run_telemetry(window)        mean telemetry over the last `window` iterations of the run, and over
                               its first `window` (what the policy did when the run started)
  render_behaviour_plot        now vs run start, controls and pitch/state side by side
  behaviour_flags_markdown     habits worth a look, each pointing at the eval metric that measures
                               it. Rewards are frozen per version (docs/reward_v3_spec.md, R1/R7), so
                               there are no weight nudges: a finding is checked on the eval suite
                               and, if real, becomes a candidate for the next version.
  render_training_curves_plot  reward, losses, entropy, throughput for the run / recent / all history

Telemetry is sampled from the training rollouts (every opponent type mixed together), so it
describes training behaviour, not a match against Necto. The eval suite measures that.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import run_history

BG, PANEL, GRID, TEXT, MUTED = "#0f172a", "#111c2e", "#64748b", "#e2e8f0", "#94a3b8"

CONTROLS = [
    ("throttle_forward_pct", "Throttle forward"), ("throttle_reverse_pct", "Reversing"),
    ("steer_left_pct", "Steering left"), ("steer_right_pct", "Steering right"),
    ("jump_rate_pct", "Jump held"), ("boost_rate_pct", "Boost held"), ("handbrake_rate_pct", "Handbrake held"),
]
STATE = [
    ("defensive_third_pct", "In own third"), ("midfield_third_pct", "In midfield"),
    ("offensive_third_pct", "In attacking third"), ("corner_zone_pct", "In a corner"),
    ("air_time_pct", "Airborne"), ("zero_boost_pct", "Empty boost"), ("mean_boost_tank", "Mean boost (0-100)"),
]


def _style(ax):
    ax.set_facecolor(PANEL)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#334155")
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.3, color=GRID)


def _mean_telemetry(records: List[Dict[str, Any]]) -> Dict[str, float]:
    tel = [r["telemetry"] for r in records if isinstance(r.get("telemetry"), dict)]
    if not tel:
        return {}
    out = {}
    for k in tel[0]:
        vals = [t[k] for t in tel if isinstance(t.get(k), (int, float))]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def run_telemetry(window: int = 10, history_file: str = run_history.HISTORY_FILE) -> Dict[str, Any]:
    run = run_history.current_run(history_file)
    recs = run["records"]
    if not recs:
        return {}
    version, start = run["key"] or ("?", 0)
    return {
        "now": _mean_telemetry(recs[-window:]),
        "start": _mean_telemetry(recs[:window]) if len(recs) >= 2 * window else {},
        "window": window,
        "version": version,
        "run_steps": int(recs[-1].get("global_step", 0)) - int(start or 0),
        "iteration": recs[-1].get("iteration", 0),
        "iterations_in_run": len(recs),
    }


def render_behaviour_plot(tel: Dict[str, Any]) -> plt.Figure:
    plt.close("all")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), dpi=100)
    fig.patch.set_facecolor(BG)
    now, start = tel.get("now") or {}, tel.get("start") or {}
    if not now:
        for ax in axes:
            ax.set_axis_off()
        fig.text(0.5, 0.5, "No telemetry for this run yet: it appears after the first training iterations.",
                 ha="center", va="center", color=MUTED, fontsize=11)
        return fig
    for ax, items, title in ((axes[0], CONTROLS, "Controls (% of steps)"), (axes[1], STATE, "Where it is (% of steps)")):
        _style(ax)
        y = np.arange(len(items))
        nv = [now.get(k, 0.0) for k, _ in items]
        ax.barh(y + (0.2 if start else 0), nv, height=0.38 if start else 0.6, color="#38bdf8", label="now")
        if start:
            sv = [start.get(k, 0.0) for k, _ in items]
            ax.barh(y - 0.2, sv, height=0.38, color="#475569", label="run start")
        for yi, v in zip(y, nv):
            ax.text(v + 1, yi + (0.2 if start else 0), f"{v:.0f}", va="center", color=TEXT, fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels([lbl for _, lbl in items], color=TEXT, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_title(title, color=TEXT, fontsize=10, loc="left")
        ax.grid(axis="y", visible=False)
    if start:
        axes[1].legend(loc="lower right", facecolor=BG, edgecolor="#334155", labelcolor=TEXT, fontsize=8)
    fig.tight_layout()
    return fig


# (telemetry key, test(now) -> bool, headline, what to check)
_FLAGS = [
    ("corner_zone_pct", lambda v: v > 20.0, "Spends {v:.0f}% of steps in the corners",
     "Goals-against causes and touches/min on the Evaluation page."),
    ("zero_boost_pct", lambda v: v > 35.0, "Empty boost {v:.0f}% of the time",
     "Boosting while retreating and the retreat scenario on the Evaluation page (T5 values boost held)."),
    ("throttle_reverse_pct", lambda v: v > 25.0, "Reversing {v:.0f}% of the time",
     "Time to first touch in the scenarios; reversing is sometimes right (backwards retreats)."),
    ("jump_rate_pct", lambda v: v > 60.0, "Jump held {v:.0f}% of steps",
     "Landing quality (landings not on wheels, speed kept after landing)."),
    ("air_time_pct", lambda v: v < 3.0, "Almost never airborne ({v:.1f}% of steps)",
     "Scenario first-touch rates on drops and bounces."),
]


def behaviour_flags_markdown(tel: Dict[str, Any]) -> str:
    if not tel or not tel.get("now"):
        return "#### Waiting for telemetry\nFlags appear after the first iterations of the run."
    now, start = tel["now"], tel.get("start") or {}
    lines = [f"#### Reward {tel['version']} run · {tel['run_steps'] / 1e6:,.1f}M steps · "
             f"last {tel['window']} iterations"]
    flagged = False
    for key, test, head, check in _FLAGS:
        v = now.get(key)
        if v is None or not test(v):
            continue
        flagged = True
        was = f" (was {start[key]:.0f}% at run start)" if key in start else ""
        lines.append(f"- **{head.format(v=v)}**{was}. Check: {check}")
    lr = abs(now.get("steer_left_pct", 0.0) - now.get("steer_right_pct", 0.0))
    if lr > 25.0:
        flagged = True
        lines.append(f"- **Steering is one-sided** ({now.get('steer_left_pct', 0):.0f}% left vs "
                     f"{now.get('steer_right_pct', 0):.0f}% right). The trainer augments with left-right mirrored "
                     "samples, so a lasting imbalance is unusual and worth a look.")
    if not flagged:
        lines.append("- Nothing out of range.")
    lines.append("\n<span style='color:#94a3b8'>Rewards are frozen for the run, so these are things to check on "
                 "the eval suite, not dials to turn. A confirmed problem becomes a candidate for the next reward "
                 "version (spec §8).</span>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------------------------
def _thin(recs: List[Dict[str, Any]], n: int = 500) -> List[Dict[str, Any]]:
    if len(recs) <= n:
        return recs
    idx = np.linspace(0, len(recs) - 1, n).astype(int)
    return [recs[i] for i in idx]


def curve_records(mode: str = "run", history_file: str = run_history.HISTORY_FILE) -> Tuple[List[Dict[str, Any]], str]:
    if mode == "recent":
        return run_history.recent(history_file, 100), "last 100 iterations"
    if mode == "full":
        return run_history.sampled(history_file, 400), "all history (sampled; mixes reward versions)"
    run = run_history.current_run(history_file)
    v = run["key"][0] if run["key"] else "?"
    return _thin(run["records"]), f"reward {v} run"


def render_training_curves_plot(history_file: str = run_history.HISTORY_FILE, mode: str = "run") -> plt.Figure:
    plt.close("all")
    recs, scope = curve_records(mode, history_file)
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), dpi=100, sharex=True)
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        _style(ax)
    if not recs:
        for ax in axes.flat:
            ax.set_axis_off()
        fig.text(0.5, 0.5, "Waiting for training data", ha="center", va="center", color=MUTED, fontsize=11)
        return fig

    x_is_run = mode == "run" and recs[0].get("reward_run_start_step") is not None
    start = int(recs[0].get("reward_run_start_step") or 0)
    xs = np.array([(r.get("global_step", 0) - start) / 1e6 if x_is_run else r.get("iteration", 0) for r in recs])
    xlabel = "M steps into the run" if x_is_run else "iteration"

    def series(key):
        return np.array([float(r.get(key, np.nan) or 0.0) for r in recs])

    def smooth(y, k=9):
        if len(y) < k:
            return y
        pad = np.pad(y, (k // 2, k // 2), mode="edge")
        return np.convolve(pad, np.ones(k) / k, mode="valid")

    panels = [
        (axes[0, 0], "mean_reward", "Mean episode reward", "#38bdf8"),
        (axes[0, 1], "value_loss", "Value loss", "#facc15"),
        (axes[1, 0], "entropy", "Policy entropy", "#c084fc"),
        (axes[1, 1], "ball_touches", "Touches per episode", "#4ade80"),
    ]
    warm = np.array([bool(r.get("critic_warmup")) for r in recs])
    for ax, key, title, colour in panels:
        y = series(key)
        ax.plot(xs, y, color=colour, alpha=0.3, linewidth=1)
        ax.plot(xs, smooth(y), color=colour, linewidth=1.8)
        ax.set_title(title, color=TEXT, fontsize=9.5, loc="left")
        if warm.any():
            ax.axvspan(xs[warm].min(), xs[warm].max(), color="#a78bfa", alpha=0.12, lw=0)
    sps = series("sps")
    axes[1, 1].text(0.99, 0.04, f"{np.nanmean(sps[-20:]):,.0f} steps/s", transform=axes[1, 1].transAxes,
                    ha="right", color=MUTED, fontsize=8)
    for ax in axes[1]:
        ax.set_xlabel(xlabel, color=MUTED, fontsize=8)
    note = scope + (" · purple: critic warm-up (policy frozen)" if warm.any() else "")
    fig.text(0.995, 0.005, note, ha="right", va="bottom", color="#64748b", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    return fig
