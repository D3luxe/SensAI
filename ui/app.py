"""
SensAI Studio: the Gradio dashboard for training, evaluating and diagnosing the bot.

  Training     run controls, curves for the current reward run, the frozen reward version, the few
               settings that apply live (optimiser, fixed opponents, league grading budget)
  Evaluation   the eval suite (scripts/eval_suite.py): run it, read a result against the version's
               baseline, follow the headline metrics over the run
  League       the self-play league's standings and a manual TrueSkill tournament
  Diagnostics  behaviour telemetry, a match viewer, tests and a pasteable system snapshot
  Setup        training config, replays and pretraining, the custom scenario library

Reward weights and the scenario mix have no controls: they belong to the frozen reward version
(config/reward_versions/<v>.json, env/reward_registry.py).
"""

from __future__ import annotations
import os
import sys
import glob
import time
import datetime
import math
import re
import json
import yaml
import torch
import gradio as gr
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Any, Dict, Optional, Union
import html

from utils.process_manager import TrainingProcessManager
from utils.visualizer import simulate_match
from utils.replay_parser import ReplayParser, DEFAULT_DEMO_DIR, get_default_demo_dir
from agent.pretrainer import BehavioralCloningTrainer
from utils.test_runner import run_all_unit_tests, get_cached_or_run_tests, format_test_results_markdown
from utils.diagnostics import (
    behaviour_flags_markdown,
    render_behaviour_plot,
    render_training_curves_plot,
    run_telemetry,
)
from utils import eval_results
from utils.scenario_manager import (
    ScenarioManager,
    render_scenario_visual_guide,
    simulate_custom_scenario,
    DEFAULT_CUSTOM_SCENARIOS
)
from ui import league_board
from ui.league_board import LEAGUE_CSS
from utils.trueskill_evaluator import TrueSkillEvaluator, get_model_display_name
from utils.league_manager import snap_tiers_to_worker_slices
from utils.config import effective_config
from env.reward_registry import active_version, load_snapshot


def load_yaml_config(path: str = "config/default_config.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f)
    return {}


def save_yaml_config(cfg: dict, path: str = "config/default_config.yaml"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)


def get_available_checkpoints() -> list:
    pts = glob.glob("checkpoints/*.pt") + glob.glob("checkpoints/**/*.pt")
    # Normalize paths and eliminate duplicates
    pts = list(dict.fromkeys([os.path.normpath(p).replace("\\", "/") for p in pts]))
    if not pts:
        return ["checkpoints/latest_model.pt (none saved yet)"]
    return sorted(pts, key=lambda x: os.path.getmtime(x) if os.path.exists(x) else 0.0, reverse=True)


def get_available_opponent_options() -> list:
    base_options = ["Heuristic Chaser (Rule-Based Aggressive Bot)"]
    ckpts = get_available_checkpoints()
    return base_options + ckpts


CUSTOM_CSS = """
/* Futuristic Cyber / Rocket League Dark Theme */
.gradio-container {
    max-width: 1520px !important;
    margin: auto !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
}

.hero-status-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 16px 22px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    margin-bottom: 12px;
}

.status-badge-running {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    background: rgba(34, 197, 94, 0.15);
    color: #4ade80;
    border: 1px solid #22c55e;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.9em;
    letter-spacing: 0.5px;
}

.status-badge-paused {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    background: rgba(234, 179, 8, 0.15);
    color: #facc15;
    border: 1px solid #eab308;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.9em;
    letter-spacing: 0.5px;
}

.status-badge-stopped {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 14px;
    background: rgba(148, 163, 184, 0.15);
    color: #94a3b8;
    border: 1px solid #64748b;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.9em;
    letter-spacing: 0.5px;
}

button.primary-btn {
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

.action-bar-row {
    align-items: center !important;
    gap: 10px !important;
}

.cyber-panel {
    background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px 20px;
    margin-bottom: 14px;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
}

.status-callout-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-left: 4px solid #38bdf8;
    border-radius: 6px;
    padding: 10px 16px;
    margin-top: 10px;
    font-size: 0.95em;
    color: #f1f5f9;
    min-height: 44px;
    display: flex;
    align-items: center;
    box-shadow: inset 0 1px 3px rgba(0,0,0,0.4);
}

/* Clean Modern Tab Navigation */
.tabs > .tab-nav {
    border-bottom: 1px solid #334155 !important;
    gap: 8px !important;
    padding-bottom: 4px !important;
    margin-bottom: 16px !important;
    flex-wrap: nowrap !important;
    overflow-x: auto !important;
}

.tabs > .tab-nav > button {
    font-size: 0.98em !important;
    font-weight: 600 !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 10px 22px !important;
    transition: all 0.2s ease !important;
    color: #94a3b8 !important;
    white-space: nowrap !important;
}

.tabs > .tab-nav > button.selected {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: rgba(56, 189, 248, 0.08) !important;
}

/* Sleek, un-cluttered slider & input labels (eliminate giant opaque blue button badges) */
.gradio-container .block label,
.gradio-container .block label span,
.gradio-container .block-label,
.gradio-container span.block-label,
.gradio-container [data-testid="block-label"],
.gradio-container span[data-testid="block-info"],
.gradio-container .label-wrap span,
.gradio-container .gradio-slider label,
.gradio-container .gradio-slider .head span {
    background: transparent !important;
    background-color: transparent !important;
    color: #93c5fd !important;
    font-size: 0.88em !important;
    font-weight: 600 !important;
    padding: 0 !important;
    margin-bottom: 2px !important;
    white-space: nowrap !important;
    border: none !important;
    box-shadow: none !important;
}

.gradio-slider {
    padding: 4px 6px !important;
}

/* ============================================================================
   LEAGUE BOARD  --  shared pieces. The board itself is ui/league_board.py (LEAGUE_CSS).
   Animations are transform/opacity only (compositor-driven, no layout or paint
   per frame) and every one of them is disabled under prefers-reduced-motion.
   ========================================================================= */

.league-board { display: flex; flex-direction: column; gap: 10px; margin-bottom: 10px; }

.lb-panel {
    background: linear-gradient(160deg, rgba(17, 24, 39, 0.92) 0%, rgba(9, 13, 22, 0.96) 100%);
    border: 1px solid #1f2a3d;
    border-radius: 10px;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
}

.lb-panel-head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; flex-wrap: wrap;
    padding: 9px 16px;
    border-bottom: 1px solid rgba(51, 65, 85, 0.55);
}

.lb-panel-title {
    font-size: 0.82em; font-weight: 800; letter-spacing: 1.1px;
    text-transform: uppercase; color: #e2e8f0;
}

.lb-panel-meta { font-size: 0.78em; color: #7c8ba1; letter-spacing: 0.3px; }
.lb-panel-meta b { color: #38bdf8; }

/* ---- King banner ---------------------------------------------------- */

.lb-king {
    position: relative; overflow: hidden;
    display: flex; align-items: center; justify-content: space-between;
    gap: 18px; flex-wrap: wrap;
    padding: 16px 22px;
    border: 1px solid rgba(234, 179, 8, 0.45);
    border-left: 5px solid #eab308;
    border-radius: 10px;
    background:
        radial-gradient(120% 180% at 0% 0%, rgba(234, 179, 8, 0.16) 0%, rgba(234, 179, 8, 0) 55%),
        linear-gradient(160deg, rgba(24, 30, 45, 0.96) 0%, rgba(9, 13, 22, 0.98) 100%);
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.4);
}

/* Slow diagonal sheen. One transform on a pseudo-element; no repaint. */
.lb-king::after {
    content: ""; position: absolute; top: -60%; left: -30%;
    width: 40%; height: 220%;
    background: linear-gradient(100deg, rgba(255,255,255,0) 0%, rgba(255,255,255,0.05) 50%, rgba(255,255,255,0) 100%);
    transform: translateX(-140%) rotate(8deg);
    animation: lb-sheen 7s ease-in-out infinite;
    pointer-events: none;
}

@keyframes lb-sheen {
    0%, 72% { transform: translateX(-140%) rotate(8deg); }
    100%    { transform: translateX(420%) rotate(8deg); }
}

.lb-king-id { display: flex; align-items: center; gap: 14px; min-width: 0; }

.lb-crown {
    font-size: 1.9em; line-height: 1;
    filter: drop-shadow(0 0 10px rgba(234, 179, 8, 0.55));
    animation: lb-crown-float 4.5s ease-in-out infinite;
}

@keyframes lb-crown-float {
    0%, 100% { transform: translateY(0); }
    50%      { transform: translateY(-3px); }
}

.lb-king-label {
    font-size: 0.7em; font-weight: 800; letter-spacing: 1.4px;
    text-transform: uppercase; color: #eab308;
}

.lb-king-name {
    font-size: 1.45em; font-weight: 900; color: #f8fafc;
    letter-spacing: -0.3px; line-height: 1.2;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}

.lb-king-sub { font-size: 0.8em; color: #94a3b8; margin-top: 1px; }

.lb-king-stats { display: flex; gap: 26px; flex-wrap: wrap; }

.lb-stat { text-align: right; }
.lb-stat-label {
    display: block; font-size: 0.66em; font-weight: 800; letter-spacing: 1px;
    text-transform: uppercase; color: #64748b; margin-bottom: 2px;
}
.lb-stat-value { font-size: 1.3em; font-weight: 800; color: #f1f5f9; font-variant-numeric: tabular-nums; }
.lb-stat-value.accent { color: #38bdf8; }
.lb-stat-value.good   { color: #4ade80; }
.lb-stat-value.warn   { color: #facc15; }

/* ---- Confidence chip ------------------------------------------------ */

.lb-chip {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.66em; font-weight: 800; letter-spacing: 0.8px;
    text-transform: uppercase; padding: 2px 8px; border-radius: 9999px;
    white-space: nowrap;
}
.lb-chip-ranked      { background: rgba(56, 189, 248, 0.14); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.45); }
.lb-chip-provisional { background: rgba(148, 163, 184, 0.12); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.35); }
.lb-chip-anchor      { background: rgba(168, 85, 247, 0.14); color: #c4b5fd; border: 1px solid rgba(168, 85, 247, 0.4); }
/* A locked rating is a checkpoint that earned anchor treatment: fixed, still playing. */
.lb-chip-locked      { background: rgba(148, 163, 184, 0.16); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.38); }
.lb-chip-king        { background: rgba(234, 179, 8, 0.16); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.5); }

/* ---- "How it works" explainer ---------------------------------------- */

/* CSS-only disclosure: no JS, no per-frame cost. Opens on hover and on keyboard
   focus, so it is reachable without a pointer. */
.hiw { position: relative; display: inline-flex; outline: none; }

.hiw-chip {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 2px 9px; border-radius: 9999px; cursor: help;
    background: rgba(56, 189, 248, 0.1);
    border: 1px solid rgba(56, 189, 248, 0.3);
    color: #7dd3fc; font-weight: 700; letter-spacing: 0.4px;
    white-space: nowrap;
    transition: background-color 0.15s ease, border-color 0.15s ease;
}
.hiw:hover .hiw-chip,
.hiw:focus-visible .hiw-chip {
    background: rgba(56, 189, 248, 0.18);
    border-color: rgba(56, 189, 248, 0.55);
}

.hiw-pop {
    position: absolute; top: calc(100% + 8px); left: 0; z-index: 40;
    width: min(430px, 78vw);
    display: flex; flex-direction: column; gap: 9px;
    padding: 13px 15px;
    background: #0d1421;
    border: 1px solid #2b3a52; border-radius: 10px;
    box-shadow: 0 12px 30px rgba(0, 0, 0, 0.55);
    opacity: 0; visibility: hidden; transform: translateY(-4px);
    transition: opacity 0.16s ease, transform 0.16s ease, visibility 0.16s;
    text-align: left; white-space: normal; cursor: default;
}
.hiw:hover .hiw-pop,
.hiw:focus-within .hiw-pop {
    opacity: 1; visibility: visible; transform: translateY(0);
}

.hiw-title {
    font-size: 1.02em; font-weight: 800; color: #f1f5f9;
    letter-spacing: 0.3px;
}

.hiw-step { display: flex; align-items: flex-start; gap: 10px; }
.hiw-step > div { display: flex; flex-direction: column; gap: 1px; min-width: 0; }
.hiw-step b { color: #7dd3fc; font-size: 0.97em; font-weight: 800; }
.hiw-step span { color: #94a3b8; line-height: 1.45; }

.hiw-num {
    flex-shrink: 0; width: 19px; height: 19px; margin-top: 1px;
    display: inline-flex; align-items: center; justify-content: center;
    border-radius: 50%; font-size: 0.82em; font-weight: 800;
    background: rgba(56, 189, 248, 0.14);
    border: 1px solid rgba(56, 189, 248, 0.4);
    color: #7dd3fc;
}

.hiw-foot {
    color: #64748b; font-style: italic; line-height: 1.45;
    border-top: 1px solid rgba(51, 65, 85, 0.5); padding-top: 8px;
}

/* Near the right edge the popover would overflow; flip its anchor. */
@media (max-width: 620px) {
    .hiw-pop { left: auto; right: 0; }
}

@media (prefers-reduced-motion: reduce) {
    .hiw-pop { transition: none; }
}

/* ---- Anchor legend --------------------------------------------------- */

.lb-anchor-legend {
    display: flex; align-items: center; gap: 8px; flex-wrap: wrap;
    width: 100%; box-sizing: border-box;
    padding: 6px 14px; margin: 0 0 10px;
    /* Kept as a guard, not a fix for an observed bug: standalone, the popover already
       paints above the King banner without it. Gradio wraps each HTML component in its
       own container, so this makes the legend's stacking explicit rather than relying
       on whatever the host's wrappers happen to do. */
    position: relative; z-index: 60;
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(168, 85, 247, 0.28);
    border-radius: 8px;
    font-size: 0.78em;
}

.lb-anchor-legend-label {
    font-weight: 800; letter-spacing: 0.9px; text-transform: uppercase;
    color: #c4b5fd;
}

.lb-anchor-chip {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 2px 9px; border-radius: 9999px;
    background: rgba(168, 85, 247, 0.12);
    border: 1px solid rgba(168, 85, 247, 0.3);
    color: #e2e8f0;
}

.lb-anchor-mu { color: #c4b5fd; font-variant-numeric: tabular-nums; }
.lb-anchor-note { color: #64748b; font-style: italic; margin-left: auto; white-space: nowrap; }

/* Below the width where the note can sit on the same line without crowding the
   chips, drop it rather than letting it push them onto another row. */
@media (max-width: 780px) {
    .lb-anchor-note { display: none; }
}

/* ---- Empty throne ---------------------------------------------------- */

.lb-king-empty {
    border-color: rgba(100, 116, 139, 0.4);
    border-left-color: #64748b;
    background:
        radial-gradient(120% 180% at 0% 0%, rgba(100, 116, 139, 0.12) 0%, rgba(100, 116, 139, 0) 55%),
        linear-gradient(160deg, rgba(24, 30, 45, 0.96) 0%, rgba(9, 13, 22, 0.98) 100%);
}
.lb-king-empty::after { animation: none; }
.lb-king-empty .lb-king-label { color: #94a3b8; }
.lb-king-empty .lb-king-name { color: #cbd5e1; font-size: 1.2em; }

.lb-crown-dim {
    filter: none; opacity: 0.55; animation: none;
}

/* ---- Standings shared pieces (rows are .lg-row, ui/league_board.py) --- */

.lb-rank {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.95em; font-weight: 800; color: #64748b; font-variant-numeric: tabular-nums;
}
.lb-rank-top { color: #facc15; }

.lb-name { font-weight: 700; color: #f1f5f9; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lb-num  { font-variant-numeric: tabular-nums; font-weight: 700; }
.lb-muted { color: #7c8ba1; font-variant-numeric: tabular-nums; }
.lb-record { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; color: #94a3b8; }

@keyframes lb-bar-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* ---- Progress bar (pipeline trials) ---------------------------------- */

.lb-prog-track { height: 7px; border-radius: 4px; background: rgba(51, 65, 85, 0.6); overflow: hidden; }
.lb-prog-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #0ea5e9 0%, #4ade80 100%);
    transform-origin: left center;
    animation: lb-bar-grow 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.lb-empty {
    padding: 16px 20px; text-align: center; color: #7c8ba1; font-size: 0.87em;
    border: 1px dashed #2b3a52; border-radius: 8px; margin: 12px 16px;
}

.lb-face-note { padding: 8px 16px 0; font-size: 0.78em; color: #7c8ba1; letter-spacing: 0.3px; }
.lb-face-note b { color: #38bdf8; }

/* ---- Benchmark scatter ---------------------------------------------- */

.bm-chart { padding: 14px 16px 10px; }
.bm-svg { width: 100%; height: auto; display: block; overflow: visible; }
.bm-grid { stroke: rgba(51, 65, 85, 0.45); stroke-width: 1; }
.bm-zero { stroke: rgba(148, 163, 184, 0.7); stroke-width: 1.2; }
.bm-axis-label { fill: #64748b; font-size: 9.5px; font-family: ui-monospace, monospace; }
.bm-axis-title { fill: #7c8ba1; font-size: 9.5px; letter-spacing: 0.5px; }
.bm-trend { fill: none; stroke-width: 1.5; opacity: 0.5; stroke-dasharray: 5 4; }
.bm-mark { stroke: #0b1220; stroke-width: 1; }
.bm-legend { display: flex; gap: 14px; flex-wrap: wrap; align-items: center;
    padding: 0 16px 12px; font-size: 0.78em; color: #94a3b8; }
.bm-legend-item { display: inline-flex; align-items: center; gap: 6px; }
.bm-legend-swatch { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
.bm-note { padding: 0 16px 14px; font-size: 0.76em; color: #64748b; line-height: 1.5; }
.bm-note b { color: #94a3b8; font-weight: 600; }

@media (prefers-reduced-motion: reduce) {
    .lb-king::after, .lb-crown, .lb-prog-fill { animation: none !important; }
    .lb-prog-fill { transform: none !important; }
}

/* ---------------------------------------------------------------------------------------------
   Studio layout (header, status bar, toolbars, hints)
   --------------------------------------------------------------------------------------------- */
.app-header { align-items: flex-end !important; gap: 14px !important; margin-bottom: 4px; }
.app-title { font-size: 1.35em; font-weight: 800; letter-spacing: 0.3px; color: #e2e8f0; margin: 2px 0 8px; }
.app-title span { color: #38bdf8; font-weight: 600; }
.header-controls button { min-height: 40px; }

.status-bar {
    display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px 22px;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid #334155; border-radius: 10px; padding: 10px 16px;
}
.sb-left { display: flex; align-items: center; gap: 10px; }
.sb-stats { display: flex; flex-wrap: wrap; gap: 6px 22px; }
.sb-item { display: flex; flex-direction: column; line-height: 1.15; }
.sb-k { font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.7px; color: #64748b; }
.sb-v { font-size: 0.98em; font-weight: 700; color: #f1f5f9; font-variant-numeric: tabular-nums; }
.sb-warm {
    padding: 3px 10px; border-radius: 999px; font-size: 0.78em; font-weight: 700;
    background: rgba(167, 139, 250, 0.15); color: #c4b5fd; border: 1px solid rgba(167, 139, 250, 0.45);
}
.sb-feedback { margin-top: 6px; font-size: 0.88em; color: #93c5fd; }

.toolbar { align-items: center !important; gap: 8px !important; }
.hint, .hint p { color: #94a3b8 !important; font-size: 0.88em !important; line-height: 1.45; }
.side-panel { border-right: 1px solid #1e293b; padding-right: 10px !important; }
.flags ul { padding-left: 1.1em; }
.flags li { margin-bottom: 6px; }

/* ---------------------------------------------------------------------------------------------
   Reward version card
   --------------------------------------------------------------------------------------------- */
.rw-card {
    background: linear-gradient(160deg, #111c2e 0%, #0b1322 100%);
    border: 1px solid #263349; border-radius: 10px; padding: 12px 14px; color: #cbd5e1; font-size: 0.9em;
}
.rw-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.rw-title { font-weight: 800; font-size: 1.1em; color: #f1f5f9; }
.rw-frozen {
    font-size: 0.72em; font-weight: 700; text-transform: uppercase; letter-spacing: 0.8px;
    padding: 2px 9px; border-radius: 999px; color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.45);
    background: rgba(56, 189, 248, 0.10); cursor: help;
}
.rw-table, .ev-table { width: 100%; border-collapse: collapse; margin-bottom: 10px; background: transparent !important; border: none !important; }
.rw-table td, .rw-table tr, .ev-table td, .ev-table th, .ev-table tr {
    border: none !important; background: transparent !important;
}
.rw-table td { padding: 4px 4px; border-bottom: 1px solid #1e293b !important; vertical-align: top; }
.rw-table tr:last-child td { border-bottom: none; }
.rw-w { font-weight: 700; color: #f1f5f9; font-variant-numeric: tabular-nums; white-space: nowrap; }
.rw-dim { color: #64748b; font-weight: 500; font-size: 0.92em; }
.rw-sub { font-size: 0.72em; text-transform: uppercase; letter-spacing: 0.7px; color: #64748b; margin: 4px 0 5px; }
.rw-bar { display: flex; height: 10px; border-radius: 5px; overflow: hidden; background: #1e293b; }
.rw-bar span { display: block; height: 100%; }
.rw-legend { display: flex; flex-wrap: wrap; gap: 3px 10px; margin-top: 6px; font-size: 0.82em; color: #94a3b8; }
.rw-leg i { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 4px; vertical-align: 0; }
.rw-foot { margin-top: 10px; font-size: 0.78em; color: #64748b; line-height: 1.5; }
.rw-foot code { font-size: 0.95em; }

/* ---------------------------------------------------------------------------------------------
   Eval scorecard and comparison tables
   --------------------------------------------------------------------------------------------- */
.ev-scorecard { display: flex; flex-direction: column; gap: 14px; }
.ev-meta { color: #cbd5e1; font-size: 0.92em; line-height: 1.5; }
.ev-sub { color: #64748b; font-size: 0.92em; }
.ev-warn { color: #fbbf24; font-weight: 700; }
.ev-section-title {
    font-size: 0.74em; text-transform: uppercase; letter-spacing: 0.8px; color: #64748b; font-weight: 700; margin-bottom: 6px;
}
.ev-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(210px, 1fr)); gap: 10px; }
.ev-card {
    background: #0f172a; border: 1px solid #1e293b; border-left: 4px solid #334155; border-radius: 8px; padding: 10px 12px;
}
.ev-card-primary { grid-column: span 2; }
.ev-card-primary .ev-value { font-size: 2.1em; }
.ev-edge-better { border-left-color: #22c55e; }
.ev-edge-worse { border-left-color: #f43f5e; }
.ev-edge-noise, .ev-edge-none { border-left-color: #334155; }
.ev-edge-neutral { border-left-color: #a78bfa; }
.ev-label { font-size: 0.82em; color: #94a3b8; line-height: 1.3; min-height: 2.1em; }
.ev-value { font-size: 1.6em; font-weight: 800; color: #f1f5f9; font-variant-numeric: tabular-nums; line-height: 1.2; }
.ev-base { font-size: 0.8em; color: #64748b; margin-top: 2px; }
.ev-spread { font-size: 0.74em; color: #475569; margin-top: 1px; }
.ev-d { font-weight: 700; }
.ev-better { color: #4ade80; }
.ev-worse { color: #fb7185; }
.ev-noise, .ev-none { color: #64748b; }
.ev-neutral { color: #c4b5fd; }
.ev-empty { color: #94a3b8; padding: 18px; border: 1px dashed #334155; border-radius: 8px; text-align: center; }
.ev-tables { display: grid; grid-template-columns: repeat(auto-fill, minmax(460px, 1fr)); gap: 16px; }
.ev-table { width: 100%; border-collapse: collapse; font-size: 0.88em; }
.ev-table th { text-align: left; color: #64748b; font-weight: 600; padding: 4px 6px; border-bottom: 1px solid #334155 !important; }
.ev-table td { padding: 4px 6px; border-bottom: 1px solid #1e293b !important; color: #cbd5e1; }
.ev-table .ev-n { text-align: right; font-variant-numeric: tabular-nums; }
.ev-table .ev-v { font-size: 0.85em; white-space: nowrap; }
.ev-row-better td:first-child, .ev-row-worse td:first-child { font-weight: 700; color: #f1f5f9; }
@media (max-width: 700px) { .ev-card-primary { grid-column: span 1; } .ev-tables { grid-template-columns: 1fr; } }
"""


CUSTOM_CSS += LEAGUE_CSS

def format_elapsed_time(seconds: Union[int, float]) -> str:
    sec = int(seconds or 0)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def _run_progress(metrics: dict) -> dict:
    """Where the current reward run is, from the trainer's latest metrics record."""
    version = metrics.get("reward_version") or active_version()
    start = metrics.get("reward_run_start_step")
    step = int(metrics.get("global_step", 0) or 0)
    run_steps = step - int(start) if start is not None else None
    return {"version": version, "run_steps": run_steps, "warmup": bool(metrics.get("critic_warmup"))}


def build_status_card_html(status_info: dict, feedback_msg: str = "") -> str:
    running = status_info.get("running", False)
    paused = status_info.get("paused", False)
    metrics = status_info.get("metrics", {}) or {}
    run = _run_progress(metrics)

    if running and not paused:
        badge = '<span class="status-badge-running">● Training</span>'
    elif running and paused:
        badge = '<span class="status-badge-paused">❚❚ Paused</span>'
    else:
        badge = '<span class="status-badge-stopped">○ Stopped</span>'
    run_txt = f"{run['run_steps'] / 1e6:,.1f}M steps" if run["run_steps"] is not None else "not started"
    items = [
        ("Reward", f"{run['version']} · {run_txt}"),
        ("Iteration", f"{int(metrics.get('iteration', 0) or 0):,}"),
        ("Speed", f"{int(metrics.get('sps', 0) or 0):,} steps/s"),
        ("Elapsed", format_elapsed_time(status_info.get("elapsed_seconds", 0))),
    ]
    stats = "".join(f"<div class='sb-item'><span class='sb-k'>{k}</span><span class='sb-v'>{html.escape(v)}</span></div>"
                    for k, v in items)
    warm = ("<span class='sb-warm' title='The critic is fitting the new reward; the policy is frozen until it finishes.'>"
            "critic warm-up</span>" if run["warmup"] and running else "")
    fb = f"<div class='sb-feedback'>{feedback_msg}</div>" if feedback_msg else ""
    return f"<div class='status-bar'><div class='sb-left'>{badge}{warm}</div><div class='sb-stats'>{stats}</div></div>{fb}"


def build_reward_card_html(metrics: Optional[dict] = None) -> str:
    """The frozen reward version the trainer runs: its terms, weights, anneal and scenario mix."""
    version = active_version()
    try:
        snap = load_snapshot(version)
    except Exception as e:
        return f"<div class='rw-card'>Reward {html.escape(version)}: snapshot unreadable ({html.escape(str(e))})</div>"
    s = snap["settings"]
    rew, ann, sc = s["rewards"], s.get("reward_annealing") or {}, s["scenarios"]
    run = _run_progress(metrics or {})
    decay = int(ann.get("decay_steps", 0) or 0)
    targets = (ann.get("targets") or {}) if ann.get("enabled") else {}
    frac = min(1.0, max(0.0, (run["run_steps"] or 0) / decay)) if decay and run["version"] == version else 0.0

    def weight_cell(key):
        base = float(rew.get(key, 0.0))
        if key in targets:
            now = base + (float(targets[key]) - base) * frac
            return (f"{now:.2f} <span class='rw-dim'>→ {float(targets[key]):g} over {decay / 1e6:,.0f}M "
                    f"({frac * 100:.0f}%)</span>")
        return f"{base:g}"

    if version in ("v3", "v4"):
        g, ab = float(rew["goal_reward"]), float(rew["aggression_bias"])
        terms = [
            ("Goal", f"+{g:g} / −{g * (1 - ab):g}", f"aggression_bias {ab:g}"),
            ("Ball position", weight_cell("ball_position_weight"), "potential"),
            ("Touch", weight_cell("touch_weight"), "× ball Δv / 2300"),
            ("Closeness to ball", weight_cell("closeness_weight"),
             "potential" if float(rew["closeness_weight"]) or "closeness_weight" in targets else "retired"),
            ("Boost held", weight_cell("boost_weight"), "potential"),
        ]
        if "race_weight" in rew:
            terms.append(("Ball race", weight_cell("race_weight"), "potential: nearer the ball than the opponent"))
    else:
        terms = [(k.replace("_weight", "").replace("_", " "), weight_cell(k), "") for k in sorted(rew)]
    rows = "".join(f"<tr><td>{html.escape(n)}</td><td class='rw-w'>{w}</td><td class='rw-dim'>{html.escape(d)}</td></tr>"
                   for n, w, d in terms)

    names = {"replay_prob": "replay", "kickoff_prob": "kickoff", "aerial_prob": "aerial", "save_prob": "goalie save",
             "wall_prob": "wall", "wall_rebound_prob": "wall rebound", "turnaround_prob": "turnaround",
             "dribble_flick_prob": "dribble", "bounce_drop_prob": "bounce/drop", "retreat_prob": "retreat",
             "custom_prob": "custom"}
    palette = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f472b6", "#a78bfa", "#fb923c", "#2dd4bf", "#f87171", "#a3e635", "#94a3b8"]
    mix = sorted(((names.get(k, k), float(v)) for k, v in sc.items() if float(v) > 0), key=lambda x: -x[1])
    total = sum(v for _, v in mix) or 1.0
    bar = "".join(f"<span style='width:{100 * v / total:.2f}%;background:{palette[i % len(palette)]}' "
                  f"title='{html.escape(n)} {100 * v / total:.0f}%'></span>" for i, (n, v) in enumerate(mix))
    legend = "".join(f"<span class='rw-leg'><i style='background:{palette[i % len(palette)]}'></i>{html.escape(n)} "
                     f"{100 * v / total:.0f}%</span>" for i, (n, v) in enumerate(mix))
    ident = snap.get("identity", {})
    return f"""
    <div class='rw-card'>
      <div class='rw-head'><span class='rw-title'>Reward {html.escape(version)}</span>
        <span class='rw-frozen' title='Weights, anneal and scenario mix are fixed for the run. A change is a new version.'>frozen</span></div>
      <table class='rw-table'>{rows}</table>
      <div class='rw-sub'>Scenario mix</div>
      <div class='rw-bar'>{bar}</div>
      <div class='rw-legend'>{legend}</div>
      <div class='rw-foot'>code {html.escape(str(ident.get('code_sha', '?')))} · settings {html.escape(str(ident.get('settings_sha', '?')))}
        · starts from {html.escape(os.path.basename(str(snap.get('start_checkpoint', '–'))))}<br>
        Spec: <code>{html.escape(str(snap.get('spec', 'docs/')))}</code>. Change it by making a new version, not by editing this one.</div>
    </div>"""


def build_full_diagnostic_export() -> tuple[str, str]:
    """
    A pasteable snapshot of the whole system: process, reward run, checkpoint, config, eval results
    and behaviour flags. Returns (short overview markdown, full export markdown).
    """
    mgr = TrainingProcessManager.get_instance()
    status = mgr.get_status_info()
    running, paused = status.get("running", False), status.get("paused", False)
    state = "RUNNING" if running and not paused else ("PAUSED" if paused else "STOPPED")
    metrics = status.get("metrics", {}) or {}
    run = _run_progress(metrics)
    cfg = effective_config()
    hp, env = cfg["hyperparameters"], cfg["environment"]
    version = active_version()
    snap = load_snapshot(version)
    opponent_mix = describe_opponent_mix(cfg.get("league", {}) or {}, metrics.get("league", {}) or {},
                                         num_envs=int(env.get("num_envs", 0) or 0))

    latest = "checkpoints/latest_model.pt"
    ckpt_line = "none"
    if os.path.exists(latest):
        try:
            c = torch.load(latest, map_location="cpu", weights_only=False)
            ckpt_line = (f"`{latest}` iteration {c.get('iteration', '?'):,}, reward {c.get('reward_identity', 'unstamped')}, "
                         f"run start step {c.get('reward_run_start_step', '?')}")
        except Exception as e:
            ckpt_line = f"`{latest}` (unreadable: {e})"

    tests = get_cached_or_run_tests(force_refresh=False)
    tests_line = f"{tests.get('passed', 0)}/{tests.get('total_tests', tests.get('total', 0))} passed"
    tel = run_telemetry(window=10)
    flags = behaviour_flags_markdown(tel)
    evals = eval_results.list_results()
    eval_lines = "\n".join(f"* {eval_results.label_for(e)} ({e['created']})" for e in evals[:8]) or "* none"
    run_txt = f"{run['run_steps'] / 1e6:,.1f}M steps" if run["run_steps"] is not None else "not started"

    export_text = f"""# SensAI system snapshot
**Taken:** {time.strftime('%Y-%m-%d %H:%M:%S')} · **Process:** {state} (PID {status.get('pid')}, up {format_elapsed_time(status.get('elapsed_seconds', 0))})

## Reward run
* **Active version:** {version} · identity {snap.get('identity')}
* **Run:** {run['version']} · {run_txt}{' · critic warm-up' if run['warmup'] else ''}
* **Settings:**
```json
{json.dumps(snap['settings'], indent=1)}
```

## Training (latest iteration)
* Iteration {metrics.get('iteration', 0):,} · global step {metrics.get('global_step', 0):,} · {metrics.get('sps', 0):,} steps/s
* Mean reward {metrics.get('mean_reward', 0.0):+.3f} · policy loss {metrics.get('policy_loss', 0.0):.4f} · value loss {metrics.get('value_loss', 0.0):.4f} · entropy {metrics.get('entropy', 0.0):.4f}
* Latest checkpoint: {ckpt_line}

## Config
* LR `{hp.get('learning_rate')}` · entropy `{hp.get('ent_coef')}` · clip `{hp.get('clip_range')}` · gamma `{hp.get('gamma')}` · GAE λ `{hp.get('gae_lambda')}`
* Batch `{hp.get('batch_size')}` · minibatch `{hp.get('mini_batch_size')}` · epochs `{hp.get('n_epochs')}` · envs `{env.get('num_envs')}` · tick skip `{env.get('tick_skip')}`
* Opponents: {opponent_mix}

## Eval results (newest first)
{eval_lines}

## Behaviour (training telemetry)
{flags}

## Tests
{tests_line}

## Recent process output
```text
{mgr.get_logs(max_lines=30)}
```
"""
    overview = (f"**{state}** · reward {run['version']} · {run_txt} · iteration {metrics.get('iteration', 0):,} · "
                f"{metrics.get('sps', 0):,} steps/s · tests {tests_line}\n\nLatest checkpoint: {ckpt_line}")
    return overview, export_text


def load_league_state_safely(path: str = "logs/league_state.json") -> Dict[str, Any]:
    """Safely loads league promotion and sports ticker state from disk without crashing on lock contention."""
    if not os.path.exists(path):
        return {}
    for _ in range(3):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (PermissionError, json.JSONDecodeError):
            time.sleep(0.05)
        except Exception:
            break
    return {}


def get_cockpit_leaderboard_df(
    evaluator: TrueSkillEvaluator,
    max_rows: int = 15,
    league_state: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Returns a formatted DataFrame of top-performing checkpoints and models
    ranked by conservative TrueSkill rating (mu - 3*sigma).
    Highlights checkpoint iterations and active League Status (King, Elite Sparrer, In Gauntlet, Anchor, Archived).
    """
    df = evaluator.get_leaderboard_dataframe()
    if df.empty:
        return pd.DataFrame(columns=[
            "Rank", "Iteration / Model", "Status", "Confidence", "Rating (μ)",
            "Uncertainty (σ)", "Conservative Score", "Points", "Win Rate",
            "Record (W-L-D)", "Goal Diff", "Matches"
        ])

    state = league_state if league_state is not None else load_league_state_safely()
    active_king = state.get("king_of_the_hill", "")
    elite_pool = [str(p).replace("\\", "/").lower() for p in state.get("elite_pool", [])]
    contenders = [str(p).replace("\\", "/").lower() for p in state.get("contender_queue", [])]

    # Dynamic fallback for elite pool if file was empty or newly generated
    if not elite_pool and evaluator and hasattr(evaluator, "ratings"):
        valid = []
        for key, rec in evaluator.ratings.items():
            if "latest_model" in key.lower():
                continue
            if rec.is_anchor or key == "heuristic" or os.path.exists(rec.path):
                valid.append((key.replace("\\", "/").lower(), evaluator.ranking_key(rec), rec.matches_played))
        valid.sort(key=lambda x: (x[1], x[2]), reverse=True)
        elite_pool = [x[0] for x in valid[:10]]

    norm_king = str(active_king).replace("\\", "/").lower()

    rename_dict = {}
    for col in df.columns:
        if "Rating" in col:
            rename_dict[col] = "Rating (μ)"
        elif "Uncertainty" in col:
            rename_dict[col] = "Uncertainty (σ)"
        elif col == "Model":
            rename_dict[col] = "Iteration / Model"
    df = df.rename(columns=rename_dict)

    def determine_status(raw_name: str) -> str:
        name_str = str(raw_name)
        matched_rec = None
        for key, r in evaluator.ratings.items():
            if r.name.lower() == name_str.lower() or key.lower() == name_str.lower():
                matched_rec = r
                break

        path_str = matched_rec.path.replace("\\", "/").lower() if matched_rec else name_str.replace("\\", "/").lower()
        low_name = name_str.lower()

        if norm_king and (path_str == norm_king or low_name in norm_king or norm_king in path_str):
            return "👑 King"

        for ep in elite_pool:
            if path_str == ep or low_name in ep or ep in path_str:
                return "🛡️ Elite Pool"

        for cq in contenders:
            if path_str == cq or low_name in cq or cq in path_str:
                return "⚔️ In Gauntlet"

        if matched_rec and matched_rec.is_anchor:
            return "⚓ Anchor"

        if matched_rec and not os.path.exists(matched_rec.path):
            return "📦 Archived"

        return "Standby"

    def format_iteration_label(name: str) -> str:
        name_str = str(name)
        if "checkpoint_iter_" in name_str.lower():
            try:
                parts = name_str.lower().split("checkpoint_iter_")
                iter_num = parts[1].split(".")[0].split()[0]
                return f"⚡ Iteration {iter_num} ({name_str})"
            except Exception:
                return f"⚡ {name_str}"
        elif "latest" in name_str.lower():
            return "🟢 Latest Policy (latest_model.pt)"
        return name_str

    if "Iteration / Model" in df.columns:
        df["Status"] = df["Iteration / Model"].apply(determine_status)
        df["Iteration / Model"] = df["Iteration / Model"].apply(format_iteration_label)

        cols = list(df.columns)
        if "Status" in cols and "Iteration / Model" in cols:
            cols.remove("Status")
            idx = cols.index("Iteration / Model")
            cols.insert(idx + 1, "Status")
            df = df[cols]

    return df.head(max_rows)


_LB_EVALUATOR_GATE = lambda rec: getattr(rec, "sigma", 8.33) <= 1.5


def _lb_confidence(rec) -> tuple:
    """(chip_html, is_ranked) describing whether a rating may be ranked on raw mu."""
    if getattr(rec, "is_anchor", False):
        return '<span class="lb-chip lb-chip-anchor">Anchor</span>', True
    if getattr(rec, "rating_locked", False):
        # Distinct from Ranked: the rating is not merely trusted, it is final. Worth its
        # own chip so a number that will never move again does not read as a live one.
        return '<span class="lb-chip lb-chip-locked">&#128274; Locked</span>', True
    ranked = _LB_EVALUATOR_GATE(rec)
    if ranked:
        return '<span class="lb-chip lb-chip-ranked">Ranked</span>', True
    return '<span class="lb-chip lb-chip-provisional">Provisional</span>', False


def _lb_iteration(rec) -> int:
    """Iteration number behind a checkpoint rating, or -1 for anchors and baselines."""
    match = re.search(r"checkpoint_iter_(\d+)", f"{getattr(rec, 'path', '')} {getattr(rec, 'name', '')}".lower())
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return -1


def _lb_short_name(rec) -> str:
    """'Iteration 130740' for checkpoints, the display name for everything else."""
    it = _lb_iteration(rec)
    return f"Iteration {it}" if it >= 0 else rec.name


# Seconds for one best-of-9 series between two current checkpoints, measured on this
# machine. It moves with how the policy plays: a pair that scores quickly finishes a
# series in well under a second, while two evenly matched checkpoints that rally take
# several. Used only to estimate the duty cycle for display, never to decide anything.
SECONDS_PER_EVAL_SERIES = 3.3


def gauntlet_budget_estimate(
    series_per_step: int,
    league: Dict[str, Any],
    logging_cfg: Dict[str, Any],
    hyperparams: Dict[str, Any],
    sps: float,
) -> Dict[str, float]:
    """
    What a given gauntlet trial budget costs and buys.

    A grading event fires once per checkpoint save and admits at most one newcomer, so
    the arithmetic is short. It delivers `series_per_step` series to a single contender,
    ranking one costs target_eval_matches, and a contender's turn comes round once per
    max_active_contenders events. The share of saves that ever get ranked is therefore
    series_per_step / target_eval_matches, independent of both the checkpoint interval
    and the queue depth, because those two scale the arrival rate and the grading rate
    together.
    """
    target = max(1, int(league.get("target_eval_matches", 30)))
    queue = max(1, int(league.get("max_active_contenders", 3)))
    debut = max(0, int(league.get("eval_series_per_grade", 1))) * 2
    interval = max(1, int(logging_cfg.get("checkpoint_interval", 200)))
    batch = max(1, int(hyperparams.get("batch_size", 16384)))
    sps = sps if sps and sps > 0 else 9500.0

    window_s = interval * batch / sps
    step = max(1, int(series_per_step))
    return {
        "window_s": window_s,
        "duty_pct": 100.0 * (step + debut) * SECONDS_PER_EVAL_SERIES / window_s,
        "minutes_to_rank": (target / step) * queue * window_s / 60.0,
        "ranked_pct": 100.0 * min(1.0, step / target),
        "slot_minutes": (target / step) * window_s / 60.0,
    }


def opponent_mix_shares(
    league: Dict[str, Any], num_envs: Optional[int] = None
) -> Dict[str, float]:
    """
    What percentage of training environments each tier actually gets.

    Two corrections to the raw ratios. The fixed training-opponent list is taken off the
    top, so the standard ratios divide only what it leaves behind. And when num_envs is
    known, the tiers are snapped to whole env-worker slices exactly as the scheduler
    snaps them, because that is the split the environments really run: asking for 20%
    of 128 environments yields 18.75%, and a panel that prints 20% is repeating in
    miniature the mistake this display was fixed for.
    """
    opp_ratio = float(league.get("training_opponent_ratio", 0.0) or 0.0)
    if not (league.get("training_opponents") or []):
        opp_ratio = 0.0
    opp_ratio = max(0.0, min(1.0, opp_ratio))

    sp = float(league.get("self_play_ratio", 0.50))
    king = float(league.get("king_ratio", 0.25))
    pool = float(league.get("pool_ratio", 0.25))
    total = sp + king + pool
    if total <= 1e-6:
        sp, king, pool, total = 0.50, 0.25, 0.25, 1.0

    remaining = 1.0 - opp_ratio
    shares = {
        "fixed": opp_ratio * 100.0,
        "self_play": remaining * (sp / total) * 100.0,
        "king": remaining * (king / total) * 100.0,
        "pool": remaining * (pool / total) * 100.0,
    }
    if not num_envs or num_envs < 1:
        return shares

    n_training = min(num_envs, int(round(num_envs * opp_ratio)))
    rest = max(0, num_envs - n_training)
    n_sp = max(1, int(round(rest * sp / total))) if rest else 0
    n_king = max(1, int(round(rest * king / total))) if rest else 0
    n_pool = max(0, rest - n_sp - n_king)
    n_training, n_sp, n_king, n_pool = snap_tiers_to_worker_slices(
        num_envs, n_training, n_sp, n_king, n_pool,
        int(league.get("pool_group_size", 4) or 4),
    )
    return {
        "fixed": 100.0 * n_training / num_envs,
        "self_play": 100.0 * n_sp / num_envs,
        "king": 100.0 * n_king / num_envs,
        "pool": 100.0 * n_pool / num_envs,
    }


def _pct(value: float) -> str:
    """Trims a trailing .0 so whole percentages do not read as false precision."""
    rounded = round(value, 1)
    return f"{int(rounded)}%" if float(rounded).is_integer() else f"{rounded}%"


def describe_opponent_mix(
    league: Dict[str, Any],
    telemetry: Optional[Dict[str, Any]] = None,
    num_envs: Optional[int] = None,
) -> str:
    """
    One line naming what the training environments are really facing, with each tier's share
    as the scheduler actually snaps it.
    """
    if not league.get("enabled", True):
        # Without the league there is no King or pool: the fixed list keeps its share, the rest self-play
        fixed = float(opponent_mix_shares(league, num_envs)["fixed"])
        listed = ", ".join(os.path.basename(str(x)) for x in league.get("training_opponents") or [])
        if fixed:
            return f"League disabled | Fixed `{_pct(fixed)}` ({listed}) | Self-play `{_pct(100.0 - fixed)}`"
        return "League disabled | Self-play `100%`"

    telemetry = telemetry or {}
    shares = opponent_mix_shares(league, num_envs)
    king = telemetry.get("king_of_the_hill") or "None"

    parts = [f"Self-play `{_pct(shares['self_play'])}`"]
    if king and king != "None":
        parts.append(f"King `{_pct(shares['king'])}` ({king})")
    else:
        # No King means the King and pool tiers fold back into self-play; saying "25%"
        # here would name a tier that is not running.
        parts = [f"Self-play `{_pct(shares['self_play'] + shares['king'] + shares['pool'])}` (no King seated)"]
        if shares["fixed"]:
            parts.append(
                f"Fixed `{_pct(shares['fixed'])}` "
                f"({', '.join(os.path.basename(str(x)) for x in league.get('training_opponents') or [])})"
            )
        return " | ".join(parts)

    pool_n = telemetry.get("elite_pool_size")
    pool_note = f" across {pool_n} checkpoints" if pool_n else ""
    parts.append(f"Elite pool `{_pct(shares['pool'])}`{pool_note}")
    if shares["fixed"]:
        listed = ", ".join(os.path.basename(str(x)) for x in league.get("training_opponents") or [])
        parts.append(f"Fixed `{_pct(shares['fixed'])}` ({listed})")
    return " | ".join(parts)


def build_how_it_works_html(league_state: Optional[Dict[str, Any]] = None) -> str:
    """
    A hover explainer for the path a checkpoint takes from training to King.

    Numbers are read from the live league config rather than written into the copy, so
    the explainer cannot drift away from the thresholds actually in force. CSS-only:
    reveals on :hover and :focus-within, so it costs nothing per frame and is reachable
    from the keyboard.
    """
    cfg = effective_config()
    league = cfg.get("league", {}) or {}
    logging_cfg = cfg.get("logging", {}) or {}

    interval = logging_cfg.get("checkpoint_interval", 200)
    best_of = league.get("series_length", 9)
    to_win = league.get("series_wins_needed", 5)
    debut = league.get("eval_series_per_grade", 1)
    per_trial = league.get("contender_series_per_step", 3)
    target = league.get("target_eval_matches", 30)
    gate = league.get("eligibility_sigma", 1.5)
    pool_size = league.get("max_pool_size", 10)
    streak = league.get("max_consecutive_losses", 4)

    # The King's share is king_ratio of what the fixed training_opponents list leaves
    # behind, not king_ratio of everything. See opponent_mix_shares.
    king_pct = _pct(opponent_mix_shares(league)["king"]).rstrip("%")

    steps = [
        ("1", "Saved",
         f"A checkpoint is written every {interval} iterations and enters as un-rated."),
        ("2", "Debut",
         f"It plays {debut} series against the nearest-strength anchor and {debut} against "
         f"the King. A series is best-of-{best_of}, first to {to_win}, each episode ending "
         "at the first goal."),
        ("3", "Gauntlet",
         f"If the debut holds up it joins the contender queue and plays up to {per_trial} "
         "series per trial, split across its opponents. Trials go to whichever contender "
         "is least measured, not whichever currently looks best."),
        ("4", "Ranked",
         f"After {target} series with uncertainty down to σ ≤ {gate}, its rating is "
         "trusted and it graduates to the Elite Pool."),
        ("5", "King",
         f"The Elite Pool is the top {pool_size} by rating μ among trusted models. The "
         f"highest is crowned and becomes {king_pct}% of training opponents. Anchors are "
         "never crowned."),
        ("6", "Scale",
         "Every rating is re-fitted from every series ever played, so a result counts the "
         "same whenever it happened. One rating is pinned: the v3 king at μ 25, which a "
         f"{_pct(100 * league.get('calibration_share', 0.1))} share of series is played against. "
         "Necto and Nexto are fitted like anyone else."),
    ]
    rows = "".join(
        f'<div class="hiw-step"><span class="hiw-num">{n}</span>'
        f'<div><b>{title}</b><span>{body}</span></div></div>'
        for n, title, body in steps
    )

    return f"""
    <span class="hiw" tabindex="0" role="button" aria-label="How a checkpoint becomes King">
        <span class="hiw-chip">&#9432; How it works</span>
        <span class="hiw-pop" role="tooltip">
            <span class="hiw-title">From training run to King</span>
            {rows}
            <span class="hiw-foot">
                Falls out at any stage on {streak} straight series losses, or once its
                rating is confidently below the King's. Nothing is dropped while its
                rating is still too uncertain to judge.
            </span>
        </span>
    </span>
    """


def build_anchor_legend_html(evaluator: TrueSkillEvaluator) -> str:
    """The scale strip above the board: the one pinned rating and the fitted references."""
    return league_board.scale_legend_html(evaluator, build_how_it_works_html())


def _board_version() -> str:
    try:
        return active_version()
    except Exception:
        return ""


def build_cockpit_leaderboard_summary_html(evaluator: TrueSkillEvaluator, league_state: Optional[Dict[str, Any]] = None) -> str:
    """King banner and the season strip; see ui/league_board.py."""
    global _LB_EVALUATOR_GATE
    _LB_EVALUATOR_GATE = evaluator.is_rank_eligible
    state = league_state if league_state is not None else load_league_state_safely()
    return league_board.king_banner_html(evaluator, state, _board_version())


# Marker shape and colour per reference. Shape carries the distinction as well as
# colour, so the series stay separable for a colour-blind reader and in a screenshot.
_BM_SERIES_STYLE = [
    ("#a78bfa", "circle"),    # violet
    ("#34d399", "square"),    # emerald
    ("#fb7185", "triangle"),  # rose
    ("#38bdf8", "diamond"),   # sky
]


def _bm_marker(shape: str, x: float, y: float, color: str, tip: str) -> str:
    """One data point. The radius is deliberately small; these plots get dense over a long run."""
    r = 3.4
    if shape == "square":
        return (f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r:.1f}" height="{2 * r:.1f}" '
                f'rx="0.8" fill="{color}" class="bm-mark"><title>{tip}</title></rect>')
    if shape == "triangle":
        pts = f"{x:.1f},{y - r - 0.6:.1f} {x + r + 0.4:.1f},{y + r:.1f} {x - r - 0.4:.1f},{y + r:.1f}"
        return f'<polygon points="{pts}" fill="{color}" class="bm-mark"><title>{tip}</title></polygon>'
    if shape == "diamond":
        pts = (f"{x:.1f},{y - r - 0.7:.1f} {x + r + 0.7:.1f},{y:.1f} "
               f"{x:.1f},{y + r + 0.7:.1f} {x - r - 0.7:.1f},{y:.1f}")
        return f'<polygon points="{pts}" fill="{color}" class="bm-mark"><title>{tip}</title></polygon>'
    return (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{color}" '
            f'class="bm-mark"><title>{tip}</title></circle>')


def _build_benchmark_chart(state: Dict[str, Any]) -> str:
    """
    Goal margin per episode against each fixed reference, in the order the readings were played.

    Why margin rather than series wins: series win rate against these references is
    pinned at the rails and has no gradient left. Checkpoints take 100% of series off
    the heuristic and the BC baseline, and won 9 of 96 off Necto across iterations
    160000-190200 both before and after a real improvement. The margin over that same
    span moved -0.64 to -0.25, a 4.7 sigma change the series record shows as flat.

    Why one line per reference and never a combined score: the two measure different
    things -- Necto and Nexto punish different mistakes -- so a divergence between the
    lines is a style shift, which an average would hide. (The closed cycle that first
    motivated this, Nexto over Necto over the bot over Nexto, was an artefact of Necto's
    broken kickoff and is gone: Nexto now beats Necto 38-4.)
    """
    history = [
        e for e in (state.get("benchmark_history") or [])
        if isinstance(e, dict) and e.get("iteration") is not None and isinstance(e.get("results"), dict)
    ]

    if len(history) < 2:
        waiting = ("One reading so far, and a curve needs two."
                   if len(history) == 1 else "No benchmark has run yet.")
        return f"""
        <div class="lb-empty">
            &#128202; <b>{waiting}</b><br>
            Necto and Nexto play the King on the <code>benchmark_interval</code> schedule. Goal
            margin keeps a gradient where series results against them are pinned at zero wins.
        </div>
        """

    # Plotted in the order the readings were played, not by iteration: checkpoint numbering
    # restarts from each run's start checkpoint, so five runs' readings share one iteration
    # range and an iteration axis stacks them on top of each other. Evenly spaced by reading
    # rather than by wall time, so a night of idle does not open a gap in the middle.
    history.sort(key=lambda e: (str(e.get("at", "")), e["iteration"]))
    order = {id(e): i for i, e in enumerate(history)}

    # Series order is fixed by first appearance, so a colour never migrates between
    # references as readings accumulate.
    names: List[str] = []
    for e in history:
        for name in e["results"]:
            if name not in names:
                names.append(name)

    points: Dict[str, List[Tuple[int, float, Dict[str, Any]]]] = {n: [] for n in names}
    when: Dict[int, Tuple[str, int]] = {}
    for e in history:
        when[order[id(e)]] = (str(e.get("at", "")), int(e["iteration"]))
        for name, res in e["results"].items():
            eps = int(res.get("episodes", 0) or 0)
            if eps <= 0:
                continue  # written before the episode count was recorded
            gf = int(res.get("goals_for", 0) or 0)
            ga = int(res.get("goals_against", 0) or 0)
            points[name].append((order[id(e)], (gf - ga) / eps, res))
    points = {n: p for n, p in points.items() if p}

    if not points:
        return """
        <div class="lb-empty">
            &#128202; <b>Readings exist but carry no episode count.</b><br>
            They predate the margin readout. The next benchmark records it.
        </div>
        """

    W, H = 720.0, 232.0
    PAD_L, PAD_R, PAD_T, PAD_B = 44.0, 12.0, 12.0, 30.0
    iters = [it for p in points.values() for it, _, _ in p]
    x_lo, x_hi = min(iters), max(iters)
    x_span = max(x_hi - x_lo, 1)

    def sx(it: int) -> float:
        return PAD_L + (W - PAD_L - PAD_R) * (it - x_lo) / x_span

    def sy(margin: float) -> float:
        # Margin per episode is bounded by +/-1, so the axis is fixed rather than fitted.
        # A fixed axis keeps the zero line in one place between refreshes, which is what
        # makes the chart readable at a glance instead of rescaling under the reader.
        clamped = max(-1.0, min(1.0, margin))
        return PAD_T + (H - PAD_T - PAD_B) * (1.0 - (clamped + 1.0) / 2.0)

    parts: List[str] = []
    for val, label in ((1.0, "+1"), (0.5, "+0.5"), (0.0, "0"), (-0.5, "-0.5"), (-1.0, "-1")):
        y = sy(val)
        cls = "bm-zero" if val == 0.0 else "bm-grid"
        parts.append(f'<line x1="{PAD_L:.0f}" y1="{y:.1f}" x2="{W - PAD_R:.0f}" y2="{y:.1f}" class="{cls}" />')
        parts.append(f'<text x="{PAD_L - 6:.0f}" y="{y + 3:.1f}" text-anchor="end" class="bm-axis-label">{label}</text>')

    def _day(i: int) -> str:
        try:
            return datetime.datetime.fromisoformat(when[i][0]).strftime("%b %d")
        except (KeyError, ValueError):
            return f"#{i + 1}"

    tick_iters = (x_lo, (x_lo + x_hi) // 2, x_hi) if x_hi > x_lo else (x_lo,)
    for it in tick_iters:
        parts.append(f'<text x="{sx(it):.1f}" y="{H - PAD_B + 14:.0f}" text-anchor="middle" '
                     f'class="bm-axis-label">{_day(it)}</text>')
    mid_x = (PAD_L + W - PAD_R) / 2.0
    mid_y = (PAD_T + H - PAD_B) / 2.0
    parts.append(f'<text x="{mid_x:.0f}" y="{H - 2:.0f}" text-anchor="middle" '
                 f'class="bm-axis-title">READINGS, OLDEST TO NEWEST</text>')
    parts.append(f'<text x="11" y="{mid_y:.0f}" class="bm-axis-title" text-anchor="middle" '
                 f'transform="rotate(-90 11 {mid_y:.0f})">GOAL MARGIN / EPISODE</text>')

    legend: List[str] = []
    for idx, name in enumerate(n for n in names if n in points):
        color, shape = _BM_SERIES_STYLE[idx % len(_BM_SERIES_STYLE)]
        series = points[name]
        # A least-squares fit rather than a line through every point. Joining the dots on
        # a noisy series draws a zigzag that reads as structure when it is sampling error:
        # adjacent readings against Nexto differ by a full point, which is the matchup
        # moving rather than the policy. The fit shows the direction and leaves the spread
        # visible in the markers themselves.
        if len(series) > 2:
            n = float(len(series))
            mean_x = sum(it for it, _, _ in series) / n
            mean_y = sum(m for _, m, _ in series) / n
            denom = sum((it - mean_x) ** 2 for it, _, _ in series)
            if denom > 0:
                slope = sum((it - mean_x) * (m - mean_y) for it, m, _ in series) / denom
                x0, x1 = series[0][0], series[-1][0]
                parts.append(
                    f'<line x1="{sx(x0):.1f}" y1="{sy(mean_y + slope * (x0 - mean_x)):.1f}" '
                    f'x2="{sx(x1):.1f}" y2="{sy(mean_y + slope * (x1 - mean_x)):.1f}" '
                    f'class="bm-trend" stroke="{color}" />'
                )
        for it, m, res in series:
            tip = (f"{name} &#183; iter {when.get(it, ('', 0))[1]:,} &#183; {when.get(it, ('', 0))[0][:16].replace('T', ' ')} &#183; margin {m:+.2f}/ep &#183; goals "
                   f"{res.get('goals_for', 0)}-{res.get('goals_against', 0)} over "
                   f"{res.get('episodes', 0)} episodes &#183; "
                   f"{res.get('series_won', 0)}/{res.get('series', 0)} series")
            parts.append(_bm_marker(shape, sx(it), sy(m), color, tip))
        legend.append(f'<span class="bm-legend-item">'
                      f'<span class="bm-legend-swatch" style="background: {color};"></span>'
                      f'vs {name} <span class="lb-muted">{series[-1][1]:+.2f}</span></span>')

    return f"""
    <div class="bm-chart">
        <svg class="bm-svg" viewBox="0 0 {W:.0f} {H:.0f}" role="img"
             aria-label="Goal margin per episode against each fixed reference, oldest reading to newest">
            {''.join(parts)}
        </svg>
    </div>
    <div class="bm-legend"><span class="lb-muted">Latest</span>{''.join(legend)}
        <span class="lb-muted">last {len(history)} readings, across runs</span></div>
    <div class="bm-note">
        Above the zero line the checkpoint outscores the reference. Each reference is its own
        line and they are <b>never combined</b>: read one for progress and a <b>divergence between
        them as a style shift</b>. These series also feed the rating fit, where they place Necto
        and Nexto on the scale; neither is pinned, so they cannot move it.
    </div>
    """


def build_league_wire_and_queue_html(evaluator: TrueSkillEvaluator, league_state: Optional[Dict[str, Any]] = None) -> str:
    """
    The league board below the King banner: rating ladder, elite pool, pipeline, head to head,
    recent series and benchmarks. The name is kept from the ticker era because the timer and the
    refresh button are wired to it.
    """
    state = league_state if league_state is not None else load_league_state_safely()
    return league_board.board_html(evaluator, state, _board_version(), _build_benchmark_chart(state))


_CKPT_STAMP_CACHE: Dict[str, tuple] = {}


def _checkpoint_stamp(path: str) -> dict:
    """iteration / reward version / steps into its run, cached by mtime (loading a checkpoint is ~50 ms)."""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return {}
    hit = _CKPT_STAMP_CACHE.get(path)
    if hit and hit[0] == mt:
        return hit[1]
    info = {}
    try:
        c = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(c, dict):
            stamp = c.get("reward_identity")
            start = c.get("reward_run_start_step")
            info = {"iteration": c.get("iteration"),
                    "version": stamp.get("version") if isinstance(stamp, dict) else ("v2" if "model_state_dict" in c else None),
                    "run_steps": (c.get("global_step", 0) - start) if start is not None else None}
    except Exception:
        pass
    _CKPT_STAMP_CACHE[path] = (mt, info)
    return info


def eval_checkpoint_choices() -> list:
    """(label, path) for the eval suite: the live model, pinned baselines, then numbered checkpoints newest first."""
    paths = []
    if os.path.exists("checkpoints/latest_model.pt"):
        paths.append("checkpoints/latest_model.pt")
    paths += sorted(glob.glob("checkpoints/baselines/*.pt"))
    numbered = glob.glob("checkpoints/checkpoint_iter_*.pt")
    numbered.sort(key=lambda p: int(re.sub(r"\D", "", os.path.basename(p)) or 0), reverse=True)
    paths += numbered
    out = []
    for p in dict.fromkeys(os.path.normpath(x).replace("\\", "/") for x in paths):
        s = _checkpoint_stamp(p)
        bits = [os.path.basename(p)]
        if s.get("version"):
            bits.append(s["version"] + (f" +{s['run_steps'] / 1e6:,.0f}M" if s.get("run_steps") is not None else ""))
        if s.get("iteration") is not None and "iter" not in bits[0]:
            bits.append(f"iter {s['iteration']:,}")
        out.append((" · ".join(bits), p))
    return out


def _eval_result_choices() -> list:
    return [(eval_results.label_for(e), e["path"]) for e in eval_results.list_results()]


def _default_reference_path() -> Optional[str]:
    """The checkpoint a head-to-head defaults to: the one the active reward version started from."""
    try:
        p = load_snapshot(active_version()).get("start_checkpoint")
    except Exception:
        p = None
    return p if p and os.path.exists(p) else None


def _default_baseline_path() -> Optional[str]:
    try:
        p = load_snapshot(active_version()).get("baseline_eval")
    except Exception:
        p = None
    return p if p and os.path.exists(p) else None


def create_ui():
    mgr = TrainingProcessManager.get_instance()
    bc_trainer = BehavioralCloningTrainer()
    default_cfg = effective_config()
    hp_cfg = default_cfg["hyperparameters"]
    env_cfg = default_cfg["environment"]
    log_cfg = default_cfg.get("logging", {}) or {}
    league_cfg_ui = default_cfg.get("league", {}) or {}
    init_status = mgr.get_status_info()
    ts_evaluator = TrueSkillEvaluator()

    ui_num_envs = max(1, int(env_cfg.get("num_envs", 64) or 64))
    ui_workers = max(1, int(env_cfg.get("num_env_workers", 1) or 1))
    env_block = max(1, ui_num_envs // ui_workers)

    with gr.Blocks(title="SensAI Studio") as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")

        # -------------------------------------------------------------------------------------
        # Header: what is running, and the four controls that act on it
        # -------------------------------------------------------------------------------------
        with gr.Row(elem_classes=["app-header"], equal_height=True):
            with gr.Column(scale=7, min_width=520):
                gr.HTML("<div class='app-title'>SensAI <span>Studio</span></div>")
                status_card = gr.HTML(build_status_card_html(init_status))
            with gr.Column(scale=5, min_width=420, elem_classes=["header-controls"]):
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary", interactive=not init_status["running"], min_width=80)
                    pause_btn = gr.Button("Pause", interactive=init_status["running"], min_width=80)
                    stop_btn = gr.Button("Stop", variant="stop", interactive=init_status["running"], min_width=80)
                    ckpt_btn = gr.Button("Save checkpoint", min_width=120)
                resume_chk = gr.Checkbox(
                    label="Continue the current reward run",
                    value=True,
                    info="Newest checkpoint of the active reward version, or its pinned start checkpoint. "
                         "Unchecked: random weights.",
                )

        with gr.Tabs():
            # =====================================================================================
            # TRAINING
            # =====================================================================================
            with gr.Tab("Training"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=8, min_width=560):
                        with gr.Row(elem_classes=["toolbar"]):
                            metrics_window_radio = gr.Radio(
                                [("This reward run", "run"), ("Last 100 iterations", "recent"), ("All history", "full")],
                                value="run", show_label=False, container=False, scale=4)
                            refresh_metrics_btn = gr.Button("Refresh", size="sm", scale=1, min_width=90)
                        live_metrics_plot = gr.Plot(value=render_training_curves_plot(mode="run"), show_label=False)
                        with gr.Accordion("Process output", open=False):
                            with gr.Row(elem_classes=["toolbar"]):
                                refresh_logs_btn = gr.Button("Refresh", size="sm", min_width=90)
                                clear_logs_btn = gr.Button("Clear view", size="sm", min_width=90)
                            console_output = gr.TextArea(value=mgr.get_logs(), show_label=False, lines=16,
                                                         max_lines=24, interactive=False, autoscroll=True)

                    with gr.Column(scale=4, min_width=360):
                        reward_card = gr.HTML(build_reward_card_html(init_status.get("metrics")))

                        with gr.Accordion("Optimiser (applies live)", open=True):
                            with gr.Row():
                                lr_input = gr.Number(value=hp_cfg.get("learning_rate", 3e-4), label="Learning rate", min_width=100)
                                ent_input = gr.Number(value=hp_cfg.get("ent_coef", 0.005), label="Entropy coef", min_width=100)
                                clip_input = gr.Number(value=hp_cfg.get("clip_range", 0.2), label="Clip range", min_width=100)
                            live_hp_btn = gr.Button("Apply", size="sm")
                            live_hp_msg = gr.Markdown("")

                        with gr.Accordion("Training opponents", open=False):
                            gr.Markdown(
                                "Models listed here get a fixed block of environments, split evenly between them. "
                                "The rest follow the league: 50% self-play, 25% King, 25% pool.",
                                elem_classes=["hint"])
                            with gr.Row():
                                training_opponents_select = gr.Dropdown(
                                    choices=get_available_opponent_options(),
                                    value=[str(x) for x in league_cfg_ui.get("training_opponents", []) if x],
                                    multiselect=True, label="Fixed opponents", scale=4)
                                refresh_opponent_btn = gr.Button("Scan", size="sm", scale=1, min_width=70)

                            def _opp_env_readout(count: float) -> str:
                                count = int(count or 0)
                                if count <= 0:
                                    return "**0 environments**: the league picks every opponent."
                                return (f"**{count} of {ui_num_envs} environments** ({100.0 * count / ui_num_envs:.3g}%, "
                                        f"{count // env_block} of {ui_workers} workers)")

                            _opp_start = int(round(float(league_cfg_ui.get("training_opponent_ratio", 0.0) or 0.0) * ui_num_envs))
                            _opp_start = min(ui_num_envs, (_opp_start // env_block) * env_block)
                            baseline_opp_slider = gr.Slider(
                                0, ui_num_envs, value=_opp_start, step=env_block, label="Environments for these opponents",
                                info=f"Steps of {env_block}: one worker's block, so no worker runs two opponent models.")
                            opp_env_readout = gr.Markdown(_opp_env_readout(_opp_start))
                            baseline_opp_slider.change(fn=_opp_env_readout, inputs=[baseline_opp_slider], outputs=[opp_env_readout])
                            apply_opp_btn = gr.Button("Apply", size="sm")
                            opp_apply_msg = gr.Markdown("")

                        with gr.Accordion("League grading budget", open=False):
                            def _budget_readout(step: float) -> str:
                                st = mgr.get_status_info()
                                est = gauntlet_budget_estimate(
                                    int(step or 1), default_cfg.get("league", {}) or {}, log_cfg, hp_cfg,
                                    float((st.get("metrics") or {}).get("sps") or 0.0))
                                return (f"About {est['duty_pct']:.0f}% of the evaluation window; a contender is ranked in "
                                        f"~{est['minutes_to_rank']:.0f} min; ~{est['ranked_pct']:.0f}% of saves get ranked.")

                            _budget_start = int(league_cfg_ui.get("contender_series_per_step", 16))
                            _budget_opts = sorted({8, 16, 24, 32, _budget_start})
                            gauntlet_budget_radio = gr.Radio(_budget_opts, value=_budget_start,
                                                             label="Series per gauntlet trial",
                                                             info="Grading runs beside training; more series rank sooner and take more CPU.")
                            gauntlet_budget_readout = gr.Markdown(_budget_readout(_budget_start))
                            gauntlet_budget_radio.change(fn=_budget_readout, inputs=[gauntlet_budget_radio], outputs=[gauntlet_budget_readout])
                            apply_budget_btn = gr.Button("Apply", size="sm")
                            budget_apply_msg = gr.Markdown("")

            # =====================================================================================
            # EVALUATION
            # =====================================================================================
            with gr.Tab("Evaluation"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=3, min_width=320, elem_classes=["side-panel"]):
                        gr.Markdown("### Run the eval suite")
                        gr.Markdown(
                            "Matches against the real Necto and Nexto plus four seeded scenarios. Every number is "
                            "measured on the pitch, so any two checkpoints compare fairly whatever reward trained "
                            "them. Run one every ~50M steps of a run.", elem_classes=["hint"])
                        _eval_choices = eval_checkpoint_choices()
                        eval_ckpt_dd = gr.Dropdown(choices=_eval_choices,
                                                   value=_eval_choices[0][1] if _eval_choices else None,
                                                   label="Checkpoint")
                        eval_size_radio = gr.Radio([("Full (3 seeds)", "full"), ("Quick smoke test", "quick")],
                                                   value="full", label="Size")
                        _ref_default = _default_reference_path()
                        eval_ref_dd = gr.Dropdown(choices=[("(none)", "")] + _eval_choices, value=_ref_default or "",
                                                  label="Head to head against",
                                                  info="Also plays the two checkpoints directly. More sensitive than "
                                                       "each one's score against Necto; defaults to the checkpoint "
                                                       "this reward version started from.")
                        with gr.Row():
                            eval_workers = gr.Number(value=4, precision=0, label="Worker processes", minimum=1, maximum=16, min_width=100)
                            eval_name = gr.Textbox(label="Name (optional)", min_width=120,
                                                   placeholder="auto-named from steps into the run",
                                                   info="A repeat of the same name is saved alongside as _2, never over the first.")
                        with gr.Row():
                            run_eval_btn = gr.Button("Run eval", variant="primary")
                            refresh_eval_ckpts_btn = gr.Button("Rescan", size="sm", min_width=80)
                        eval_log = gr.Textbox(show_label=False, lines=9, max_lines=9, interactive=False,
                                              placeholder="Progress appears here. Running beside training slows both.")

                    with gr.Column(scale=9, min_width=640):
                        _results = _eval_result_choices()
                        _baseline = _default_baseline_path()
                        with gr.Row(elem_classes=["toolbar"]):
                            eval_result_dd = gr.Dropdown(choices=_results, value=_results[0][1] if _results else None,
                                                         label="Result", scale=4)
                            eval_base_dd = gr.Dropdown(choices=[("(none)", "")] + _results, value=_baseline or "",
                                                       label="Compared with", scale=4,
                                                       info="Defaults to the reward version's baseline eval.")
                            refresh_evals_btn = gr.Button("Refresh", size="sm", scale=1, min_width=80)
                        eval_scorecard = gr.HTML()
                        with gr.Tabs():
                            with gr.Tab("Over the run"):
                                eval_trend_plot = gr.Plot(show_label=False)
                            with gr.Tab("Every metric"):
                                eval_only_diff = gr.Checkbox(value=False, label="Only differences beyond the noise")
                                eval_table = gr.HTML()

            # =====================================================================================
            # LEAGUE
            # =====================================================================================
            with gr.Tab("League"):
                with gr.Row(elem_classes=["toolbar"]):
                    cockpit_anchor_legend = gr.HTML(build_anchor_legend_html(ts_evaluator))
                    refresh_cockpit_lb_btn = gr.Button("Refresh", size="sm", scale=0, min_width=90)
                cockpit_lb_summary = gr.HTML(build_cockpit_leaderboard_summary_html(ts_evaluator))
                cockpit_league_ticker = gr.HTML(build_league_wire_and_queue_html(ts_evaluator))
                with gr.Accordion("Full standings", open=False):
                    cockpit_lb_table = gr.Dataframe(value=get_cockpit_leaderboard_df(ts_evaluator), interactive=False,
                                                    label="Ranked by μ among rank-eligible models (σ ≤ 1.5); provisional below")
                with gr.Accordion("Manual TrueSkill tournament", open=False):
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5):
                            all_ckpts = get_available_checkpoints()
                            ts_ckpt_multiselect = gr.Dropdown(choices=all_ckpts, value=[], multiselect=True,
                                                              label="Checkpoints")
                            with gr.Row():
                                ts_select_all_btn = gr.Button("Select all", size="sm")
                                ts_clear_all_btn = gr.Button("Clear", size="sm")
                                ts_refresh_ckpts_btn = gr.Button("Rescan", size="sm")
                            anchor_choices = ["Baseline Chaser (Heuristic)"]
                            for label, path in (("Pretrained Baseline (BC)", "checkpoints/pretrained_baseline.pt"),
                                                ("Necto (EARL TorchScript)", "checkpoints/necto-model.pt"),
                                                ("Nexto (EARL TorchScript)", "checkpoints/nexto-model.pt")):
                                if os.path.exists(path):
                                    anchor_choices.append(label)
                            ts_anchors_checkbox = gr.CheckboxGroup(choices=anchor_choices, value=[], label="Anchors")
                            with gr.Row():
                                ts_matches = gr.Radio([2, 4, 6], value=2, label="Matches per pair (home/away)")
                                ts_steps = gr.Number(value=350, precision=0, label="Steps per match", minimum=100, maximum=3000)
                            ts_overtime_check = gr.Checkbox(value=True, label="Golden-goal overtime on a tie")
                            with gr.Row():
                                run_tournament_btn = gr.Button("Run tournament", variant="primary")
                                reset_leaderboard_btn = gr.Button("Reset manual leaderboard", size="sm")
                            ts_status_md = gr.Markdown("")
                        with gr.Column(scale=7):
                            ts_leaderboard_table = gr.Dataframe(value=ts_evaluator.get_leaderboard_dataframe(), interactive=False,
                                                                label="Standings")
                            ts_leaderboard_plot = gr.Plot(value=ts_evaluator.render_leaderboard_plot(), show_label=False)

            # =====================================================================================
            # DIAGNOSTICS
            # =====================================================================================
            with gr.Tab("Diagnostics"):
                with gr.Tabs():
                    with gr.Tab("Behaviour"):
                        with gr.Row(elem_classes=["toolbar"]):
                            diag_window = gr.Radio([("Last 10 iterations", 10), ("Last 50", 50), ("Last 200", 200)],
                                                   value=10, show_label=False, container=False, scale=4)
                            refresh_diag_btn = gr.Button("Refresh", size="sm", scale=1, min_width=90)
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=4, min_width=320):
                                diag_flags = gr.Markdown(elem_classes=["flags"])
                            with gr.Column(scale=8, min_width=520):
                                diag_behaviour_plot = gr.Plot(show_label=False)
                        gr.Markdown("Telemetry comes from the training rollouts, every opponent type mixed; it "
                                    "shows how habits move during the run. How well the bot plays is measured on "
                                    "the Evaluation tab.", elem_classes=["hint"])

                    with gr.Tab("Watch a match"):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=3, min_width=300, elem_classes=["side-panel"]):
                                _sim_ckpts = get_available_checkpoints()
                                ckpt_dropdown = gr.Dropdown(choices=_sim_ckpts, value=_sim_ckpts[0], label="Blue checkpoint")
                                opponent_mode = gr.Radio(["Itself", "Heuristic chaser", "Another checkpoint"],
                                                         value="Itself", label="Orange")
                                orange_ckpt_dropdown = gr.Dropdown(choices=_sim_ckpts, value=_sim_ckpts[0],
                                                                   label="Orange checkpoint", visible=False)
                                sim_steps = gr.Number(value=400, precision=0, label="Steps (15 per second)", minimum=50, maximum=3000)
                                with gr.Row():
                                    run_sim_btn = gr.Button("Simulate", variant="primary")
                                    refresh_ckpts_btn = gr.Button("Rescan", size="sm", min_width=80)
                                sim_stats_box = gr.Markdown("")
                            with gr.Column(scale=9, min_width=600):
                                visualizer_plot = gr.Plot(show_label=False)
                                reward_breakdown_plot = gr.Plot(show_label=False)

                    with gr.Tab("Health & snapshot"):
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=5):
                                with gr.Row(elem_classes=["toolbar"]):
                                    gr.Markdown("### Unit tests")
                                    run_unit_tests_btn = gr.Button("Run all", size="sm", scale=0, min_width=90)
                                unit_tests_overview_md = gr.Markdown(value=format_test_results_markdown(get_cached_or_run_tests()))
                                with gr.Accordion("Test output", open=False):
                                    unit_tests_stdout = gr.Code(language="markdown", lines=12, interactive=False, show_label=False)
                            with gr.Column(scale=7):
                                with gr.Row(elem_classes=["toolbar"]):
                                    gr.Markdown("### Snapshot for an assistant")
                                    refresh_snapshot_btn = gr.Button("Build", size="sm", scale=0, min_width=90)
                                diag_overview_md = gr.Markdown("")
                                diag_export_raw = gr.Code(language="markdown", lines=18, interactive=False, show_label=False)

            # =====================================================================================
            # SETUP
            # =====================================================================================
            with gr.Tab("Setup"):
                with gr.Tabs():
                    with gr.Tab("Training config"):
                        gr.Markdown("Saved to `config/default_config.yaml`; takes effect on the next start. Gamma "
                                    "belongs to the reward version (the potentials are built on it), so it is set "
                                    "there, not here.", elem_classes=["hint"])
                        with gr.Row(equal_height=False):
                            with gr.Column():
                                gr.Markdown("#### PPO")
                                with gr.Row():
                                    gae_lambda_input = gr.Number(value=hp_cfg.get("gae_lambda", 0.95), label="GAE λ")
                                    n_epochs_input = gr.Number(value=hp_cfg.get("n_epochs", 4), precision=0, label="Epochs per iteration")
                                with gr.Row():
                                    batch_size_input = gr.Number(value=hp_cfg.get("batch_size", 16384), precision=0, label="Rollout batch (steps)")
                                    mini_batch_input = gr.Number(value=hp_cfg.get("mini_batch_size", 1024), precision=0, label="Minibatch")
                                gr.Markdown(f"Gamma: **{hp_cfg.get('gamma')}** (reward {active_version()})")
                            with gr.Column():
                                gr.Markdown("#### Simulation")
                                with gr.Row():
                                    num_envs_input = gr.Number(value=env_cfg.get("num_envs", 128), precision=0, label="Arenas")
                                    tick_skip_input = gr.Number(value=env_cfg.get("tick_skip", 8), precision=0, label="Tick skip")
                                with gr.Row():
                                    max_steps_input = gr.Number(value=env_cfg.get("max_episode_steps", 600), precision=0, label="Max episode steps")
                                    game_mode_dropdown = gr.Dropdown(["1v1", "2v2", "3v3"], value=env_cfg.get("game_mode", "1v1"), label="Mode")
                            with gr.Column():
                                gr.Markdown("#### Checkpoints")
                                autosave_interval_input = gr.Number(value=log_cfg.get("autosave_interval", 20), precision=0,
                                                                    label="Autosave every (iterations)", info="Rewrites latest_model.pt only.")
                                checkpoint_interval_input = gr.Number(value=log_cfg.get("checkpoint_interval", 200), precision=0,
                                                                      label="League checkpoint every (iterations)")
                                archive_stride_input = gr.Number(value=log_cfg.get("archive_stride", 5000), precision=0,
                                                                 label="Keep one forever every (iterations)", info="0 disables.")
                        save_cfg_btn = gr.Button("Save config", variant="primary")
                        cfg_save_msg = gr.Markdown("")

                    with gr.Tab("Replays & pretraining"):
                        def build_replay_stats_md():
                            st = ReplayParser().get_pool_stats()
                            frames = st.get("total_frames", 0)
                            if not frames:
                                return ("<div class='cyber-panel'><b>Replay pool is empty.</b> Replay starts fall back to "
                                        "kickoffs until replays are ingested.</div>")
                            return (f"<div class='cyber-panel'><b>Replay pool:</b> {st.get('num_matches', 0)} matches · "
                                    f"{frames:,} frames (~{frames / 15.0 / 60.0:,.0f} min of play) · "
                                    f"{st.get('file_size_mb', 0.0):.1f} MB</div>")

                        replay_stats_box = gr.HTML(build_replay_stats_md())
                        with gr.Row(equal_height=False):
                            with gr.Column():
                                gr.Markdown("#### Ingest replays")
                                with gr.Row():
                                    demos_dir_input = gr.Textbox(value=get_default_demo_dir(), label="Replay folder", scale=4)
                                    scan_demos_btn = gr.Button("Scan", size="sm", scale=1, min_width=70)
                                with gr.Row():
                                    max_replays_input = gr.Number(value=20, precision=0, label="Max replays", minimum=1)
                                    sort_replays_radio = gr.Radio(["newest", "oldest"], value="newest", label="Order")
                                demos_table = gr.Dataframe(headers=["File", "Size (KB)", "Modified"], datatype=["str", "number", "str"],
                                                           value=[], label="Found")
                                with gr.Row():
                                    ingest_selected_btn = gr.Button("Ingest found", variant="primary")
                                    ingest_all_btn = gr.Button("Ingest whole folder")
                                    clear_pool_btn = gr.Button("Clear pool", variant="stop")
                                replays_status_box = gr.Markdown("")
                                replay_uploader = gr.File(file_count="multiple", file_types=[".zip", ".npz", ".json", ".replay"],
                                                          label="Or drop replay files / archives")
                                upload_status_box = gr.Markdown("")
                            with gr.Column():
                                gr.Markdown("#### Behavioural cloning pretrainer")
                                with gr.Row():
                                    pretrain_epochs = gr.Number(value=5, precision=0, label="Epochs", minimum=1)
                                    pretrain_lr_input = gr.Number(value=0.0005, label="Learning rate")
                                    pretrain_batch_dropdown = gr.Dropdown([64, 128, 256, 512], value=256, label="Batch")
                                _pre_ckpts = get_available_checkpoints()
                                pretrain_base_dropdown = gr.Dropdown(choices=_pre_ckpts, value=_pre_ckpts[0], label="Start from")
                                with gr.Row():
                                    run_pretrain_btn = gr.Button("Run pretraining", variant="primary")
                                    stop_pretrain_btn = gr.Button("Stop", variant="stop")
                                pretrain_status_box = gr.Markdown("")
                                gr.Markdown("#### Replay guidance during PPO (applies live)")
                                with gr.Row():
                                    bc_weight_input = gr.Number(value=float(hp_cfg.get("bc_regularization_weight", 0.0)), label="Weight")
                                    bc_decay_input = gr.Number(value=int(hp_cfg.get("bc_decay_steps", 500_000_000)), precision=0,
                                                               label="Decays to 0 over (steps)")
                                apply_bc_btn = gr.Button("Apply", size="sm")
                                bc_msg = gr.Markdown("")

                    with gr.Tab("Custom scenarios"):
                        _custom_share = float(default_cfg["scenarios"].get("custom_prob", 0.0) or 0.0)
                        gr.Markdown(
                            f"The active reward version samples custom scenarios **{_custom_share * 100:.0f}%** of the time"
                            + (" (they are not used in training)" if _custom_share == 0 else "")
                            + ". The mix belongs to the version; this library is for building and previewing drills.",
                            elem_classes=["hint"])
                        sc_mgr = ScenarioManager.get_instance()
                        all_scenarios = sc_mgr.get_all_scenarios()
                        initial_sc = all_scenarios[0] if all_scenarios else DEFAULT_CUSTOM_SCENARIOS[0]
                        with gr.Row(equal_height=False):
                            with gr.Column(scale=5):
                                sc_preview_plot = gr.Plot(value=render_scenario_visual_guide(initial_sc), show_label=False)
                                with gr.Row():
                                    load_scenario_dropdown = gr.Dropdown(
                                        choices=[f"{sc['name']} ({sc['id']})" for sc in all_scenarios],
                                        value=f"{initial_sc['name']} ({initial_sc['id']})" if all_scenarios else None,
                                        label="Library", scale=3)
                                    load_scenario_btn = gr.Button("Load", size="sm", scale=1, min_width=70)
                                preset_dropdown = gr.Dropdown(
                                    choices=["(Select Template Preset...)"] + [sc["name"] for sc in DEFAULT_CUSTOM_SCENARIOS],
                                    value="(Select Template Preset...)", label="Start from a template")
                                with gr.Accordion("Preview 150 steps with the latest model", open=False):
                                    sim_scenario_btn = gr.Button("Simulate", size="sm")
                                    sc_sim_plot = gr.Plot(show_label=False)
                                    sc_sim_stats = gr.JSON(label="Rollout")
                            with gr.Column(scale=6):
                                with gr.Row():
                                    sc_id_input = gr.Textbox(label="ID", value=initial_sc.get("id", ""), scale=2)
                                    sc_name_input = gr.Textbox(label="Name", value=initial_sc.get("name", ""), scale=3)
                                    sc_enabled_cb = gr.Checkbox(label="Enabled", value=initial_sc.get("enabled", True), scale=1)
                                sc_desc_input = gr.Textbox(label="Description", value=initial_sc.get("description", ""), lines=2)
                                gr.Markdown("**Car (blue)**")
                                with gr.Row():
                                    car_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc["car"]["pos"][0]), step=25.0, label="X")
                                    car_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc["car"]["pos"][1]), step=25.0, label="Y")
                                    car_pos_z = gr.Slider(17.0, 1600.0, value=float(initial_sc["car"]["pos"][2]), step=10.0, label="Z")
                                with gr.Row():
                                    car_yaw = gr.Slider(-180.0, 180.0, value=float(initial_sc["car"].get("yaw", 90.0)), step=5.0, label="Yaw (90 = +Y)")
                                    car_speed = gr.Slider(0.0, 2300.0, value=float(math.hypot(initial_sc["car"]["vel"][0], initial_sc["car"]["vel"][1])), step=25.0, label="Speed")
                                    car_boost = gr.Slider(0.0, 100.0, value=float(initial_sc["car"].get("boost", 50.0)), step=5.0, label="Boost")
                                gr.Markdown("**Ball**")
                                with gr.Row():
                                    ball_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc["ball"]["pos"][0]), step=25.0, label="X")
                                    ball_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc["ball"]["pos"][1]), step=25.0, label="Y")
                                    ball_pos_z = gr.Slider(93.15, 1800.0, value=float(initial_sc["ball"]["pos"][2]), step=10.0, label="Z")
                                with gr.Row():
                                    ball_vel_x = gr.Slider(-2500.0, 2500.0, value=float(initial_sc["ball"]["vel"][0]), step=25.0, label="Vel X")
                                    ball_vel_y = gr.Slider(-2500.0, 2500.0, value=float(initial_sc["ball"]["vel"][1]), step=25.0, label="Vel Y")
                                    ball_vel_z = gr.Slider(-1500.0, 1500.0, value=float(initial_sc["ball"]["vel"][2]), step=25.0, label="Vel Z")
                                gr.Markdown("**Opponent (orange) and variation**")
                                with gr.Row():
                                    opp_mode_radio = gr.Radio(["goalie", "shadow", "custom", "none"],
                                                              value=initial_sc.get("opponent", {}).get("mode", "goalie"), label="Placement")
                                    opp_boost = gr.Number(value=float(initial_sc.get("opponent", {}).get("boost", 60.0)), label="Boost")
                                with gr.Row(visible=(initial_sc.get("opponent", {}).get("mode", "goalie") == "custom")) as opp_custom_row:
                                    opp_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc.get("opponent", {}).get("pos", [0, 4800, 17])[0]), step=25.0, label="X")
                                    opp_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc.get("opponent", {}).get("pos", [0, 4800, 17])[1]), step=25.0, label="Y")
                                    opp_yaw = gr.Slider(-180.0, 180.0, value=float(initial_sc.get("opponent", {}).get("yaw", -90.0)), step=5.0, label="Yaw")
                                with gr.Row():
                                    pos_jitter = gr.Number(value=float(initial_sc.get("variance", {}).get("pos_jitter", 80.0)), label="Position jitter (± uu)")
                                    vel_jitter = gr.Number(value=float(initial_sc.get("variance", {}).get("vel_jitter", 60.0)), label="Velocity jitter (± uu/s)")
                                    mirror_symmetry = gr.Checkbox(value=bool(initial_sc.get("variance", {}).get("mirror_symmetry", True)), label="Mirror left/right half the time")
                                with gr.Row():
                                    save_scenario_btn = gr.Button("Save", variant="primary")
                                    new_scenario_btn = gr.Button("New")
                                    delete_scenario_btn = gr.Button("Delete", variant="stop")
                                scenario_action_msg = gr.Markdown("")

        # =========================================================================================
        # HANDLERS
        # =========================================================================================
        def _button_updates(status):
            running, paused = status.get("running", False), status.get("paused", False)
            return (
                gr.update(value="Start", interactive=not running),
                gr.update(value="Resume" if paused else "Pause", variant="primary" if paused else "secondary", interactive=running),
                gr.update(interactive=running),
            )

        def sync_ui_state(feedback_msg: str = ""):
            status = mgr.get_status_info()
            return (build_status_card_html(status, feedback_msg),) + _button_updates(status)

        def on_start(resume_latest: bool = True):
            ckpt = None
            if resume_latest:
                # Only checkpoints trained on the active reward version continue its run; the first
                # start of a version resumes from the checkpoint its snapshot pins
                # (config/reward_versions/<v>.json "start_checkpoint"). Unstamped = v2.
                version = active_version()
                best_ckpt, best_iter = None, -1
                candidates = ["checkpoints/latest_model.pt"] + glob.glob("checkpoints/checkpoint_iter_*.pt")
                for c_path in candidates:
                    if not os.path.exists(c_path):
                        continue
                    s = _checkpoint_stamp(c_path)
                    if s.get("version") != version or s.get("iteration") is None:
                        continue
                    if int(s["iteration"]) > best_iter:
                        best_iter, best_ckpt = int(s["iteration"]), c_path
                start = load_snapshot(version).get("start_checkpoint")
                ckpt = best_ckpt or (start if start and os.path.exists(start) else None)
            success, msg = mgr.start_training(checkpoint_path=ckpt)
            time.sleep(0.3)
            return sync_ui_state(f"{'✅' if success else '❌'} {msg}")

        def on_stop():
            success, msg = mgr.stop_training()
            time.sleep(0.3)
            return sync_ui_state(f"{'🛑' if success else '⚠️'} {msg}")

        def on_pause():
            success, msg = mgr.toggle_pause()
            return sync_ui_state(f"ℹ️ {msg}")

        def on_save_checkpoint():
            success, msg = mgr.trigger_save_checkpoint()
            return sync_ui_state(f"{'💾' if success else '⚠️'} {msg}")

        control_outputs = [status_card, start_btn, pause_btn, stop_btn]
        start_btn.click(fn=on_start, inputs=[resume_chk], outputs=control_outputs)
        stop_btn.click(fn=on_stop, outputs=control_outputs)
        pause_btn.click(fn=on_pause, outputs=control_outputs)
        ckpt_btn.click(fn=on_save_checkpoint, outputs=control_outputs)

        def _save_yaml_section(section: str, values: dict):
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                base_cfg.setdefault(section, {}).update(values)
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass

        # ---- Training tab --------------------------------------------------------------------
        def on_apply_live_hyperparams(lr_val, ent_val, clip_val):
            payload = {"learning_rate": float(lr_val), "ent_coef": float(ent_val), "clip_range": float(clip_val)}
            mgr.update_live_config(payload)
            _save_yaml_section("hyperparameters", payload)
            return f"Applied at {time.strftime('%H:%M:%S')}: LR {float(lr_val):.2e}, entropy {float(ent_val):.4f}, clip {float(clip_val):.2f}"

        live_hp_btn.click(fn=on_apply_live_hyperparams, inputs=[lr_input, ent_input, clip_input], outputs=[live_hp_msg])

        def on_apply_opponent_mix(opp_list, opp_envs):
            # The slider is in environments; the league stores a ratio. The slider only lands on
            # worker-block boundaries, so every stored ratio is one the scheduler honours exactly.
            opp_envs = max(0, min(ui_num_envs, int(opp_envs or 0)))
            selected = []
            for item in (opp_list or []):
                text = str(item).strip()
                if text:
                    selected.append("heuristic" if text.startswith("Heuristic") else text)
            # A repeated entry would quietly receive double weight, since the share is split evenly
            selected = list(dict.fromkeys(selected))
            payload = {"training_opponents": selected, "training_opponent_ratio": opp_envs / float(ui_num_envs)}
            mgr.update_live_config(payload)
            _save_yaml_section("league", payload)
            if not selected or opp_envs <= 0:
                return f"Fixed opponents cleared at {time.strftime('%H:%M:%S')}: the league picks every opponent."
            each = opp_envs // len(selected)
            names = ", ".join(f"`{os.path.basename(x)}`" for x in selected)
            return f"Applied at {time.strftime('%H:%M:%S')}: {names} on {opp_envs} environments ({each} each)."

        apply_opp_btn.click(fn=on_apply_opponent_mix, inputs=[training_opponents_select, baseline_opp_slider], outputs=[opp_apply_msg])
        refresh_opponent_btn.click(fn=lambda: gr.Dropdown(choices=get_available_opponent_options()), outputs=[training_opponents_select])

        def on_apply_gauntlet_budget(series_per_step):
            step = max(1, int(series_per_step or 1))
            # Live config reaches the trainer, which hands it to each grading child; the yaml keeps it
            mgr.update_live_config({"contender_series_per_step": step})
            _save_yaml_section("league", {"contender_series_per_step": step})
            return f"Applied at {time.strftime('%H:%M:%S')}: {step} series per trial."

        apply_budget_btn.click(fn=on_apply_gauntlet_budget, inputs=[gauntlet_budget_radio], outputs=[budget_apply_msg])

        def on_change_view_mode(mode_val):
            _last_view_mode[0] = mode_val
            return render_training_curves_plot(mode=mode_val)

        metrics_window_radio.change(fn=on_change_view_mode, inputs=[metrics_window_radio], outputs=[live_metrics_plot])
        refresh_metrics_btn.click(fn=on_change_view_mode, inputs=[metrics_window_radio], outputs=[live_metrics_plot])
        refresh_logs_btn.click(fn=mgr.get_logs, outputs=[console_output])
        clear_logs_btn.click(fn=lambda: "", outputs=[console_output])

        # ---- Evaluation tab ------------------------------------------------------------------
        def _load_or_none(path):
            try:
                return eval_results.load(path) if path else None
            except Exception:
                return None

        def on_show_eval(result_path, base_path, only_diff):
            res = _load_or_none(result_path)
            # a result compared with itself says nothing
            base = _load_or_none(base_path) if base_path and base_path != result_path else None
            if res is None:
                empty = ("<div class='ev-empty'>No eval results yet. Pick a checkpoint on the left and run the "
                         "suite; the baseline for this reward version is compared automatically.</div>")
                return empty, eval_results.trend_figure(active_version(), base), ""
            name = os.path.splitext(os.path.basename(result_path))[0]
            base_name = os.path.splitext(os.path.basename(base_path))[0] if base is not None else None
            card = eval_results.scorecard_html(res, base, name, base_name)
            table = (eval_results.comparison_table_html(base, res, only_clear=bool(only_diff)) if base is not None
                     else "<div class='ev-empty'>Pick a result to compare with to see every metric side by side.</div>")
            stamp = res.get("reward_identity") if isinstance(res.get("reward_identity"), dict) else {}
            return card, eval_results.trend_figure(stamp.get("version") or active_version(), base), table

        eval_view_inputs = [eval_result_dd, eval_base_dd, eval_only_diff]
        eval_view_outputs = [eval_scorecard, eval_trend_plot, eval_table]
        for comp in (eval_result_dd, eval_base_dd, eval_only_diff):
            comp.change(fn=on_show_eval, inputs=eval_view_inputs, outputs=eval_view_outputs)

        def on_refresh_evals(current, base):
            choices = _eval_result_choices()
            paths = [p for _, p in choices]
            cur = current if current in paths else (paths[0] if paths else None)
            b = base if (base in paths or base == "") else (_default_baseline_path() or "")
            return (gr.Dropdown(choices=choices, value=cur), gr.Dropdown(choices=[("(none)", "")] + choices, value=b))

        refresh_evals_btn.click(fn=on_refresh_evals, inputs=[eval_result_dd, eval_base_dd],
                                outputs=[eval_result_dd, eval_base_dd]).then(
            fn=on_show_eval, inputs=eval_view_inputs, outputs=eval_view_outputs)
        def on_rescan_eval_checkpoints(reference):
            choices = eval_checkpoint_choices()
            paths = [p for _, p in choices]
            ref = reference if reference in paths else (_default_reference_path() or "")
            return gr.Dropdown(choices=choices), gr.Dropdown(choices=[("(none)", "")] + choices, value=ref)

        refresh_eval_ckpts_btn.click(fn=on_rescan_eval_checkpoints, inputs=[eval_ref_dd],
                                     outputs=[eval_ckpt_dd, eval_ref_dd])

        def on_run_eval(ckpt, size, workers, name, reference):
            import subprocess
            if not ckpt or not os.path.exists(ckpt):
                yield "Pick a checkpoint first.", gr.skip(), gr.skip()
                return
            cmd = [sys.executable, "-u", "scripts/eval_suite.py", "--checkpoint", ckpt,
                   "--workers", str(max(1, int(workers or 1)))]
            if reference:
                if not os.path.exists(reference):
                    yield f"Head-to-head checkpoint not found: {reference}", gr.skip(), gr.skip()
                    return
                if os.path.normpath(reference) == os.path.normpath(ckpt):
                    yield "A checkpoint played against itself says nothing; pick another or (none).", gr.skip(), gr.skip()
                    return
                cmd += ["--reference", reference]
            if size == "quick":
                cmd.append("--quick")
            if name and str(name).strip():
                cmd += ["--name", re.sub(r"[^\w.-]", "_", str(name).strip())]
            lines = [f"$ {' '.join(cmd[1:])}"]
            yield "\n".join(lines), gr.skip(), gr.skip()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                    encoding="utf-8", errors="replace")
            written = None
            for line in proc.stdout:
                line = line.rstrip()
                if line.startswith("wrote "):
                    written = line.split()[1].replace("\\", "/")
                if line.startswith(("eval suite", "  done", "wrote", "Traceback", "  File", "Error", "WARNING")) or "Error" in line:
                    lines.append(line)
                    yield "\n".join(lines[-9:]), gr.skip(), gr.skip()
            proc.wait()
            lines.append("Finished." if proc.returncode == 0 else f"Failed (exit {proc.returncode}).")
            choices = _eval_result_choices()
            yield ("\n".join(lines[-9:]),
                   gr.Dropdown(choices=choices, value=written or (choices[0][1] if choices else None)),
                   gr.Dropdown(choices=[("(none)", "")] + choices))

        run_eval_btn.click(fn=on_run_eval, inputs=[eval_ckpt_dd, eval_size_radio, eval_workers, eval_name, eval_ref_dd],
                           outputs=[eval_log, eval_result_dd, eval_base_dd]).then(
            fn=on_show_eval, inputs=eval_view_inputs, outputs=eval_view_outputs)

        # ---- League tab ----------------------------------------------------------------------
        def on_refresh_cockpit_leaderboard():
            ts_evaluator.load_leaderboard()
            return (build_cockpit_leaderboard_summary_html(ts_evaluator), build_league_wire_and_queue_html(ts_evaluator),
                    get_cockpit_leaderboard_df(ts_evaluator))

        refresh_cockpit_lb_btn.click(fn=on_refresh_cockpit_leaderboard,
                                     outputs=[cockpit_lb_summary, cockpit_league_ticker, cockpit_lb_table])

        ts_select_all_btn.click(fn=lambda: gr.Dropdown(value=get_available_checkpoints()), outputs=[ts_ckpt_multiselect])
        ts_clear_all_btn.click(fn=lambda: gr.Dropdown(value=[]), outputs=[ts_ckpt_multiselect])
        ts_refresh_ckpts_btn.click(fn=lambda: gr.Dropdown(choices=get_available_checkpoints()), outputs=[ts_ckpt_multiselect])

        def on_ts_reset_leaderboard():
            ts_evaluator.reset_leaderboard()
            return "Manual leaderboard reset.", ts_evaluator.get_leaderboard_dataframe(), ts_evaluator.render_leaderboard_plot()

        reset_leaderboard_btn.click(fn=on_ts_reset_leaderboard, outputs=[ts_status_md, ts_leaderboard_table, ts_leaderboard_plot])

        def on_run_ts_tournament(ckpts, anchors, series_per_pair, steps, enable_ot):
            model_list = list(ckpts or [])
            for a in (anchors or []):
                al = a.lower()
                if "heuristic" in al:
                    model_list.append("heuristic")
                elif "pretrained" in al:
                    model_list.append("checkpoints/pretrained_baseline.pt")
                elif "necto" in al:
                    model_list.append("checkpoints/necto-model.pt")
                elif "nexto" in al:
                    model_list.append("checkpoints/nexto-model.pt")
            model_list = list(dict.fromkeys(os.path.normpath(m).replace("\\", "/") for m in model_list if m))
            if len(model_list) < 2:
                yield "Pick at least two contestants.", ts_evaluator.get_leaderboard_dataframe(), ts_evaluator.render_leaderboard_plot()
                return
            yield f"Starting: {len(model_list)} models.", ts_evaluator.get_leaderboard_dataframe(), ts_evaluator.render_leaderboard_plot()
            for update in ts_evaluator.run_tournament(model_paths=model_list, series_per_pair=max(1, int(series_per_pair)),
                                                      max_steps=int(steps), device="cpu"):
                res = ", ".join(f"{r['blue_name']} {r['blue_goals']}-{r['orange_goals']} {r['orange_name']}"
                                f"{' (OT)' if r['overtime'] else ''}" for r in update["results"])
                yield (f"Pairing {update['pairing_index']}/{update['total_pairings']}: {res}",
                       ts_evaluator.get_leaderboard_dataframe(), ts_evaluator.render_leaderboard_plot())
            yield f"Done: {len(model_list)} models ranked.", ts_evaluator.get_leaderboard_dataframe(), ts_evaluator.render_leaderboard_plot()

        run_tournament_btn.click(fn=on_run_ts_tournament,
                                 inputs=[ts_ckpt_multiselect, ts_anchors_checkbox, ts_matches, ts_steps, ts_overtime_check],
                                 outputs=[ts_status_md, ts_leaderboard_table, ts_leaderboard_plot])

        # ---- Diagnostics tab -----------------------------------------------------------------
        def on_refresh_behaviour(window):
            tel = run_telemetry(window=int(window or 10))
            return behaviour_flags_markdown(tel), render_behaviour_plot(tel)

        refresh_diag_btn.click(fn=on_refresh_behaviour, inputs=[diag_window], outputs=[diag_flags, diag_behaviour_plot])
        diag_window.change(fn=on_refresh_behaviour, inputs=[diag_window], outputs=[diag_flags, diag_behaviour_plot])

        def on_scan_checkpoints():
            ckpts = get_available_checkpoints()
            return gr.Dropdown(choices=ckpts, value=ckpts[0]), gr.Dropdown(choices=ckpts, value=ckpts[0])

        refresh_ckpts_btn.click(fn=on_scan_checkpoints, outputs=[ckpt_dropdown, orange_ckpt_dropdown])
        opponent_mode.change(fn=lambda m: gr.Dropdown(visible=(m == "Another checkpoint")), inputs=[opponent_mode],
                             outputs=[orange_ckpt_dropdown])

        def on_run_simulation(blue_choice, opp_mode, orange_choice, steps):
            blue_path = blue_choice.split(" ")[0] if blue_choice else None
            if not blue_path or not os.path.exists(blue_path):
                blue_path = "checkpoints/latest_model.pt" if os.path.exists("checkpoints/latest_model.pt") else None
            if opp_mode == "Heuristic chaser":
                orange_path = "baseline"
            elif opp_mode == "Another checkpoint":
                orange_path = orange_choice.split(" ")[0] if orange_choice else None
                if not orange_path or not os.path.exists(orange_path):
                    orange_path = "baseline"
            else:
                orange_path = "same_as_blue"
            res = simulate_match(blue_model_path=blue_path, orange_model_path=orange_path, max_steps=int(steps or 400))
            if isinstance(res, dict):
                p_fig, r_fig, stats = res.get("plot"), res.get("reward_plot"), res.get("stats", {})
            else:
                p_fig, r_fig, stats = res
            total_s = stats.get("simulation_steps", stats.get("total_steps", int(steps or 400)))
            md = (f"**Blue {stats.get('blue_goals', stats.get('goals_blue', 0))} – "
                  f"{stats.get('orange_goals', stats.get('goals_orange', 0))} Orange** over {total_s / 15.0:.0f} s\n\n"
                  f"Touches: blue {stats.get('blue_touches', stats.get('touches_blue', 0))}, "
                  f"orange {stats.get('orange_touches', stats.get('touches_orange', 0))}\n\n"
                  f"Reward (active version): blue {stats.get('blue_total_reward', stats.get('rewards_blue', 0.0)):+.2f}, "
                  f"orange {stats.get('orange_total_reward', stats.get('rewards_orange', 0.0)):+.2f}")
            return p_fig, r_fig, md

        run_sim_btn.click(fn=on_run_simulation, inputs=[ckpt_dropdown, opponent_mode, orange_ckpt_dropdown, sim_steps],
                          outputs=[visualizer_plot, reward_breakdown_plot, sim_stats_box])

        def on_run_unit_tests():
            res = run_all_unit_tests(verbose=True)
            return format_test_results_markdown(res), res.get("raw_output", "")

        run_unit_tests_btn.click(fn=on_run_unit_tests, outputs=[unit_tests_overview_md, unit_tests_stdout])
        refresh_snapshot_btn.click(fn=build_full_diagnostic_export, outputs=[diag_overview_md, diag_export_raw])

        # ---- Setup tab -----------------------------------------------------------------------
        def on_save_yaml(gae, bs, mbs, n_ep, n_env, t_skip, m_steps, g_mode, autosave_int, ckpt_int, archive_str):
            base_cfg = load_yaml_config("config/default_config.yaml")
            # Merged, never replaced: the yaml carries keys this page has no field for
            base_cfg.setdefault("hyperparameters", {}).update({
                "gae_lambda": float(gae), "batch_size": int(bs), "mini_batch_size": int(mbs), "n_epochs": int(n_ep)})
            base_cfg.setdefault("environment", {}).update({
                "num_envs": int(n_env), "tick_skip": int(t_skip), "max_episode_steps": int(m_steps), "game_mode": str(g_mode)})
            base_cfg.setdefault("logging", {}).update({
                "autosave_interval": max(1, int(autosave_int)), "checkpoint_interval": max(1, int(ckpt_int)),
                "archive_stride": max(0, int(archive_str))})
            save_yaml_config(base_cfg)
            return f"Saved at {time.strftime('%H:%M:%S')}. Takes effect on the next start."

        save_cfg_btn.click(fn=on_save_yaml, inputs=[
            gae_lambda_input, batch_size_input, mini_batch_input, n_epochs_input, num_envs_input, tick_skip_input,
            max_steps_input, game_mode_dropdown, autosave_interval_input, checkpoint_interval_input, archive_stride_input],
            outputs=[cfg_save_msg])

        def on_apply_bc(w, decay):
            payload = {"bc_regularization_weight": float(w or 0.0), "bc_decay_steps": int(decay or 0)}
            mgr.update_live_config(payload)
            _save_yaml_section("hyperparameters", payload)
            return f"Applied at {time.strftime('%H:%M:%S')}."

        apply_bc_btn.click(fn=on_apply_bc, inputs=[bc_weight_input, bc_decay_input], outputs=[bc_msg])

        def on_scan_demos(demo_dir, max_replays, sort_mode):
            files = ReplayParser(demo_dir=str(demo_dir).strip()).scan_demos(max_replays=int(max_replays or 20), sort=str(sort_mode))
            rows = []
            for fp in files:
                try:
                    rows.append([os.path.basename(fp), round(os.path.getsize(fp) / 1024, 1),
                                 time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(fp)))])
                except OSError:
                    rows.append([os.path.basename(fp), 0.0, "?"])
            return rows, f"Found {len(rows)} replays in `{demo_dir}`."

        scan_demos_btn.click(fn=on_scan_demos, inputs=[demos_dir_input, max_replays_input, sort_replays_radio],
                             outputs=[demos_table, replays_status_box])

        def _ingest_message(p, res, demo_dir):
            if res["total_frames"] > 0:
                return f"Ingested {res['parsed_files']} replays ({res['total_frames']:,} frames) in {res['elapsed_seconds']:.1f} s."
            rej = getattr(p, "last_ingest_report", {}).get("rejected_files", [])
            return (f"No frames ingested: {len(rej)} file(s) could not be decoded." if rej
                    else f"No replay files found in `{demo_dir}`.")

        def on_ingest_replays(demo_dir, max_replays, sort_mode):
            p = ReplayParser(demo_dir=str(demo_dir).strip())
            res = p.ingest_directory(max_replays=int(max_replays or 20), sort=str(sort_mode))
            return build_replay_stats_md(), _ingest_message(p, res, demo_dir)

        def on_ingest_all_replays(demo_dir):
            p = ReplayParser(demo_dir=str(demo_dir).strip())
            res = p.ingest_directory(max_replays=999999, sort="newest")
            return build_replay_stats_md(), _ingest_message(p, res, demo_dir)

        def on_clear_replays():
            ReplayParser().clear_pool()
            return build_replay_stats_md(), "Replay pool cleared."

        ingest_selected_btn.click(fn=on_ingest_replays, inputs=[demos_dir_input, max_replays_input, sort_replays_radio],
                                  outputs=[replay_stats_box, replays_status_box])
        ingest_all_btn.click(fn=on_ingest_all_replays, inputs=[demos_dir_input], outputs=[replay_stats_box, replays_status_box])
        clear_pool_btn.click(fn=on_clear_replays, outputs=[replay_stats_box, replays_status_box])

        def on_upload_ingest(uploaded_files):
            if not uploaded_files:
                return build_replay_stats_md(), "No files uploaded."
            p = ReplayParser()
            added, frames = 0, 0
            import shutil
            for fp in [f.name if hasattr(f, "name") else str(f) for f in uploaded_files]:
                if os.path.splitext(fp)[1].lower() == ".zip":
                    n, fr = p.ingest_zip(fp)
                    added, frames = added + n, frames + fr
                else:
                    try:
                        shutil.copy2(fp, os.path.join(p.demo_dir, os.path.basename(fp)))
                        added += 1
                    except OSError:
                        pass
            if frames == 0 and added > 0:
                frames = p.ingest_directory(max_replays=added, sort="newest").get("total_frames", 0)
            msg = (f"Ingested {added} file(s), {frames:,} frames." if frames > 0
                   else "The uploaded files yielded no frames. Check they are valid Rocket League replays.")
            return build_replay_stats_md(), msg

        replay_uploader.upload(fn=on_upload_ingest, inputs=[replay_uploader], outputs=[replay_stats_box, upload_status_box])

        def on_run_pretraining(epochs, lr, batch_size, base_ckpt):
            chosen = base_ckpt.split(" ")[0] if base_ckpt and not base_ckpt.startswith("checkpoints/latest_model.pt (none") else None
            res = bc_trainer.train(epochs=int(epochs or 1), batch_size=int(batch_size), lr=float(lr), base_checkpoint=chosen)
            return ("**Done:** " if res.get("success", True) else "**Failed:** ") + str(res.get("message", ""))

        def on_stop_pretraining():
            if bc_trainer.is_running():
                bc_trainer.request_stop()
                return "Stop requested."
            return "The pretrainer is not running."

        run_pretrain_btn.click(fn=on_run_pretraining, inputs=[pretrain_epochs, pretrain_lr_input, pretrain_batch_dropdown,
                                                             pretrain_base_dropdown], outputs=[pretrain_status_box])
        stop_pretrain_btn.click(fn=on_stop_pretraining, outputs=[pretrain_status_box])

        # ---- Custom scenarios ----------------------------------------------------------------
        def assemble_scenario_payload(s_id, s_name, s_enabled, s_desc, c_x, c_y, c_z, c_yaw, c_spd, c_boost,
                                      b_x, b_y, b_z, b_vx, b_vy, b_vz, o_mode, o_boost, o_x, o_y, o_yaw,
                                      p_jit, v_jit, mirror) -> dict:
            yaw_rad = math.radians(float(c_yaw))
            spd = float(c_spd)
            opp = {"mode": str(o_mode), "boost": float(o_boost or 0.0)}
            if o_mode == "custom":
                opp.update(pos=[float(o_x), float(o_y), 17.0], yaw=float(o_yaw), vel=[0.0, 0.0, 0.0])
            return {
                "id": str(s_id).strip(), "name": str(s_name).strip(), "enabled": bool(s_enabled),
                "description": str(s_desc).strip(),
                "car": {"pos": [float(c_x), float(c_y), float(c_z)],
                        "vel": [spd * math.cos(yaw_rad), spd * math.sin(yaw_rad), 0.0],
                        "yaw": float(c_yaw), "boost": float(c_boost)},
                "ball": {"pos": [float(b_x), float(b_y), float(b_z)], "vel": [float(b_vx), float(b_vy), float(b_vz)]},
                "opponent": opp,
                "variance": {"pos_jitter": float(p_jit or 0.0), "vel_jitter": float(v_jit or 0.0), "mirror_symmetry": bool(mirror)},
            }

        all_sc_inputs = [sc_id_input, sc_name_input, sc_enabled_cb, sc_desc_input,
                         car_pos_x, car_pos_y, car_pos_z, car_yaw, car_speed, car_boost,
                         ball_pos_x, ball_pos_y, ball_pos_z, ball_vel_x, ball_vel_y, ball_vel_z,
                         opp_mode_radio, opp_boost, opp_pos_x, opp_pos_y, opp_yaw,
                         pos_jitter, vel_jitter, mirror_symmetry]

        def on_update_visual_preview(*args):
            return render_scenario_visual_guide(assemble_scenario_payload(*args))

        for comp in (car_pos_x, car_pos_y, car_pos_z, car_yaw, car_speed, car_boost, ball_pos_x, ball_pos_y, ball_pos_z,
                     ball_vel_x, ball_vel_y, ball_vel_z, opp_mode_radio, opp_pos_x, opp_pos_y, opp_yaw):
            # sliders redraw on release, not on every drag tick
            listener = comp.release if isinstance(comp, gr.Slider) else comp.change
            listener(fn=on_update_visual_preview, inputs=all_sc_inputs, outputs=[sc_preview_plot])
        opp_mode_radio.change(fn=lambda m: gr.Row(visible=(m == "custom")), inputs=[opp_mode_radio], outputs=[opp_custom_row])

        def _scenario_values(match, sc_id=None, name=None):
            c, b = match["car"], match["ball"]
            o, v = match.get("opponent", {}), match.get("variance", {})
            o_pos = o.get("pos", [0, 4800, 17])
            return (sc_id or match["id"], name or match["name"], match.get("enabled", True), match.get("description", ""),
                    c["pos"][0], c["pos"][1], c["pos"][2], c.get("yaw", 90.0), math.hypot(c["vel"][0], c["vel"][1]), c.get("boost", 50.0),
                    b["pos"][0], b["pos"][1], b["pos"][2], b["vel"][0], b["vel"][1], b["vel"][2],
                    o.get("mode", "goalie"), o.get("boost", 60.0), o_pos[0], o_pos[1], o.get("yaw", -90.0),
                    v.get("pos_jitter", 80.0), v.get("vel_jitter", 60.0), v.get("mirror_symmetry", True))

        def on_select_preset_template(preset_name):
            match = next((s for s in DEFAULT_CUSTOM_SCENARIOS if s["name"] == preset_name), None)
            if not match:
                return (gr.update(),) * len(all_sc_inputs)
            return _scenario_values(match, f"{match['id']}_{int(time.time()) % 1000}", f"{match['name']} (Custom)")

        preset_dropdown.change(fn=on_select_preset_template, inputs=[preset_dropdown], outputs=all_sc_inputs)

        def _library_choices():
            return [f"{s['name']} ({s['id']})" for s in sc_mgr.get_all_scenarios()]

        def on_save_custom_scenario(*args):
            sc = assemble_scenario_payload(*args)
            if not sc["id"]:
                return "The scenario needs an ID.", gr.Dropdown()
            sc_mgr.save_scenario(sc)
            return f"Saved '{sc['name']}'.", gr.Dropdown(choices=_library_choices(), value=f"{sc['name']} ({sc['id']})")

        save_scenario_btn.click(fn=on_save_custom_scenario, inputs=all_sc_inputs, outputs=[scenario_action_msg, load_scenario_dropdown])

        def on_new_scenario_form():
            return (f"custom_drill_{int(time.time()) % 10000}", "New drill", True, "",
                    0.0, -2500.0, 17.0, 90.0, 500.0, 50.0, 0.0, 0.0, 93.15, 0.0, 0.0, 0.0,
                    "goalie", 50.0, 0.0, 4800.0, -90.0, 80.0, 60.0, True, "Form cleared.")

        new_scenario_btn.click(fn=on_new_scenario_form, outputs=all_sc_inputs + [scenario_action_msg])

        def on_delete_custom_scenario(sc_id):
            if not sc_id:
                return "No scenario selected.", gr.Dropdown()
            ok = sc_mgr.delete_scenario(str(sc_id).strip())
            choices = _library_choices()
            return (f"Deleted '{sc_id}'." if ok else f"Could not delete '{sc_id}'."), gr.Dropdown(choices=choices, value=choices[0] if choices else None)

        delete_scenario_btn.click(fn=on_delete_custom_scenario, inputs=[sc_id_input], outputs=[scenario_action_msg, load_scenario_dropdown])

        def on_load_scenario_from_library(selected_choice):
            sc_id = str(selected_choice or "").split("(")[-1].rstrip(")").strip()
            match = sc_mgr.get_scenario(sc_id) if sc_id else None
            return _scenario_values(match) if match else (gr.update(),) * len(all_sc_inputs)

        load_scenario_btn.click(fn=on_load_scenario_from_library, inputs=[load_scenario_dropdown], outputs=all_sc_inputs)

        def on_run_scenario_simulation(*args):
            active = "checkpoints/latest_model.pt" if os.path.exists("checkpoints/latest_model.pt") else None
            res = simulate_custom_scenario(assemble_scenario_payload(*args), model_path=active, num_steps=150)
            if isinstance(res, dict):
                return res.get("plot"), res.get("stats")
            if isinstance(res, (tuple, list)):
                return res[0], res[1]
            return None, {}

        sim_scenario_btn.click(fn=on_run_scenario_simulation, inputs=all_sc_inputs, outputs=[sc_sim_plot, sc_sim_stats])

        # =========================================================================================
        # REFRESH TIMER: status, curves, console, reward card, league. Each part is only re-rendered
        # when its source file changed, so an idle page costs nothing.
        # =========================================================================================
        _last_history = [None]
        _last_log_str = [""]
        _last_view_mode = ["run"]
        _last_league = [None]

        def _mtime_size(path):
            try:
                return (os.path.getmtime(path), os.path.getsize(path))
            except OSError:
                return (0.0, 0)

        def on_timer_tick(view_mode: str = "run"):
            status = mgr.get_status_info()
            card = build_status_card_html(status)
            buttons = _button_updates(status)

            logs = mgr.get_logs()
            logs_update = gr.skip() if logs == _last_log_str[0] else logs
            _last_log_str[0] = logs

            hist = _mtime_size("logs/history.jsonl") + (view_mode,)
            if hist == _last_history[0]:
                plot_update, reward_update = gr.skip(), gr.skip()
            else:
                _last_history[0] = hist
                _last_view_mode[0] = view_mode
                plot_update = render_training_curves_plot(mode=view_mode)
                reward_update = build_reward_card_html(status.get("metrics"))

            league = _mtime_size("logs/trueskill_leaderboard.json") + _mtime_size("logs/league_state.json")
            if league == _last_league[0]:
                lb = (gr.skip(), gr.skip(), gr.skip())
            else:
                _last_league[0] = league
                ts_evaluator.load_leaderboard()
                lb = (build_cockpit_leaderboard_summary_html(ts_evaluator), build_league_wire_and_queue_html(ts_evaluator),
                      get_cockpit_leaderboard_df(ts_evaluator))
            return (card,) + buttons + (logs_update, plot_update, reward_update) + lb

        timer_outputs = [status_card, start_btn, pause_btn, stop_btn, console_output, live_metrics_plot, reward_card,
                         cockpit_lb_summary, cockpit_league_ticker, cockpit_lb_table]
        status_timer = gr.Timer(3.0, active=True)
        status_timer.tick(fn=on_timer_tick, inputs=[metrics_window_radio], outputs=timer_outputs, show_progress="hidden")
        demo.load(fn=on_timer_tick, inputs=[metrics_window_radio], outputs=timer_outputs)
        demo.load(fn=on_show_eval, inputs=eval_view_inputs, outputs=eval_view_outputs)
        demo.load(fn=on_refresh_behaviour, inputs=[diag_window], outputs=[diag_flags, diag_behaviour_plot])

    return demo
