"""
Rocket League ML Bot - Comprehensive Gradio Management Dashboard.
Provides real-time training controls, dynamic reward tuning, live metric charts, console stream, and match replay visualizer.
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
from typing import Union

from utils.process_manager import TrainingProcessManager
from utils.visualizer import simulate_match
from utils.replay_parser import ReplayParser, DEFAULT_DEMO_DIR, get_default_demo_dir
from agent.pretrainer import BehavioralCloningTrainer
from utils.test_runner import run_all_unit_tests, get_cached_or_run_tests, format_test_results_markdown
from utils.diagnostics import (
    extract_rolling_telemetry,
    render_action_biases_plot,
    render_positional_biases_plot,
    generate_ai_coach_diagnostics,
    render_training_curves_plot,
)
from utils.scenario_manager import (
    ScenarioManager,
    render_scenario_visual_guide,
    simulate_custom_scenario,
    DEFAULT_CUSTOM_SCENARIOS
)
from utils.trueskill_evaluator import TrueSkillEvaluator, get_model_display_name
from utils.league_manager import snap_tiers_to_worker_slices


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
   LEAGUE BOARD  --  King banner, Elite standings, Gauntlet ticker.
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

/* ---- Elite standings table ------------------------------------------ */

.lb-standings { padding: 4px 8px 10px; }

.lb-row {
    display: grid;
    grid-template-columns: 38px minmax(140px, 2fr) 92px 1.3fr 56px 54px 58px 116px 48px;
    align-items: center; gap: 9px;
    padding: 7px 10px; border-radius: 7px;
    font-size: 0.86em; color: #cbd5e1;
    border: 1px solid transparent;
    transition: background-color 0.16s ease, border-color 0.16s ease;
}

.lb-row + .lb-row { margin-top: 2px; }
.lb-row:not(.lb-row-head):hover { background: rgba(56, 189, 248, 0.06); border-color: rgba(56, 189, 248, 0.22); }

.lb-row-head {
    font-size: 0.68em; font-weight: 800; letter-spacing: 1px;
    text-transform: uppercase; color: #64748b;
    border-bottom: 1px solid rgba(51, 65, 85, 0.5); border-radius: 0;
    padding-bottom: 6px; margin-bottom: 3px;
}

.lb-row-king { background: rgba(234, 179, 8, 0.07); border-color: rgba(234, 179, 8, 0.32); }
.lb-row-king:hover { background: rgba(234, 179, 8, 0.11); border-color: rgba(234, 179, 8, 0.45); }

.lb-rank {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.95em; font-weight: 800; color: #64748b; font-variant-numeric: tabular-nums;
}
.lb-rank-top { color: #facc15; }

.lb-name { font-weight: 700; color: #f1f5f9; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lb-num  { font-variant-numeric: tabular-nums; font-weight: 700; }
.lb-muted { color: #7c8ba1; font-variant-numeric: tabular-nums; }
.lb-record { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.92em; color: #94a3b8; }

/* Rating bar: width is set inline once per render, then never animated. */
.lb-bar-track {
    position: relative; height: 6px; border-radius: 3px;
    background: rgba(51, 65, 85, 0.55); overflow: hidden;
}
.lb-bar-fill {
    position: absolute; inset: 0 auto 0 0; border-radius: 3px;
    background: linear-gradient(90deg, #0ea5e9 0%, #38bdf8 100%);
    transform-origin: left center;
    animation: lb-bar-grow 0.5s cubic-bezier(0.22, 1, 0.36, 1) both;
}
.lb-bar-fill.is-king { background: linear-gradient(90deg, #ca8a04 0%, #facc15 100%); }
.lb-bar-fill.is-provisional { background: linear-gradient(90deg, #475569 0%, #94a3b8 100%); }

@keyframes lb-bar-grow { from { transform: scaleX(0); } to { transform: scaleX(1); } }

/* ---- Gauntlet ticker ------------------------------------------------ */

.lb-ticker { position: relative; overflow: hidden; padding: 0; }

.lb-ticker-head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px; flex-wrap: wrap; padding: 8px 16px;
    border-bottom: 1px solid rgba(51, 65, 85, 0.55);
}

.lb-live {
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 0.72em; font-weight: 800; letter-spacing: 1.2px; text-transform: uppercase;
    color: #fb7185; background: rgba(244, 63, 94, 0.1);
    border: 1px solid rgba(244, 63, 94, 0.35); border-radius: 9999px; padding: 3px 10px;
}

.lb-live-dot {
    width: 7px; height: 7px; border-radius: 50%; background: #fb7185;
    animation: lb-pulse 1.9s ease-in-out infinite;
}

@keyframes lb-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%      { opacity: 0.35; transform: scale(0.78); }
}

.lb-ticker-viewport {
    position: relative; overflow: hidden; padding: 9px 0;
    -webkit-mask-image: linear-gradient(90deg, transparent 0, #000 44px, #000 calc(100% - 44px), transparent 100%);
            mask-image: linear-gradient(90deg, transparent 0, #000 44px, #000 calc(100% - 44px), transparent 100%);
}

/* Two identical halves scrolled by one transform gives a seamless loop with a
   single animated element. Duration is set inline from the item count so the
   speed stays constant regardless of how much is on the wire. */
.lb-ticker-track {
    display: flex; align-items: center; gap: 10px; width: max-content;
    animation: lb-marquee linear infinite;
    will-change: transform;
}
.lb-ticker-viewport:hover .lb-ticker-track { animation-play-state: paused; }

@keyframes lb-marquee { from { transform: translateX(0); } to { transform: translateX(-50%); } }

.lb-tick {
    display: inline-flex; align-items: center; gap: 8px; flex-shrink: 0;
    background: rgba(15, 23, 42, 0.85); border: 1px solid #24324a;
    border-radius: 8px; padding: 5px 12px; font-size: 0.83em; color: #94a3b8;
}
.lb-tick b { color: #f1f5f9; font-weight: 700; }
.lb-tick-time { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.88em; color: #52627a; }

.lb-tick-badge {
    font-size: 0.76em; font-weight: 800; letter-spacing: 0.6px; text-transform: uppercase;
    padding: 2px 7px; border-radius: 4px; white-space: nowrap;
}
.lb-badge-promotion  { background: rgba(34, 197, 94, 0.16);  color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.45); }
.lb-badge-demotion   { background: rgba(244, 63, 94, 0.16);  color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.45); }
.lb-badge-coronation { background: rgba(234, 179, 8, 0.16);  color: #facc15; border: 1px solid rgba(234, 179, 8, 0.45); }
.lb-badge-admission  { background: rgba(56, 189, 248, 0.16); color: #7dd3fc; border: 1px solid rgba(56, 189, 248, 0.45); }
.lb-badge-preemption { background: rgba(168, 85, 247, 0.16); color: #c4b5fd; border: 1px solid rgba(168, 85, 247, 0.45); }

/* A demotion drops in and settles; a promotion or coronation lifts. Both run
   once, on the newest few items only, so a refresh does not re-animate the
   whole wire. */
.lb-tick-fresh.lb-tick-down { animation: lb-drop 0.55s cubic-bezier(0.34, 1.56, 0.64, 1) both; }
.lb-tick-fresh.lb-tick-up   { animation: lb-rise 0.55s cubic-bezier(0.34, 1.56, 0.64, 1) both; }
.lb-tick-fresh.lb-tick-flat { animation: lb-fade 0.45s ease-out both; }

@keyframes lb-drop { from { transform: translateY(-9px); opacity: 0; } to { transform: none; opacity: 1; } }
@keyframes lb-rise { from { transform: translateY(9px);  opacity: 0; } to { transform: none; opacity: 1; } }
@keyframes lb-fade { from { opacity: 0; } to { opacity: 1; } }

/* ---- Gauntlet trial cards ------------------------------------------- */

.lb-trials { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 10px; padding: 12px 16px; }

.lb-trial {
    background: rgba(10, 15, 30, 0.7); border: 1px solid #24324a;
    border-radius: 9px; padding: 11px 14px;
    display: flex; flex-direction: column; gap: 8px;
    transition: border-color 0.18s ease, transform 0.18s ease;
}
.lb-trial:hover { transform: translateY(-2px); border-color: rgba(56, 189, 248, 0.45); }
.lb-trial-danger { border-color: rgba(244, 63, 94, 0.45); }

.lb-trial-top { display: flex; align-items: center; justify-content: space-between; gap: 8px; }
.lb-trial-name { font-weight: 800; color: #f1f5f9; font-size: 0.93em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

.lb-prog-track { height: 7px; border-radius: 4px; background: rgba(51, 65, 85, 0.6); overflow: hidden; }
.lb-prog-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #0ea5e9 0%, #4ade80 100%);
    transform-origin: left center;
    animation: lb-bar-grow 0.55s cubic-bezier(0.22, 1, 0.36, 1) both;
}

.lb-trial-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 3px 12px; font-size: 0.81em; color: #8fa0b6; }
.lb-trial-grid b { color: #e2e8f0; font-variant-numeric: tabular-nums; }
.lb-trial-foot {
    display: flex; align-items: center; justify-content: space-between;
    font-size: 0.79em; color: #7c8ba1;
    border-top: 1px solid rgba(51, 65, 85, 0.45); padding-top: 6px;
}

.lb-empty {
    padding: 16px 20px; text-align: center; color: #7c8ba1; font-size: 0.87em;
    border: 1px dashed #2b3a52; border-radius: 8px; margin: 12px 16px;
}

/* ---- Elite Pool / Benchmark flip ------------------------------------
   Two faces of one panel. Two radios rather than one checkbox, so clicking the
   tab you are already on is a no-op instead of flipping away. Sibling selectors
   rather than :has() so this holds on the older webviews Gradio embeds. The flip
   is pure CSS, so it costs no server round trip; a re-render returns the card to
   the roster face, which happens only when the leaderboard or league state
   changes on disk. */

.lb-flip-input { position: absolute; opacity: 0; pointer-events: none; width: 0; height: 0; }

.lb-flip-tabs { display: inline-flex; gap: 2px; padding: 2px; border-radius: 7px;
    background: rgba(15, 23, 42, 0.75); border: 1px solid #22304a; }

.lb-flip-tab {
    cursor: pointer; user-select: none;
    font-size: 0.72em; font-weight: 700; letter-spacing: 0.6px; text-transform: uppercase;
    padding: 4px 10px; border-radius: 5px; color: #7c8ba1;
    transition: background 0.18s ease, color 0.18s ease;
}
.lb-flip-tab:hover { color: #cbd5e1; }

.lb-flip-face { display: none; }
#lbface-pool:checked  ~ .lb-face-pool  { display: block; }
#lbface-chart:checked ~ .lb-face-chart { display: block; }
#lbface-pool:checked  ~ .lb-panel-head .lb-flip-tab-pool,
#lbface-chart:checked ~ .lb-panel-head .lb-flip-tab-chart {
    background: rgba(56, 189, 248, 0.16); color: #7dd3fc;
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

@media (max-width: 900px) {
    /* Drops the sigma, record and games columns; the remaining six keep their order,
       so the header and the body rows stay aligned at every width. */
    .lb-row { grid-template-columns: 34px minmax(110px, 2fr) 88px 1.1fr 54px 56px; }
    .lb-row > .lb-hide-sm { display: none; }
}

@media (prefers-reduced-motion: reduce) {
    .lb-king::after, .lb-crown, .lb-live-dot, .lb-ticker-track,
    .lb-bar-fill, .lb-prog-fill, .lb-tick-fresh { animation: none !important; }
    .lb-bar-fill, .lb-prog-fill { transform: none !important; }
    .lb-trial:hover { transform: none; }
}

"""


def format_elapsed_time(seconds: Union[int, float]) -> str:
    sec = int(seconds or 0)
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}h {m:02d}m {s:02d}s"
    return f"{m:02d}m {s:02d}s"


def build_status_card_html(status_info: dict, feedback_msg: str = "") -> str:
    running = status_info.get("running", False)
    paused = status_info.get("paused", False)
    pid = status_info.get("pid")
    elapsed = status_info.get("elapsed_seconds", 0)
    metrics = status_info.get("metrics", {})

    elapsed_str = format_elapsed_time(elapsed)
    iter_num = metrics.get("iteration", 0)
    step_num = metrics.get("global_step", 0)
    sps = metrics.get("sps", 0)
    rew = metrics.get("mean_reward", 0.0)

    if running and not paused:
        badge = '<span class="status-badge-running">● RUNNING</span>'
    elif running and paused:
        badge = '<span class="status-badge-paused">❚❚ PAUSED</span>'
    else:
        badge = '<span class="status-badge-stopped">○ STOPPED</span>'

    html = f"""
    <div class="hero-status-card">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div style="display: flex; align-items: center; gap: 16px;">
                {badge}
                <span style="color: #94a3b8; font-size: 0.95em;">PID: <b style="color: #f1f5f9;">{pid if pid else 'None'}</b></span>
                <span style="color: #94a3b8; font-size: 0.95em;">Elapsed: <b style="color: #f1f5f9;">{elapsed_str}</b></span>
            </div>
            <div style="display: flex; align-items: center; gap: 20px; color: #cbd5e1; font-size: 0.95em;">
                <span>Iteration: <b style="color: #38bdf8;">{iter_num:,}</b></span>
                <span>Global Steps: <b style="color: #818cf8;">{step_num:,}</b></span>
                <span>Speed: <b style="color: #34d399;">{sps:,} SPS</b></span>
                <span>Mean Reward: <b style="color: {'#4ade80' if rew >= 0 else '#f87171'};">{rew:+.2f}</b></span>
            </div>
        </div>
        {f'<div style="margin-top: 10px; padding-top: 8px; border-top: 1px solid #334155; color: #60a5fa; font-size: 0.9em;">{feedback_msg}</div>' if feedback_msg else ''}
    </div>
    """
    return html


def build_full_diagnostic_export() -> tuple[str, str]:
    """
    Assembles a comprehensive, single-source-of-truth diagnostic summary of the entire bot training system.
    Returns:
        (formatted_overview_markdown, copy_paste_export_string)
    """
    mgr = TrainingProcessManager.get_instance()
    status = mgr.get_status_info()
    running = status.get("running", False)
    paused = status.get("paused", False)
    pid = status.get("pid", "None")
    elapsed = status.get("elapsed_seconds", 0)
    elapsed_str = format_elapsed_time(elapsed)
    metrics = status.get("metrics", {})

    # 1. System & Physics Engine
    try:
        from env.physics_engine import ROCKETSIM_AVAILABLE
    except Exception:
        ROCKETSIM_AVAILABLE = False
    
    engine_str = "C++ RocketSim (High Speed Bullet Physics ~4000+ SPS)" if ROCKETSIM_AVAILABLE else "Pure-Python Fallback (~1100 SPS)"

    # 2. Hyperparameters & Environment Config
    default_cfg = load_yaml_config("config/default_config.yaml")
    hp = default_cfg.get("hyperparameters", {})
    env = default_cfg.get("environment", {})
    rew = default_cfg.get("rewards", {})
    
    # Overlay live config
    if os.path.exists("config/live_config.json"):
        try:
            with open("config/live_config.json", "r") as f:
                ld = json.load(f)
                if "rewards" in ld and isinstance(ld["rewards"], dict):
                    rew.update(ld["rewards"])
                if "learning_rate" in ld:
                    hp["learning_rate"] = ld["learning_rate"]
                if "ent_coef" in ld:
                    hp["ent_coef"] = ld["ent_coef"]
                if "clip_range" in ld:
                    hp["clip_range"] = ld["clip_range"]
                if "baseline_opponent_type" in ld:
                    env["baseline_opponent_type"] = ld["baseline_opponent_type"]
                if "baseline_opponent_ratio" in ld:
                    env["baseline_opponent_ratio"] = ld["baseline_opponent_ratio"]
        except Exception:
            pass

    # What the environments actually face, which is the league split rather than the
    # legacy baseline_opponent_* keys the trainer ignores while the league is running.
    opponent_mix = describe_opponent_mix(
        default_cfg.get("league", {}) or {},
        metrics.get("league", {}) or {},
        fallback=str(env.get("baseline_opponent_type", "heuristic")),
        num_envs=int(env.get("num_envs", 0) or 0),
    )

    # 3. Model Architecture & Weights
    pts = get_available_checkpoints()
    latest_ckpt = pts[0] if pts else "None"
    model_details = "None loaded"
    if latest_ckpt != "None" and os.path.exists(latest_ckpt):
        try:
            sz = os.path.getsize(latest_ckpt) / (1024 * 1024)
            model_details = f"`{latest_ckpt}` ({sz:.2f} MB)"
        except Exception:
            model_details = f"`{latest_ckpt}`"

    # 4. Unit Tests Status
    test_results = get_cached_or_run_tests(force_refresh=False)
    tests_summary = f"{test_results.get('passed', 0)}/{test_results.get('total_tests', test_results.get('total', 0))} Passed ({'ALL PASSING' if test_results.get('all_passed') else 'FAILURES DETECTED'})"

    # 5. Telemetry & Coach Analysis
    telem = extract_rolling_telemetry("logs/history.jsonl", window=10)
    coach_report = generate_ai_coach_diagnostics(telem, active_rewards=rew)

    # 6. Recent Logs
    recent_logs = mgr.get_logs(max_lines=30)

    # Build Markdown Export
    export_text = f"""# 🏎️ SensAI Training & System State Snapshot
**Timestamp:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Process Status:** {'RUNNING' if running and not paused else ('PAUSED' if paused else 'STOPPED')} (PID: {pid}, Elapsed: {elapsed_str})

## 1. System Health & Unit Tests
* **Physics Engine:** {engine_str}
* **Unit Tests Status:** {tests_summary}
* **Active Model Weights:** {model_details}

## 2. Live Training Metrics (Current Rollout)
* **Iteration:** {metrics.get('iteration', 0):,}
* **Global Steps:** {metrics.get('global_step', 0):,}
* **Throughput:** {metrics.get('sps', 0):,} Steps/Sec
* **Mean Reward:** {metrics.get('mean_reward', 0.0):+.3f}
* **Policy Loss:** {metrics.get('policy_loss', 0.0):.4f} | **Value Loss:** {metrics.get('value_loss', 0.0):.4f} | **Entropy:** {metrics.get('entropy', 0.0):.4f}

## 3. Active Hyperparameters & Opponent Mix
* **Learning Rate:** `{hp.get('learning_rate', 3e-4)}` | **Entropy Coef:** `{hp.get('ent_coef', 0.005)}` | **Clip Range:** `{hp.get('clip_range', 0.2)}`
* **Gamma:** `{hp.get('gamma', 0.99)}` | **GAE Lambda:** `{hp.get('gae_lambda', 0.95)}`
* **Batch Size:** `{hp.get('batch_size', 8192)}` | **Mini-Batch Size:** `{hp.get('mini_batch_size', 512)}` | **Epochs:** `{hp.get('n_epochs', 10)}`
* **Vectorized Envs:** `{env.get('num_envs', 64)}` | **Tick Skip:** `{env.get('tick_skip', 8)}`
* **Opponent Mix:** {opponent_mix}

## 4. Active Reward Weights
```yaml
{yaml.dump(rew, default_flow_style=False).strip()}
```

## 5. AI Coach Behavioral Diagnosis
{coach_report}

## 6. Recent Process Output (Last 30 Lines)
```text
{recent_logs}
```
"""

    overview_md = f"""
### 📋 System Health Overview
* **Status:** `{'RUNNING' if running and not paused else ('PAUSED' if paused else 'STOPPED')}` (PID: `{pid}`)
* **Throughput:** `{metrics.get('sps', 0):,} SPS` | **Mean Reward:** `{metrics.get('mean_reward', 0.0):+.2f}`
* **Unit Tests:** `{tests_summary}`
* **Physics Engine:** `{engine_str}`
* **Active Weights:** {model_details}
* **Opponent Mix:** {opponent_mix}

*Copy the raw Markdown on the right into your conversation with the AI assistant for instant debugging.*
"""
    return overview_md, export_text


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
    fallback: str = "heuristic",
    num_envs: Optional[int] = None,
) -> str:
    """
    One line naming what the training environments are really facing.

    Not the legacy baseline_opponent_* pair. Those two keys only take effect when the
    league is disabled, and the trainer ignores them otherwise, so quoting them while the
    league runs reports a matchup nobody is playing. This panel advertised Nexto at 5%
    through an entire session in which no environment faced Nexto at all, which is worse
    than saying nothing while the opponent mix is the thing under investigation.
    """
    if not league.get("enabled", True):
        return f"League disabled. Static baseline: `{fallback}`"

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
    cfg = (load_yaml_config("config/default_config.yaml") or {})
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
    lock_at = league.get("rating_lock_matches", 64)

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
        ("6", "Locked",
         f"At {lock_at} series its rating stops moving for good. It keeps playing as an "
         "opponent and as a yardstick, but a long reign would otherwise re-score it on "
         "whoever happened to challenge it next."),
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
    """
    A compact legend for the calibrated reference ladder.

    These sit outside the standings because their mu is a declaration, not a result:
    they never move, they cannot be dethroned, and listing them among the checkpoints
    makes a fixed yardstick look like a competitor.
    """
    anchors = evaluator.get_anchor_ratings() if hasattr(evaluator, "get_anchor_ratings") else []
    if not anchors:
        # No ladder to show yet, but the explainer is still worth having.
        return f'<div class="lb-anchor-legend">{build_how_it_works_html()}</div>'

    chips = []
    for rec in anchors:
        chips.append(
            f'<span class="lb-anchor-chip">'
            f'<b>{rec.name}</b>'
            f'<span class="lb-anchor-mu">&mu; {rec.mu:.1f}</span>'
            f'</span>'
        )
    return f"""
    <div class="lb-anchor-legend">
        {build_how_it_works_html()}
        <span class="lb-anchor-legend-label">&#9875; Reference ladder</span>
        {''.join(chips)}
        <span class="lb-anchor-note">fixed calibration &middot; excluded from standings</span>
    </div>
    """


def build_cockpit_leaderboard_summary_html(evaluator: TrueSkillEvaluator, league_state: Optional[Dict[str, Any]] = None) -> str:
    """
    King banner plus season totals: the headline slab of the league board.

    Leads with mu and sigma rather than the conservative score, because ranking is now
    gated on sigma and ordered by mu; showing mu - 3*sigma as the headline number would
    describe a ranking the league no longer uses.
    """
    global _LB_EVALUATOR_GATE
    _LB_EVALUATOR_GATE = evaluator.is_rank_eligible

    ratings = [
        r for r in evaluator.ratings.values()
        if "latest_model" not in r.path.lower() and "latest_model" not in r.name.lower()
    ]
    if not ratings:
        return """
        <div class="lb-panel" style="padding: 14px 20px; color: #7c8ba1; font-size: 0.88em; margin-bottom: 10px;">
            &#8505;&#65039; <b>League standby.</b> No checkpoints graded yet. Each numbered checkpoint is
            graded automatically on save (see League Checkpoint Interval).
        </div>
        """

    state = league_state if league_state is not None else load_league_state_safely()
    active_king_path = state.get("king_of_the_hill")

    sorted_ratings = sorted(ratings, key=lambda r: (evaluator.ranking_key(r), r.matches_played), reverse=True)

    king = None
    if active_king_path:
        norm_target = str(active_king_path).replace("\\", "/").lower()
        for r in ratings:
            if r.path.replace("\\", "/").lower() == norm_target or r.name.lower() in norm_target:
                king = r
                break
    # An anchor is never King. The league bars it, and the banner must agree: falling
    # back to "the best rated thing" here crowned Nexto on its declared mu and made a
    # fixed yardstick look like the reigning champion.
    if king is not None and king.is_anchor:
        king = None

    ckpts = [r for r in sorted_ratings if not r.is_anchor and _lb_iteration(r) >= 0]
    latest_iter = max((_lb_iteration(r) for r in ckpts), default=-1)
    ranked_count = sum(1 for r in ratings if evaluator.is_rank_eligible(r))
    total_matches = sum(r.matches_played for r in ratings) // 2

    if king is None:
        # Cold start: no rated checkpoint, so the King and pool tiers fall back to pure
        # self-play. Say that plainly rather than showing an empty crown.
        return f"""
    <div class="lb-king lb-king-empty">
        <div class="lb-king-id">
            <span class="lb-crown lb-crown-dim">&#9876;&#65039;</span>
            <div style="min-width: 0;">
                <div class="lb-king-label">Throne Vacant</div>
                <div class="lb-king-name">Awaiting a rated checkpoint</div>
                <div class="lb-king-sub">
                    Anchors cannot hold the crown &middot; King and pool tiers are running
                    as self-play until the first checkpoints are graded
                </div>
            </div>
        </div>
        <div class="lb-king-stats">
            <div class="lb-stat">
                <span class="lb-stat-label">Latest Ckpt</span>
                <span class="lb-stat-value accent">{latest_iter if latest_iter >= 0 else '&mdash;'}</span>
            </div>
            <div class="lb-stat">
                <span class="lb-stat-label">Roster</span>
                <span class="lb-stat-value">{ranked_count}<span style="color:#64748b; font-size:0.7em; font-weight:600;"> ranked / {len(ratings)}</span></span>
            </div>
            <div class="lb-stat">
                <span class="lb-stat-label">Matches</span>
                <span class="lb-stat-value">{total_matches:,}</span>
            </div>
        </div>
    </div>
    """

    king_chip, _ = _lb_confidence(king)
    king_iter = _lb_iteration(king)
    king_sub = f"checkpoint_iter_{king_iter}" if king_iter >= 0 else king.path

    return f"""
    <div class="lb-king">
        <div class="lb-king-id">
            <span class="lb-crown">&#128081;</span>
            <div style="min-width: 0;">
                <div class="lb-king-label">King of the Hill</div>
                <div class="lb-king-name">{_lb_short_name(king)} {king_chip}</div>
                <div class="lb-king-sub">{king_sub} &middot; {king.wins}W-{king.losses}L-{king.draws}D over {king.matches_played} matches</div>
            </div>
        </div>
        <div class="lb-king-stats">
            <div class="lb-stat">
                <span class="lb-stat-label">Rating</span>
                <span class="lb-stat-value warn">{king.mu:.2f}</span>
            </div>
            <div class="lb-stat">
                <span class="lb-stat-label">Uncertainty</span>
                <span class="lb-stat-value">&plusmn;{king.sigma:.2f}</span>
            </div>
            <div class="lb-stat">
                <span class="lb-stat-label">Points</span>
                <span class="lb-stat-value good">{king.points_rate:.1f}%</span>
            </div>
            <div class="lb-stat">
                <span class="lb-stat-label">Latest Ckpt</span>
                <span class="lb-stat-value accent">{latest_iter if latest_iter >= 0 else '&mdash;'}</span>
            </div>
            <div class="lb-stat">
                <span class="lb-stat-label">Roster</span>
                <span class="lb-stat-value">{ranked_count}<span style="color:#64748b; font-size:0.7em; font-weight:600;"> ranked / {len(ratings)}</span></span>
            </div>
            <div class="lb-stat">
                <span class="lb-stat-label">Matches</span>
                <span class="lb-stat-value">{total_matches:,}</span>
            </div>
        </div>
    </div>
    """


def _build_gauntlet_ticker(state: Dict[str, Any]) -> str:
    """
    The Gauntlet wire: a seamless CSS marquee of promotions, demotions, coronations,
    admissions and preemptions.

    The track holds the item list twice and is translated by exactly -50%, so the loop
    is seamless with a single animated element and no JavaScript. Duration scales with
    item count to hold a constant scroll speed, and the newest three events get a
    one-shot entrance keyed to their direction: demotions drop, promotions rise.
    """
    events = state.get("event_history", [])
    recent = list(reversed(events[-14:]))

    if not recent:
        body = """
        <div class="lb-ticker-viewport">
            <div style="padding: 2px 16px; color: #7c8ba1; font-size: 0.85em;">
                Wire is quiet. Promotions, demotions and coronations broadcast here as trials resolve.
            </div>
        </div>
        """
        return f"""
        <div class="lb-panel lb-ticker">
            <div class="lb-ticker-head">
                <span class="lb-live"><span class="lb-live-dot"></span>Gauntlet Wire</span>
                <span class="lb-panel-meta">Promotion &amp; demotion feed</span>
            </div>
            {body}
        </div>
        """

    badges = {
        "promotion":  ('<span class="lb-tick-badge lb-badge-promotion">&#127942; Promoted</span>', "lb-tick-up"),
        "demotion":   ('<span class="lb-tick-badge lb-badge-demotion">&#128317; Demoted</span>', "lb-tick-down"),
        "coronation": ('<span class="lb-tick-badge lb-badge-coronation">&#128081; New King</span>', "lb-tick-up"),
        "admission":  ('<span class="lb-tick-badge lb-badge-admission">&#9876;&#65039; In Queue</span>', "lb-tick-flat"),
        "preemption": ('<span class="lb-tick-badge lb-badge-preemption">&#128260; Preempt</span>', "lb-tick-flat"),
    }

    ticks = []
    for idx, ev in enumerate(recent):
        badge, direction = badges.get(
            str(ev.get("type", "")).lower(),
            ('<span class="lb-tick-badge lb-badge-admission">&#9889; Update</span>', "lb-tick-flat")
        )
        time_str = ""
        raw_ts = ev.get("timestamp", "")
        if raw_ts:
            try:
                time_str = datetime.datetime.fromisoformat(raw_ts).strftime("%H:%M:%S")
            except Exception:
                time_str = ""
        # Only the newest few animate in; re-animating the whole wire on every poll
        # would be noise, and would restart mid-scroll.
        fresh = " lb-tick-fresh" if idx < 3 else ""
        stamp = f'<span class="lb-tick-time">{time_str}</span>' if time_str else ""
        ticks.append(
            f'<div class="lb-tick{fresh} {direction}">{badge}<b>{ev.get("model", "Model")}</b>'
            f'<span>{ev.get("detail", "")}</span>{stamp}</div>'
        )

    # ~9s of travel per item keeps the pace readable whether the wire holds 3 or 14.
    duration = max(24, len(ticks) * 9)
    lane = "".join(ticks)

    return f"""
    <div class="lb-panel lb-ticker">
        <div class="lb-ticker-head">
            <span class="lb-live"><span class="lb-live-dot"></span>Gauntlet Wire</span>
            <span class="lb-panel-meta">Hover to pause &middot; {len(ticks)} recent events</span>
        </div>
        <div class="lb-ticker-viewport">
            <div class="lb-ticker-track" style="animation-duration: {duration}s;">
                {lane}{lane}
            </div>
        </div>
    </div>
    """


def _build_gauntlet_trials(state: Dict[str, Any], evaluator: TrueSkillEvaluator) -> str:
    """Active contender trial cards, keyed on convergence rather than raw win rate."""
    contenders = state.get("contenders", [])

    if not contenders and evaluator and hasattr(evaluator, "ratings"):
        inferred = []
        for path, rec in evaluator.ratings.items():
            if (
                not rec.is_anchor
                and path != "heuristic"
                and "latest_model" not in path.lower()
                and not evaluator.is_rank_eligible(rec)
                and rec.mu >= 26.0
                and rec.points_rate >= 50.0
            ):
                inferred.append({
                    "name": rec.name, "path": path,
                    "mu": round(rec.mu, 2), "sigma": round(rec.sigma, 2),
                    "conservative_score": round(rec.conservative_rating, 2),
                    "matches_played": rec.matches_played,
                    "target_matches": evaluator.min_ranked_matches,
                    "progress_pct": min(100.0, round((rec.matches_played / max(1, evaluator.min_ranked_matches)) * 100.0, 1)),
                    "win_rate": round(rec.win_rate, 1), "points_rate": rec.points_rate,
                    "record": f"{rec.wins}W-{rec.losses}L-{rec.draws}D",
                    "consecutive_losses": 0, "max_consecutive_losses": 4,
                    "status": "In Trial",
                })
        contenders = inferred[:3]

    gate = getattr(evaluator, "eligibility_sigma", 1.5)

    if not contenders:
        inner = (
            '<div class="lb-empty">&#128564; <b>No active trials.</b> '
            'A new checkpoint entering at &mu; &ge; 26.0 with Points &ge; 50% is admitted automatically.</div>'
        )
        count_str = '<span class="lb-panel-meta">Queue idle</span>'
    else:
        cards = []
        for c in contenders:
            consec = c.get("consecutive_losses", 0)
            max_consec = c.get("max_consecutive_losses", 4)
            danger = " lb-trial-danger" if consec >= max(1, max_consec - 1) else ""
            streak_color = "#4ade80" if consec == 0 else ("#facc15" if consec < max_consec - 1 else "#fb7185")
            sigma = c.get("sigma", 8.33)
            pts = c.get("points_rate", c.get("win_rate", 0.0))
            # Distance to the eligibility gate is the thing that actually decides whether
            # this contender can ever be ranked, so it is the headline on the card.
            sigma_color = "#4ade80" if sigma <= gate else ("#facc15" if sigma <= gate * 1.6 else "#fb7185")
            pct = c.get("progress_pct", 0.0)
            cards.append(f"""
            <div class="lb-trial{danger}">
                <div class="lb-trial-top">
                    <span class="lb-trial-name">&#9889; {c.get('name', 'Contender')}</span>
                    <span class="lb-chip lb-chip-provisional">{c.get('status', 'In Trial')}</span>
                </div>
                <div>
                    <div style="display:flex; justify-content:space-between; font-size:0.79em; color:#7c8ba1; margin-bottom:4px;">
                        <span>Trial progress</span>
                        <b style="color:#38bdf8;">{c.get('matches_played', 0)} / {c.get('target_matches', 24)} games</b>
                    </div>
                    <div class="lb-prog-track"><div class="lb-prog-fill" style="width: {pct}%;"></div></div>
                </div>
                <div class="lb-trial-grid">
                    <div>Rating <b>&mu;={c.get('mu', 25.0):.2f}</b></div>
                    <div>Uncertainty <b style="color:{sigma_color};">&sigma;=&plusmn;{sigma:.2f}</b></div>
                    <div>Points <b style="color:#4ade80;">{pts:.1f}%</b></div>
                    <div>Gate <b class="lb-muted">&sigma; &le; {gate:.1f}</b></div>
                </div>
                <div class="lb-trial-foot">
                    <span>Record <b style="color:#cbd5e1;">{c.get('record', '0W-0L-0D')}</b></span>
                    <span>Loss streak <b style="color:{streak_color};">{consec} / {max_consec}</b></span>
                </div>
            </div>
            """)
        inner = f'<div class="lb-trials">{"".join(cards)}</div>'
        count_str = f'<span class="lb-panel-meta"><b>{len(contenders)}</b> in trial</span>'

    return f"""
    <div class="lb-panel">
        <div class="lb-panel-head">
            <span class="lb-panel-title">&#9876;&#65039; Gauntlet Trials</span>
            {count_str}
        </div>
        {inner}
    </div>
    """


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
    Goal margin per episode against each fixed reference, over training iteration.

    Why margin rather than series wins: series win rate against these references is
    pinned at the rails and has no gradient left. Checkpoints take 100% of series off
    the heuristic and the BC baseline, and won 9 of 96 off Necto across iterations
    160000-190200 both before and after a real improvement. The margin over that same
    span moved -0.64 to -0.25, a 4.7 sigma change the series record shows as flat.

    Why one line per reference and never a combined score: the reference set holds a
    closed cycle. Nexto beats Necto every series, loses to every checkpoint measured,
    and Necto beats those same checkpoints. No scalar spans that, so averaging the
    columns would invent a transitivity that does not exist. The chart plots each
    reference on its own and says so underneath.
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
            Fixed references play on the <code>benchmark_interval</code> schedule and change no
            rating. They are the only measure that compares across checkpoints, because the
            ladder is pool-relative and cannot see a gain the whole field shares.
        </div>
        """

    history.sort(key=lambda e: e["iteration"])

    # Series order is fixed by first appearance, so a colour never migrates between
    # references as readings accumulate.
    names: List[str] = []
    for e in history:
        for name in e["results"]:
            if name not in names:
                names.append(name)

    points: Dict[str, List[Tuple[int, float, Dict[str, Any]]]] = {n: [] for n in names}
    for e in history:
        for name, res in e["results"].items():
            eps = int(res.get("episodes", 0) or 0)
            if eps <= 0:
                continue  # written before the episode count was recorded
            gf = int(res.get("goals_for", 0) or 0)
            ga = int(res.get("goals_against", 0) or 0)
            points[name].append((int(e["iteration"]), (gf - ga) / eps, res))
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

    tick_iters = (x_lo, (x_lo + x_hi) // 2, x_hi) if x_hi > x_lo else (x_lo,)
    for it in tick_iters:
        parts.append(f'<text x="{sx(it):.1f}" y="{H - PAD_B + 14:.0f}" text-anchor="middle" '
                     f'class="bm-axis-label">{it:,}</text>')
    mid_x = (PAD_L + W - PAD_R) / 2.0
    mid_y = (PAD_T + H - PAD_B) / 2.0
    parts.append(f'<text x="{mid_x:.0f}" y="{H - 2:.0f}" text-anchor="middle" '
                 f'class="bm-axis-title">TRAINING ITERATION</text>')
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
            tip = (f"{name} &#183; iter {it:,} &#183; margin {m:+.2f}/ep &#183; goals "
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
             aria-label="Goal margin per episode against each fixed reference, by training iteration">
            {''.join(parts)}
        </svg>
    </div>
    <div class="bm-legend"><span class="lb-muted">Latest</span>{''.join(legend)}</div>
    <div class="bm-note">
        Above the zero line the checkpoint outscores the reference. Each reference is its own
        scale and they are <b>never combined</b>: Nexto beats Necto every series, loses to every
        checkpoint, and Necto beats those same checkpoints, so no single ordering holds. Read one
        line for progress and a <b>divergence between them as a style shift</b>. Nothing here
        moves a rating.
    </div>
    """


def _build_elite_standings(state: Dict[str, Any], evaluator: TrueSkillEvaluator) -> str:
    """
    The Elite Pool as a standings table rather than a card grid.

    Rows are dense and aligned so ratings can be compared down a column, which is the
    point of a standings board. The bar is scaled across the visible mu spread so small
    real differences stay visible instead of collapsing against a 0-50 axis.
    """
    items = state.get("elite_pool_details", [])
    king_path = str(state.get("king_of_the_hill", "")).replace("\\", "/").lower()

    if not items:
        paths = state.get("elite_pool", [])
        if not paths and evaluator and hasattr(evaluator, "ratings"):
            valid = []
            for key, rec in evaluator.ratings.items():
                if "latest_model" in key.lower():
                    continue
                if rec.is_anchor or key == "heuristic" or os.path.exists(rec.path):
                    valid.append((key, rec))
            valid.sort(key=lambda x: (evaluator.ranking_key(x[1]), x[1].matches_played), reverse=True)
            paths = [x[0] for x in valid[:10]]
        for rank, path in enumerate(paths, start=1):
            rec = evaluator.ratings.get(path) if evaluator and hasattr(evaluator, "ratings") else None
            if not rec:
                continue
            items.append({
                "rank": rank, "name": get_model_display_name(path), "path": path,
                "mu": round(rec.mu, 2), "sigma": round(rec.sigma, 2),
                "conservative_score": round(rec.conservative_rating, 2),
                "win_rate": round(rec.win_rate, 1), "points_rate": rec.points_rate,
                "record": f"{rec.wins}W-{rec.losses}L-{rec.draws}D",
                "matches_played": rec.matches_played,
                "is_anchor": rec.is_anchor,
                "is_king": str(path).replace("\\", "/").lower() == king_path,
            })

    if not items:
        return """
        <div class="lb-empty">&#128737;&#65039; <b>Pool initialising.</b> Models appear here as checkpoints are graded.</div>
        """

    # At cold start the pool holds nothing but anchors and the league is running pure
    # self-play, so calling it an active sparring roster would be wrong.
    anchors_only = all(m.get("is_anchor") for m in items)
    if anchors_only and not king_path:
        roster_note = (
            "<b>Standby</b> &middot; anchors only, not in rotation &middot; "
            "King and pool tiers are running as self-play until a checkpoint is rated"
        )
    else:
        gate = getattr(evaluator, "eligibility_sigma", 1.5)
        roster_note = (
            f"<b>{len(items)}</b> active &middot; ranked by &mu; behind a &sigma; &le; {gate:.1f} gate "
            "&middot; 50% self-play / 25% king / 25% pool"
        )

    mus = [float(m.get("mu", 25.0)) for m in items]
    lo, hi = min(mus), max(mus)
    span = max(hi - lo, 1.0)  # never divide by zero when the pool has converged tightly

    head = (
        '<div class="lb-row lb-row-head">'
        '<span>#</span><span>Model</span><span>Confidence</span><span>Rating</span>'
        '<span>&mu;</span><span class="lb-hide-sm">&sigma;</span><span>Pts</span>'
        '<span class="lb-hide-sm">Record</span><span class="lb-hide-sm">GP</span></div>'
    )

    rows = []
    for m in items:
        rec = evaluator.ratings.get(m.get("path", "")) if evaluator else None
        is_king = bool(m.get("is_king"))
        is_anchor = bool(m.get("is_anchor"))
        mu = float(m.get("mu", 25.0))
        sigma = float(m.get("sigma", 8.33))

        if is_anchor:
            chip = '<span class="lb-chip lb-chip-anchor">Anchor</span>'
            ranked = True
        elif getattr(rec, "rating_locked", False):
            chip = '<span class="lb-chip lb-chip-locked">&#128274; Locked</span>'
            ranked = True
        elif rec is not None and evaluator.is_rank_eligible(rec):
            chip = '<span class="lb-chip lb-chip-ranked">Ranked</span>'
            ranked = True
        elif rec is None and sigma <= getattr(evaluator, "eligibility_sigma", 1.5):
            chip = '<span class="lb-chip lb-chip-ranked">Ranked</span>'
            ranked = True
        else:
            chip = '<span class="lb-chip lb-chip-provisional">Provisional</span>'
            ranked = False

        if is_king:
            chip = '<span class="lb-chip lb-chip-king">&#128081; King</span>'

        pct = 8.0 + 92.0 * ((mu - lo) / span)
        bar_class = "is-king" if is_king else ("" if ranked else "is-provisional")
        rank_class = " lb-rank-top" if m.get("rank", 99) <= 3 else ""
        row_class = " lb-row-king" if is_king else ""
        pts = float(m.get("points_rate", m.get("win_rate", 0.0)))
        gp = int(m.get("matches_played", 0) or 0)
        # An anchor's mu is a declaration, not something it earned on the field, and a
        # reference with no games would otherwise read as a bright red 0.0%.
        if is_anchor or gp == 0:
            pts_cell = '<span class="lb-muted">&mdash;</span>'
            record_cell = '<span class="lb-muted">reference</span>' if is_anchor else '<span class="lb-muted">&mdash;</span>'
        else:
            pts_color = "#4ade80" if pts >= 50 else ("#facc15" if pts >= 40 else "#fb7185")
            pts_cell = f'<span class="lb-num" style="color: {pts_color};">{pts:.1f}%</span>'
            record_cell = f'<span class="lb-record">{m.get("record", "0W-0L-0D")}</span>'

        rows.append(f"""
        <div class="lb-row{row_class}">
            <span class="lb-rank{rank_class}">{m.get('rank', '-')}</span>
            <span class="lb-name">{m.get('name', 'Model')}</span>
            <span>{chip}</span>
            <span class="lb-bar-track" title="&mu;={mu:.2f} &plusmn;{sigma:.2f}">
                <span class="lb-bar-fill {bar_class}" style="width: {pct:.1f}%;"></span>
            </span>
            <span class="lb-num">{mu:.2f}</span>
            <span class="lb-muted lb-hide-sm">&plusmn;{sigma:.2f}</span>
            {pts_cell}
            <span class="lb-hide-sm">{record_cell}</span>
            <span class="lb-muted lb-hide-sm">{gp if gp else '&mdash;'}</span>
        </div>
        """)

    return f"""
    <div class="lb-face-note">{roster_note}</div>
    <div class="lb-standings">{head}{"".join(rows)}</div>
    """


def _build_pool_panel(state: Dict[str, Any], evaluator: TrueSkillEvaluator) -> str:
    """
    One panel, two faces: the sparring roster and the benchmark curve.

    They share a card because they answer the same question from the two directions the
    league has available, and showing both at once would double the height of the board
    for a reader who only ever wants one of them. The roster is where each model sits
    relative to the others. The curve is where the field sits against a fixed opponent,
    which is the only thing that can see a gain the whole pool shares -- pool-relative
    ratings cannot, and the reward scale moves whenever the reward function is edited.

    The toggle is a pair of radio inputs driving sibling CSS selectors, so switching
    faces is instant and costs no server round trip. A re-render returns the card to the
    roster face, which happens only when the leaderboard or league state file changes.
    """
    roster = _build_elite_standings(state, evaluator)
    chart = _build_benchmark_chart(state)
    readings = len([
        e for e in (state.get("benchmark_history") or [])
        if isinstance(e, dict) and e.get("iteration") is not None
    ])
    count_chip = f' <span class="lb-muted">{readings}</span>' if readings else ""

    return f"""
    <div class="lb-panel">
        <input class="lb-flip-input" type="radio" name="lbface" id="lbface-pool" checked>
        <input class="lb-flip-input" type="radio" name="lbface" id="lbface-chart">
        <div class="lb-panel-head">
            <span class="lb-panel-title">&#128737;&#65039; Elite Pool</span>
            <span class="lb-flip-tabs">
                <label for="lbface-pool" class="lb-flip-tab lb-flip-tab-pool"
                       title="Ratings relative to the rest of the pool">Roster</label>
                <label for="lbface-chart" class="lb-flip-tab lb-flip-tab-chart"
                       title="Goal margin against the fixed references, which no rating uses">Benchmarks{count_chip}</label>
            </span>
        </div>
        <div class="lb-flip-face lb-face-pool">{roster}</div>
        <div class="lb-flip-face lb-face-chart">{chart}</div>
    </div>
    """


def build_league_wire_and_queue_html(evaluator: TrueSkillEvaluator, league_state: Optional[Dict[str, Any]] = None) -> str:
    """Renders the full league board: Gauntlet wire, active trials, Elite Pool roster and benchmarks."""
    state = league_state if league_state is not None else load_league_state_safely()
    return (
        '<div class="league-board">'
        + _build_gauntlet_ticker(state)
        + _build_gauntlet_trials(state, evaluator)
        + _build_pool_panel(state, evaluator)
        + '</div>'
    )


def create_ui():
    mgr = TrainingProcessManager.get_instance()
    bc_trainer = BehavioralCloningTrainer()
    default_cfg = load_yaml_config("config/default_config.yaml")

    hp_cfg = default_cfg.get("hyperparameters", {})
    env_cfg = default_cfg.get("environment", {})
    rew_cfg = default_cfg.get("rewards", {})
    log_cfg = default_cfg.get("logging", {})
    sc_cfg = default_cfg.get("scenarios", {})

    # Overlay latest live config values so they persist across reloads
    if os.path.exists("config/live_config.json"):
        try:
            with open("config/live_config.json", "r") as f:
                live_data = json.load(f)
                if "rewards" in live_data and isinstance(live_data["rewards"], dict):
                    rew_cfg.update(live_data["rewards"])
                if "scenarios" in live_data and isinstance(live_data["scenarios"], dict):
                    sc_cfg.update(live_data["scenarios"])
                if "learning_rate" in live_data:
                    hp_cfg["learning_rate"] = float(live_data["learning_rate"])
                if "ent_coef" in live_data:
                    hp_cfg["ent_coef"] = float(live_data["ent_coef"])
                if "clip_range" in live_data:
                    hp_cfg["clip_range"] = float(live_data["clip_range"])
                if "baseline_opponent_ratio" in live_data:
                    env_cfg["baseline_opponent_ratio"] = float(live_data["baseline_opponent_ratio"])
                if "baseline_opponent_type" in live_data:
                    env_cfg["baseline_opponent_type"] = str(live_data["baseline_opponent_type"])
                elif "baseline_opponent_model" in live_data:
                    env_cfg["baseline_opponent_type"] = str(live_data["baseline_opponent_model"])
                if "bc_regularization_weight" in live_data:
                    hp_cfg["bc_regularization_weight"] = float(live_data["bc_regularization_weight"])
                if "bc_decay_steps" in live_data:
                    hp_cfg["bc_decay_steps"] = int(live_data["bc_decay_steps"])
        except Exception:
            pass

    init_status = mgr.get_status_info()
    ts_evaluator = TrueSkillEvaluator()

    with gr.Blocks(title="SensAI - Rocket League ML Studio") as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.Markdown(
            """
            # 🏎️⚽ SensAI - Rocket League ML Studio
            ### High-Performance Headless Reinforcement Learning with Vectorized PPO & Live Tuning
            """
        )

        # -------------------------------------------------------------
        # TOP STATUS HERO BANNER
        # -------------------------------------------------------------
        status_card = gr.HTML(build_status_card_html(init_status))

        # -------------------------------------------------------------
        # STREAMLINED ACTION CONTROLS
        # -------------------------------------------------------------
        with gr.Row(elem_classes=["action-bar-row"]):
            with gr.Column(scale=2):
                resume_chk = gr.Checkbox(
                    label="Auto-Resume Latest Checkpoint",
                    value=True,
                    info="Resumes from checkpoints/latest_model.pt. Uncheck for fresh run."
                )
            with gr.Column(scale=5):
                with gr.Row():
                    start_btn = gr.Button(
                        "🚀 Start Training" if not init_status["running"] else "🟢 Training Active",
                        variant="primary" if not init_status["running"] else "secondary",
                        interactive=not init_status["running"]
                    )
                    pause_btn = gr.Button(
                        "⏸️ Pause Training" if not init_status.get("paused", False) else "▶️ Resume Training",
                        variant="primary" if init_status.get("paused", False) else "secondary",
                        interactive=init_status["running"]
                    )
                    stop_btn = gr.Button(
                        "🛑 Stop Training",
                        variant="stop" if init_status["running"] else "secondary",
                        interactive=init_status["running"]
                    )
                    ckpt_btn = gr.Button("💾 Save Checkpoint", variant="secondary")

        # -------------------------------------------------------------
        # 4 STREAMLINED TOP-LEVEL TABS
        # -------------------------------------------------------------
        with gr.Tabs():

            # =========================================================
            # TAB 1: 🏠 LIVE COCKPIT (ALL-IN-ONE HOME DASHBOARD)
            # =========================================================
            with gr.TabItem("🏠 Live Cockpit"):
                gr.Markdown(
                    """
                    > **⚡ Real-Time Training Control Center:**
                    > Modify PPO hyperparameters, opponent bot sparring mix, and core reward weights dynamically on the fly while monitoring real-time loss/reward telemetry and process output.
                    """
                )
                with gr.Row():
                    # Left Column: Live Tuners & Dynamic Dials
                    with gr.Column(scale=5):
                        # Card 1: Live Hyperparameters
                        with gr.Group():
                            gr.Markdown("### 🧠 Live Hyperparameters")
                            with gr.Row():
                                lr_input = gr.Number(
                                    value=hp_cfg.get("learning_rate", 3e-4),
                                    label="Learning Rate",
                                    info="PPO Policy & Value step size.",
                                    scale=1
                                )
                                ent_coef_slider = gr.Slider(
                                    0.0, 0.05,
                                    value=hp_cfg.get("ent_coef", 0.005),
                                    step=0.001,
                                    label="Entropy Coef",
                                    info="Exploration bonus.",
                                    scale=2
                                )
                            clip_range_slider = gr.Slider(
                                0.05, 0.4,
                                value=hp_cfg.get("clip_range", 0.2),
                                step=0.01,
                                label="PPO Clip Range",
                                info="Surrogate clipping bounds (epsilon)."
                            )
                            live_hp_btn = gr.Button("⚡ Apply Live Hyperparameters", variant="primary")
                            live_hp_msg = gr.Markdown("")

                        # Card 2: Fixed Training Opponents
                        with gr.Group():
                            gr.Markdown("### 👥 Fixed Training Opponents")
                            gr.Markdown(
                                "*Models here are guaranteed a block of environments, split evenly between "
                                "them. The remainder keeps the standard 50% self-play / 25% King / 25% pool "
                                "split. At zero the league behaves exactly as if this list were empty. "
                                "Listed models are excluded from the pool rotation, so their share is "
                                "exactly what you set here.*"
                            )
                            league_cfg_ui = default_cfg.get("league", {}) or {}

                            # Environments, not a percentage.
                            #
                            # Each subprocess worker owns a contiguous block of environments and a rollout
                            # step waits on all of them, so a share that ends mid-block leaves one worker
                            # holding two opponent models and paying an unbatched forward pass every step.
                            # Asking for a percentage made that easy to trip over: 20% of 128 is 25.6, which
                            # put five of sixteen workers off their boundary. Stepping by the block size
                            # means every position on this slider is one the scheduler can honour exactly.
                            ui_num_envs = max(1, int(env_cfg.get("num_envs", 64) or 64))
                            ui_workers = max(1, int(env_cfg.get("num_env_workers", 1) or 1))
                            env_block = max(1, ui_num_envs // ui_workers)

                            def _opp_env_readout(count: float) -> str:
                                count = int(count or 0)
                                pct = 100.0 * count / ui_num_envs
                                if count <= 0:
                                    return (f"**0 / {ui_num_envs} environments** &middot; the league picks "
                                            "every opponent on its own")
                                blocks = count // env_block
                                return (f"**{count} / {ui_num_envs} environments** &middot; {pct:.3g}% of the "
                                        f"rollout &middot; {blocks} of {ui_workers} workers")

                            _opp_start = int(round(float(league_cfg_ui.get(
                                "training_opponent_ratio",
                                env_cfg.get("baseline_opponent_ratio", 0.0)
                            ) or 0.0) * ui_num_envs))
                            _opp_start = min(ui_num_envs, (_opp_start // env_block) * env_block)
                            with gr.Row():
                                training_opponents_select = gr.Dropdown(
                                    choices=get_available_opponent_options(),
                                    value=[str(x) for x in league_cfg_ui.get("training_opponents", []) if x],
                                    multiselect=True,
                                    label="Opponent List",
                                    info="Add or remove models. Empty means the league picks opponents on its own.",
                                    scale=3
                                )
                                refresh_opponent_btn = gr.Button("🔄 Scan", scale=1)

                            baseline_opp_slider = gr.Slider(
                                0, ui_num_envs,
                                value=_opp_start,
                                step=env_block,
                                label="Environments for this list",
                                info=(f"Steps of {env_block}, one env-worker's block. Divided evenly among "
                                      "the entries above.")
                            )
                            opp_env_readout = gr.Markdown(_opp_env_readout(_opp_start))
                            baseline_opp_slider.change(
                                fn=_opp_env_readout,
                                inputs=[baseline_opp_slider],
                                outputs=[opp_env_readout],
                            )
                            apply_opp_btn = gr.Button("⚡ Apply Opponent Mix", variant="secondary")
                            opp_apply_msg = gr.Markdown("")

                        # Card 2b: Gauntlet Evaluation Budget
                        with gr.Group():
                            gr.Markdown("### 🥊 Gauntlet Evaluation Budget")
                            gr.Markdown(
                                "*Series each contender plays per trial. This is the whole knob for how "
                                "much CPU grading takes: matches run in a separate process alongside "
                                "training, so a higher setting ranks checkpoints sooner and leaves less "
                                "machine for everything else. Applies live, no restart.*"
                            )

                            def _budget_readout(step: float) -> str:
                                st = mgr.get_status_info()
                                est = gauntlet_budget_estimate(
                                    int(step or 1),
                                    default_cfg.get("league", {}) or {},
                                    default_cfg.get("logging", {}) or {},
                                    default_cfg.get("hyperparameters", {}) or {},
                                    float((st.get("metrics") or {}).get("sps") or 0.0),
                                )
                                return (
                                    f"**{int(step)} series per trial** &middot; about "
                                    f"{est['duty_pct']:.0f}% of the machine's evaluation window &middot; "
                                    f"a contender is ranked in ~{est['minutes_to_rank']:.0f} min &middot; "
                                    f"~{est['ranked_pct']:.0f}% of saved checkpoints get ranked"
                                )

                            _budget_start = int(league_cfg_ui.get("contender_series_per_step", 16))
                            gauntlet_budget_slider = gr.Slider(
                                4, 32,
                                value=_budget_start,
                                step=4,
                                label="Series per gauntlet trial",
                                info=(
                                    "Ranking one checkpoint costs target_eval_matches series, so at 30 "
                                    "and above every save gets ranked and nothing queues."
                                ),
                            )
                            gauntlet_budget_readout = gr.Markdown(_budget_readout(_budget_start))
                            gauntlet_budget_slider.change(
                                fn=_budget_readout,
                                inputs=[gauntlet_budget_slider],
                                outputs=[gauntlet_budget_readout],
                            )
                            apply_budget_btn = gr.Button("⚡ Apply Evaluation Budget", variant="secondary")
                            budget_apply_msg = gr.Markdown("")

                        # Card 3: Quick Live Reward Weights
                        with gr.Group():
                            gr.Markdown("### 🎛️ Quick Live Reward Weights")
                            with gr.Row():
                                goal_slider = gr.Slider(0.0, 30.0, value=float(rew_cfg.get("goal_weight", 20.0)), step=1.0, label="Goal (+pts)")
                                concede_slider = gr.Slider(-30.0, 0.0, value=float(rew_cfg.get("concede_weight", -20.0)), step=1.0, label="Concede (-pts)")
                            with gr.Row():
                                save_slider = gr.Slider(0.0, 15.0, value=float(rew_cfg.get("save_weight", 3.0)), step=0.5, label="Save (+pts)")
                                touch_slider = gr.Slider(0.0, 5.0, value=float(rew_cfg.get("touch_weight", 1.2)), step=0.1, label="Touch Quality")
                            with gr.Row():
                                ball_to_goal_slider = gr.Slider(0.0, 5.0, value=float(rew_cfg.get("ball_to_goal_weight", 1.5)), step=0.1, label="Ball to Goal")
                                player_to_ball_slider = gr.Slider(0.0, 3.0, value=float(rew_cfg.get("player_to_ball_weight", 0.6)), step=0.1, label="Ball Pursuit")
                            with gr.Row():
                                boost_gain_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("boost_gain_weight", 0.6)), step=0.05, label="Boost Gain (Sqrt)")
                                boost_lose_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("boost_lose_weight", 0.3)), step=0.05, label="Boost Waste")
                                boost_pathing_slider = gr.Slider(0.0, 100.0, value=float(rew_cfg.get("boost_pathing_threshold", 50.0)), step=5.0, label="Low Boost Pathing Ceiling")
                            apply_live_rewards_btn = gr.Button("⚡ Apply Live Rewards", variant="primary")
                            live_rewards_msg = gr.Markdown("")
                            gr.Markdown("<span style='color: #94a3b8; font-size: 0.88em;'>💡 For high aerials, jump bridges, air-roll recoveries, and custom scenario probabilities, visit the <b>🎛️ Rewards & Curriculum</b> tab.</span>")

                    # Right Column: Auto-Updating Metrics Plot & Live Console Output
                    with gr.Column(scale=6):
                        with gr.Group():
                            with gr.Row():
                                gr.Markdown("### 📈 Live Training Progress & Telemetry")
                                metrics_window_radio = gr.Radio(
                                    ["Recent 100", "Full Run"],
                                    value="Recent 100",
                                    label="Telemetry Window",
                                    scale=2
                                )
                                refresh_metrics_btn = gr.Button("🔄 Refresh", size="sm", scale=1)
                            live_metrics_plot = gr.Plot(
                                value=render_training_curves_plot(mode="recent"),
                                label="Telemetry Curves (Mean Reward, Losses, Entropy, SPS)"
                            )

                        with gr.Group():
                            with gr.Row():
                                gr.Markdown("### 📜 Real-Time Process Output Stream")
                                refresh_logs_btn = gr.Button("🔄 Refresh Logs", size="sm", scale=1)
                                clear_logs_btn = gr.Button("🧹 Clear", size="sm", scale=1)
                            console_output = gr.TextArea(
                                value=mgr.get_logs(),
                                label="Training Process Output (stdout / stderr)",
                                lines=13,
                                max_lines=18,
                                interactive=False,
                                autoscroll=True
                            )

                gr.Markdown("---")
                # -------------------------------------------------------------
                # SECTION: 🏆 TOP PERFORMING ITERATIONS LEADERBOARD
                # -------------------------------------------------------------
                with gr.Group():
                    with gr.Row():
                        gr.Markdown("### 🏆 Top Performing Iterations (Live TrueSkill Leaderboard)")
                        refresh_cockpit_lb_btn = gr.Button("🔄 Refresh Standings", size="sm", scale=0)
                    # Full width of its own: squeezed between the title and the button it
                    # wrapped onto three lines and read as a broken toolbar.
                    cockpit_anchor_legend = gr.HTML(build_anchor_legend_html(ts_evaluator))

                    cockpit_lb_summary = gr.HTML(build_cockpit_leaderboard_summary_html(ts_evaluator))
                    cockpit_league_ticker = gr.HTML(build_league_wire_and_queue_html(ts_evaluator))
                    cockpit_lb_table = gr.Dataframe(
                        value=get_cockpit_leaderboard_df(ts_evaluator),
                        label="Full Standings — ranked by μ among rank-eligible models (σ ≤ 1.5), provisional models below",
                        interactive=False
                    )

            # =========================================================
            # TAB 2: 🎛️ REWARDS & CURRICULUM STUDIO
            # =========================================================
            with gr.TabItem("🎛️ Rewards & Curriculum"):
                gr.Markdown(
                    """
                    > **🏆 Advanced Reward Architecture & Dynamic Curriculum Studio:**
                    > Tune aerial jump bridge incentives, air-roll recoveries, powerslide drifts, normalized scenario probability distributions, and design custom situations.
                    """
                )

                with gr.Group():
                    gr.Markdown("### 🚀 Advanced Flight & Recovery Mechanics")
                    with gr.Row():
                        jump_bridge_slider = gr.Slider(0.0, 1.0, value=float(rew_cfg.get("jump_bridge_weight", 0.35)), step=0.05, label="Jump & Aerial Takeoff Incentive", info="Takeoff & speed-flip transition bounty (2.0x on elevated aerials).")
                        air_roll_recovery_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("air_roll_recovery_weight", 0.10)), step=0.05, label="Air-Roll & Landing Recovery", info="Rewards wheels-down recovery on descent.")
                        powerslide_slider = gr.Slider(0.0, 2.0, value=float(rew_cfg.get("powerslide_weight", 0.20)), step=0.05, label="Powerslide & Drift Cut Bounty", info="Rewards handbrake powerslides on sharp turns.")

                with gr.Group():
                    with gr.Row():
                        with gr.Column(scale=4):
                            gr.Markdown("### 🎲 Dynamic Scenario Setter Distribution (Normalized 100% Group)")
                            gr.Markdown("*Move any slider — the group dynamically rebalances and snaps to 0.01 so the total always equals 100%.*")
                        with gr.Column(scale=1):
                            scenario_total_badge = gr.HTML(
                                """
                                <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                                    <span class="status-badge-running" style="font-size: 1.0em; padding: 6px 16px;">● Total Mix: 100%</span>
                                </div>
                                """
                            )

                    with gr.Accordion("⚖️ Scenario Weight Lock & Auto-Fill Popover", open=False):
                        gr.Markdown("#### 🎯 Specify Target Scenario Weights & Auto-Balance Remaining to 100%")
                        gr.Markdown("*Check the box next to any scenarios you want to lock at a specific percentage. Any unselected/unlocked scenarios will automatically divide the remaining percentage evenly to reach exactly 100%.*")
                        with gr.Row():
                            with gr.Column():
                                with gr.Row():
                                    pop_lock_k = gr.Checkbox(label="Lock Kickoff", value=False)
                                    pop_val_k = gr.Number(label="Kickoff %", value=int(round(float(sc_cfg.get("kickoff_prob", 0.20)) * 100)), minimum=0, maximum=100, step=1)
                                with gr.Row():
                                    pop_lock_r = gr.Checkbox(label="Lock Replay", value=False)
                                    pop_val_r = gr.Number(label="Replay %", value=int(round(float(sc_cfg.get("replay_prob", 0.15)) * 100)), minimum=0, maximum=100, step=1)
                                with gr.Row():
                                    pop_lock_a = gr.Checkbox(label="Lock Aerial", value=False)
                                    pop_val_a = gr.Number(label="Aerial %", value=int(round(float(sc_cfg.get("aerial_prob", 0.11)) * 100)), minimum=0, maximum=100, step=1)
                                with gr.Row():
                                    pop_lock_c = gr.Checkbox(label="Lock Custom", value=False)
                                    pop_val_c = gr.Number(label="Custom %", value=int(round(float(sc_cfg.get("custom_prob", 0.15)) * 100)), minimum=0, maximum=100, step=1)
                            with gr.Column():
                                with gr.Row():
                                    pop_lock_tr = gr.Checkbox(label="Lock Turnaround", value=False)
                                    pop_val_tr = gr.Number(label="Turnaround %", value=int(round(float(sc_cfg.get("turnaround_prob", 0.13)) * 100)), minimum=0, maximum=100, step=1)
                                with gr.Row():
                                    pop_lock_w = gr.Checkbox(label="Lock Wall Play", value=False)
                                    pop_val_w = gr.Number(label="Wall Play %", value=int(round(float(sc_cfg.get("wall_prob", 0.07)) * 100)), minimum=0, maximum=100, step=1)
                                with gr.Row():
                                    pop_lock_wr = gr.Checkbox(label="Lock Wall Rebound", value=False)
                                    pop_val_wr = gr.Number(label="Wall Rebound %", value=int(round(float(sc_cfg.get("wall_rebound_prob", 0.08)) * 100)), minimum=0, maximum=100, step=1)
                                with gr.Row():
                                    pop_lock_s = gr.Checkbox(label="Lock Goalie Save", value=False)
                                    pop_val_s = gr.Number(label="Goalie Save %", value=int(round(float(sc_cfg.get("save_prob", 0.07)) * 100)), minimum=0, maximum=100, step=1)
                                with gr.Row():
                                    pop_lock_df = gr.Checkbox(label="Lock Dribble & Flick", value=False)
                                    pop_val_df = gr.Number(label="Dribble & Flick %", value=int(round(float(sc_cfg.get("dribble_flick_prob", 0.08)) * 100)), minimum=0, maximum=100, step=1)

                        with gr.Row():
                            pop_sync_btn = gr.Button("🔄 Sync from Sliders", variant="secondary", size="sm")
                            pop_confirm_btn = gr.Button("⚡ Confirm & Auto-Balance to 100%", variant="primary", size="sm")

                        popover_status_msg = gr.Markdown("")

                    with gr.Row():
                        with gr.Column():
                            kickoff_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("kickoff_prob", 0.20)), step=0.01, label="Kickoff Scenario Probability", info="Standard 1v1 kickoff formations.")
                            replay_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("replay_prob", 0.15)), step=0.01, label="Human Replay Scenario Probability", info="Authentic match situations sampled from replays.")
                            aerial_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("aerial_prob", 0.11)), step=0.01, label="High Aerial Scenario Probability", info="Floating & rising balls for aerial training.")
                            custom_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("custom_prob", 0.15)), step=0.01, label="🎯 Custom Scenarios Probability", info="User-designed custom situations.")

                        with gr.Column():
                            turnaround_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("turnaround_prob", 0.13)), step=0.01, label="Turnaround Recovery Probability", info="Fast downfield spawns moving away from ball.")
                            wall_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("wall_prob", 0.07)), step=0.01, label="Wall Play Scenario Probability", info="Sidewall rolling and backboard rides.")
                            wall_rebound_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("wall_rebound_prob", 0.08)), step=0.01, label="Wall Rebound & Bounce Probability", info="High-speed sidewall & backboard clears to practice reading rebounds.")
                            save_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("save_prob", 0.07)), step=0.01, label="Goalie Save Scenario Probability", info="Fast opponent shots into defending net.")
                            dribble_flick_prob_slider = gr.Slider(0.0, 1.0, value=float(sc_cfg.get("dribble_flick_prob", 0.08)), step=0.01, label="Dribble & Flick Scenario Probability", info="Settled roof carry moving downfield against challenging defender or goalie.")

                    custom_sc_count = len(ScenarioManager.get_instance().get_active_scenarios())
                    gr.HTML(
                        f"""
                        <div style="background: rgba(15, 23, 42, 0.65); border: 1px solid #334155; border-radius: 8px; padding: 9px 16px; margin-top: 8px; font-size: 0.9em; display: flex; justify-content: space-between; align-items: center; box-shadow: inset 0 1px 3px rgba(0,0,0,0.3);">
                            <span>📦 <b>Custom Scenarios Distribution Pool:</b> <b style="color: #38bdf8;">{custom_sc_count} Active Scenarios</b> enabled in training rotation.</span>
                            <span style="color: #94a3b8;">Design and test custom drills below.</span>
                        </div>
                        """
                    )

                with gr.Group():
                    gr.Markdown("### 👤 Human Replay Guidance (BC Regularization)")
                    with gr.Row():
                        bc_weight_slider = gr.Slider(0.0, 1.0, value=float(hp_cfg.get("bc_regularization_weight", 0.10)), step=0.01, label="Replay Guidance Weight", info="Nudges vehicle steering and throttle from human replays.")
                        bc_decay_input = gr.Number(value=int(hp_cfg.get("bc_decay_steps", 150000000)), precision=0, label="Replay Guidance Decay Horizon (Steps)", info="Threshold over which guidance decays to 0.0.")

                with gr.Row():
                    apply_all_curriculum_btn = gr.Button("⚡ Apply All Curriculum & Reward Dials", variant="primary")
                    reset_curriculum_btn = gr.Button("🔄 Reset to Balanced Standard Dials", variant="secondary")

                curriculum_apply_msg = gr.Markdown("")

                gr.Markdown("---")

                # Sub-Section: Custom Scenario Generator & Builder
                gr.Markdown("### 🎯 Custom Scenario Generator & Interactive Pitch Builder")
                sc_mgr = ScenarioManager.get_instance()
                all_scenarios = sc_mgr.get_all_scenarios()
                initial_sc = all_scenarios[0] if all_scenarios else DEFAULT_CUSTOM_SCENARIOS[0]

                with gr.Row():
                    # Left Column: 2D Visual Guide Preview & Simulation Rollout
                    with gr.Column(scale=5):
                        gr.Markdown("#### 🗺️ Live 2D Pitch Visual Guide")
                        sc_preview_plot = gr.Plot(
                            value=render_scenario_visual_guide(initial_sc),
                            label="Interactive 2D Pitch Preview"
                        )
                        with gr.Row():
                            preset_dropdown = gr.Dropdown(
                                choices=["(Select Template Preset...)"] + [sc["name"] for sc in DEFAULT_CUSTOM_SCENARIOS],
                                value="(Select Template Preset...)",
                                label="⚡ Quick Template Presets",
                                scale=3
                            )
                            refresh_preview_btn = gr.Button("🔄 Refresh Guide", scale=1)

                        with gr.Accordion("🧪 2-Second Trajectory Rollout Simulation", open=False):
                            gr.Markdown("*Runs 150 steps in RocketSim from this custom scenario with active bot policy to preview physics response.*")
                            sim_scenario_btn = gr.Button("🚀 Simulate Scenario Physics (2s Rollout)", variant="primary")
                            sc_sim_plot = gr.Plot(label="Trajectory Rollout Plot")
                            sc_sim_stats = gr.JSON(label="Rollout Diagnostics")

                    # Right Column: Interactive Parameter Controls
                    with gr.Column(scale=6):
                        with gr.Group():
                            gr.Markdown("#### 📝 Scenario Metadata")
                            with gr.Row():
                                sc_id_input = gr.Textbox(label="Scenario ID (Unique Key)", value=initial_sc.get("id", "opposing_third_bouncing_ball"), scale=2)
                                sc_name_input = gr.Textbox(label="Scenario Name", value=initial_sc.get("name", "Opposing 1/3rd Bouncing Powershot / Dribble"), scale=3)
                                sc_enabled_cb = gr.Checkbox(label="Active in Training Pool", value=initial_sc.get("enabled", True), scale=1)
                            sc_desc_input = gr.Textbox(
                                label="Tactical Intent / Description",
                                value=initial_sc.get("description", "Bot spawns in opposing 1/3rd behind bouncing ball."),
                                lines=2
                            )

                        with gr.Group():
                            gr.Markdown("#### 🏎️ Bot State (Car 0 / Blue)")
                            with gr.Row():
                                car_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc["car"]["pos"][0]), step=25.0, label="Pos X (Left / Right)")
                                car_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc["car"]["pos"][1]), step=25.0, label="Pos Y (Goal to Goal)")
                                car_pos_z = gr.Slider(17.0, 1600.0, value=float(initial_sc["car"]["pos"][2]), step=10.0, label="Pos Z (Altitude)")
                            with gr.Row():
                                car_yaw = gr.Slider(-180.0, 180.0, value=float(initial_sc["car"].get("yaw", 90.0)), step=5.0, label="Heading / Yaw (deg: 90° = +Y, -90° = -Y)")
                                car_speed = gr.Slider(0.0, 2300.0, value=float(math.hypot(initial_sc["car"]["vel"][0], initial_sc["car"]["vel"][1])), step=25.0, label="Forward Velocity Speed (uu/s)")
                                car_boost = gr.Slider(0.0, 100.0, value=float(initial_sc["car"].get("boost", 50.0)), step=5.0, label="Starting Boost Amount (%)")

                        with gr.Group():
                            gr.Markdown("#### ⚽ Ball State")
                            with gr.Row():
                                ball_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc["ball"]["pos"][0]), step=25.0, label="Ball Pos X")
                                ball_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc["ball"]["pos"][1]), step=25.0, label="Ball Pos Y")
                                ball_pos_z = gr.Slider(93.15, 1800.0, value=float(initial_sc["ball"]["pos"][2]), step=10.0, label="Ball Pos Z (Height)")
                            with gr.Row():
                                ball_vel_x = gr.Slider(-2500.0, 2500.0, value=float(initial_sc["ball"]["vel"][0]), step=25.0, label="Ball Vel X (uu/s)")
                                ball_vel_y = gr.Slider(-2500.0, 2500.0, value=float(initial_sc["ball"]["vel"][1]), step=25.0, label="Ball Vel Y (uu/s)")
                                ball_vel_z = gr.Slider(-1500.0, 1500.0, value=float(initial_sc["ball"]["vel"][2]), step=25.0, label="Ball Vel Z (uu/s)")

                        with gr.Group():
                            gr.Markdown("#### 👤 Opponent State (Car 1 / Orange)")
                            with gr.Row():
                                opp_mode_radio = gr.Radio(["goalie", "shadow", "custom", "none"], value=initial_sc.get("opponent", {}).get("mode", "goalie"), label="Opponent Placement Mode")
                                opp_boost = gr.Slider(0.0, 100.0, value=float(initial_sc.get("opponent", {}).get("boost", 60.0)), step=5.0, label="Opponent Boost (%)")
                            with gr.Row(visible=(initial_sc.get("opponent", {}).get("mode", "goalie") == "custom")) as opp_custom_row:
                                opp_pos_x = gr.Slider(-3800.0, 3800.0, value=float(initial_sc.get("opponent", {}).get("pos", [0, 4800, 17])[0]), step=25.0, label="Custom Opponent Pos X")
                                opp_pos_y = gr.Slider(-4800.0, 4800.0, value=float(initial_sc.get("opponent", {}).get("pos", [0, 4800, 17])[1]), step=25.0, label="Custom Opponent Pos Y")
                                opp_yaw = gr.Slider(-180.0, 180.0, value=float(initial_sc.get("opponent", {}).get("yaw", -90.0)), step=5.0, label="Custom Opponent Yaw (deg)")

                        with gr.Group():
                            gr.Markdown("#### 🎲 Training Variance & Symmetry")
                            with gr.Row():
                                pos_jitter = gr.Slider(0.0, 300.0, value=float(initial_sc.get("variance", {}).get("pos_jitter", 80.0)), step=10.0, label="Positional Jitter (±uu)", info="Adds natural positional randomness each spawn.")
                                vel_jitter = gr.Slider(0.0, 300.0, value=float(initial_sc.get("variance", {}).get("vel_jitter", 60.0)), step=10.0, label="Velocity Jitter (±uu/s)", info="Adds velocity variance each spawn.")
                                mirror_symmetry = gr.Checkbox(value=bool(initial_sc.get("variance", {}).get("mirror_symmetry", True)), label="Left/Right Mirror Symmetry (50% Chance)", info="Mirrors scenario across X-axis so bot trains both sides.")

                        with gr.Row():
                            save_scenario_btn = gr.Button("💾 Save / Update Custom Scenario", variant="primary")
                            new_scenario_btn = gr.Button("➕ New / Clear Form", variant="secondary")
                            delete_scenario_btn = gr.Button("🗑️ Delete Scenario", variant="stop")

                        scenario_action_msg = gr.Markdown("")

                gr.Markdown("#### 📚 Saved Custom Scenarios Library")
                def build_scenarios_table():
                    items = ScenarioManager.get_instance().get_all_scenarios()
                    rows = []
                    for s in items:
                        bp = s.get("ball", {}).get("pos", [0, 0, 93])
                        cp = s.get("car", {}).get("pos", [0, 0, 17])
                        rows.append([
                            s.get("id", ""),
                            s.get("name", ""),
                            s.get("enabled", True),
                            f"({bp[0]:.0f}, {bp[1]:.0f}, {bp[2]:.0f})",
                            f"({cp[0]:.0f}, {cp[1]:.0f}, {cp[2]:.0f})",
                            s.get("description", "")
                        ])
                    return pd.DataFrame(rows, columns=["ID", "Name", "Active", "Ball Pos", "Car Pos", "Description"]) if rows else pd.DataFrame(columns=["ID", "Name", "Active", "Ball Pos", "Car Pos", "Description"])

                saved_scenarios_table = gr.Dataframe(
                    value=build_scenarios_table(),
                    interactive=False,
                    label="Custom Scenarios Pool"
                )
                with gr.Row():
                    load_scenario_dropdown = gr.Dropdown(
                        choices=[f"{sc['name']} ({sc['id']})" for sc in all_scenarios],
                        value=f"{initial_sc['name']} ({initial_sc['id']})" if all_scenarios else None,
                        label="Select Scenario from Library to Load / Edit",
                        scale=3
                    )
                    load_scenario_btn = gr.Button("📥 Load Selected Scenario", scale=1)
                    refresh_library_btn = gr.Button("🔄 Refresh Library Table", scale=1)

            # =========================================================
            # TAB 3: ⚙️ ENGINE CONFIG & PRETRAINER
            # =========================================================
            with gr.TabItem("⚙️ Config & Pretrainer"):
                gr.Markdown(
                    """
                    > **⚙️ Base Architecture Configuration & Behavioral Cloning Pretrainer:**
                    > Tune offline PPO hyperparameters, vectorized arena simulation settings, ingest human `.replay` match files, and run supervised imitation pretraining.
                    """
                )
                with gr.Row():
                    # Left Column: PPO Hyperparameters & Arena Settings
                    with gr.Column(scale=5):
                        with gr.Group():
                            gr.Markdown("### 🧠 Offline PPO Hyperparameters")
                            with gr.Row():
                                gamma_slider = gr.Slider(
                                    0.9, 0.999, value=hp_cfg.get("gamma", 0.99), step=0.001,
                                    label="Discount Factor (Gamma)",
                                    info="Future rewards discount value."
                                )
                                gae_lambda_slider = gr.Slider(
                                    0.8, 1.0, value=hp_cfg.get("gae_lambda", 0.95), step=0.01,
                                    label="GAE Lambda",
                                    info="GAE variance vs bias trade-off."
                                )
                            with gr.Row():
                                batch_size_input = gr.Number(
                                    value=hp_cfg.get("batch_size", 8192), precision=0,
                                    label="Rollout Buffer Batch Size",
                                    info="Total steps per iteration across arenas."
                                )
                                mini_batch_input = gr.Number(
                                    value=hp_cfg.get("mini_batch_size", 512), precision=0,
                                    label="Mini-Batch Size",
                                    info="Gradient update chunk size."
                                )
                                n_epochs_input = gr.Number(
                                    value=hp_cfg.get("n_epochs", 10), precision=0,
                                    label="Epochs per Iteration",
                                    info="Optimization passes per rollout."
                                )

                        with gr.Group():
                            gr.Markdown("### 🏟️ Simulation & Checkpointing")
                            with gr.Row():
                                num_envs_slider = gr.Slider(
                                    1, 128, value=env_cfg.get("num_envs", 64), step=1,
                                    label="Vectorized Arenas",
                                    info="Parallel RocketSim arena instances."
                                )
                                tick_skip_slider = gr.Slider(
                                    1, 8, value=env_cfg.get("tick_skip", 8), step=1,
                                    label="Tick Skip (Action Repeat)",
                                    info="8 skip ≈ 15 decisions/sec."
                                )
                            with gr.Row():
                                max_steps_input = gr.Number(
                                    value=env_cfg.get("max_episode_steps", 750), precision=0,
                                    label="Max Episode Steps",
                                    info="750 steps ≈ 50s match time."
                                )
                                game_mode_dropdown = gr.Dropdown(
                                    ["1v1", "2v2", "3v3"], value=env_cfg.get("game_mode", "1v1"),
                                    label="Game Mode",
                                    info="Match format (1v1, 2v2, 3v3)."
                                )
                            gr.Markdown("#### 💾 Checkpointing & Retention")
                            with gr.Row():
                                autosave_interval_input = gr.Number(
                                    value=log_cfg.get("autosave_interval", 20), precision=0,
                                    label="Autosave Interval (Iters)",
                                    info="Rewrites latest_model.pt for crash recovery. Creates no new files and never enters the league."
                                )
                                checkpoint_interval_input = gr.Number(
                                    value=log_cfg.get("checkpoint_interval", 200), precision=0,
                                    label="League Checkpoint Interval (Iters)",
                                    info="Mints a numbered checkpoint that gets TrueSkill-graded. Keep this coarse: checkpoints saved a few iterations apart are near-identical and evaluation cannot separate them."
                                )
                                archive_stride_input = gr.Number(
                                    value=log_cfg.get("archive_stride", 5000), precision=0,
                                    label="Archive Stride (Iters)",
                                    info="One checkpoint per stride is kept permanently as a historical spine. Set 0 to disable."
                                )
                            gr.Markdown(
                                "*Retention is the union of three tiers, so there is no rolling cap to tune: "
                                "**provisional** (newest un-converged checkpoints, protected until the evaluator reaches them), "
                                "**ranked** (King, Elite Pool, Hall of Fame, active contenders), and "
                                "**archive** (the permanent spine above). Everything else is pruned.*"
                            )

                            save_cfg_btn = gr.Button("💾 Save Configuration to YAML", variant="primary")
                            cfg_save_msg = gr.Markdown("")

                    # Right Column: Human Replay Dataset & Imitation Pretrainer
                    with gr.Column(scale=6):
                        def build_replay_stats_md():
                            parser = ReplayParser()
                            st = parser.get_pool_stats()
                            total_frames = st.get('total_frames', 0)
                            num_matches = st.get('num_matches', 0)
                            file_size_mb = st.get('file_size_mb', 0.0)
                            est_game_time = (total_frames / 15.0) / 60.0
                            has_data = total_frames > 0
                            badge = '<span class="status-badge-running">● DATASET ACTIVE</span>' if has_data else '<span class="status-badge-stopped">○ EMPTY DATASET</span>'
                            warning_banner = f"""
                            <div style="margin-top: 10px; padding: 8px 12px; background: rgba(245, 158, 11, 0.1); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 6px; font-size: 0.85em; color: #fbbf24;">
                                ⚠️ <b>Dataset pool is empty (0 frames).</b> Replay state initialization will fall back to kickoffs and scenarios until genuine .replay, .npz, or .json files are ingested.
                            </div>
                            """ if not has_data else f"""
                            <div style="margin-top: 10px; padding: 8px 12px; background: rgba(16, 185, 129, 0.1); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 6px; font-size: 0.85em; color: #34d399;">
                                ✅ <b>Dataset pool active ({total_frames:,} genuine frames).</b> Replay scenarios will sample authentic positions from this pool.
                            </div>
                            """
                            return f"""
                            <div class="cyber-panel">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                                    <h4 style="margin: 0; color: #f1f5f9;">📦 Human Replay Dataset Pool</h4>
                                    {badge}
                                </div>
                                <div style="display: flex; gap: 20px; color: #cbd5e1; font-size: 0.9em; flex-wrap: wrap;">
                                    <span>Replay Matches: <b style="color: #38bdf8;">{num_matches}</b></span>
                                    <span>Total Frames: <b style="color: #818cf8;">{total_frames:,}</b></span>
                                    <span>Est. Duration: <b style="color: #34d399;">{est_game_time:.1f} mins</b></span>
                                    <span>Pool Size: <b style="color: #facc15;">{file_size_mb:.2f} MB</b></span>
                                </div>
                                {warning_banner}
                            </div>
                            """

                        replay_stats_box = gr.HTML(build_replay_stats_md())

                        with gr.Group():
                            gr.Markdown("### 📂 Ingest from Directory")
                            with gr.Row():
                                demos_dir_input = gr.Textbox(
                                    value=get_default_demo_dir(),
                                    label="Replay Directory Path",
                                    info="Absolute or relative path where .replay files are stored."
                                )
                                scan_demos_btn = gr.Button("🔍 Scan Replays", scale=1)

                            with gr.Row():
                                max_replays_slider = gr.Slider(
                                    1, 100, value=20, step=1,
                                    label="Max Replays to Ingest",
                                    info="Cap the number of replays to parse into training buffer."
                                )
                                sort_replays_radio = gr.Radio(
                                    ["newest", "oldest"], value="newest",
                                    label="Sort Order"
                                )

                            demos_table = gr.Dataframe(
                                headers=["Filename", "Size (KB)", "Modified"],
                                datatype=["str", "number", "str"],
                                value=[],
                                label="Discovered Replays (Select or Ingest All)"
                            )

                            with gr.Row():
                                ingest_selected_btn = gr.Button("⚡ Ingest Discovered Replays", variant="primary")
                                ingest_all_btn = gr.Button("📥 Ingest ALL Replays in Directory", variant="secondary")
                                clear_pool_btn = gr.Button("🗑️ Clear Replay Dataset Pool", variant="stop")

                            replays_status_box = gr.Markdown("")

                        with gr.Group():
                            gr.Markdown("### 📤 Upload Dataset Files (.zip, .npz, .json)")
                            replay_uploader = gr.File(
                                file_count="multiple",
                                file_types=[".zip", ".npz", ".json", ".replay"],
                                label="Drop .zip, .npz, or .json replay datasets here"
                            )
                            upload_status_box = gr.Markdown("")

                        with gr.Group():
                            gr.Markdown("### 🎓 Behavioral Cloning (Imitation Pretrainer)")
                            with gr.Row():
                                pretrain_epochs_slider = gr.Slider(1, 20, value=5, step=1, label="Pretraining Epochs")
                                pretrain_lr_input = gr.Number(value=0.0005, label="BC Learning Rate")
                            with gr.Row():
                                pretrain_batch_dropdown = gr.Dropdown([64, 128, 256, 512], value=256, label="BC Batch Size")
                                pretrain_base_dropdown = gr.Dropdown(
                                    choices=get_available_checkpoints(),
                                    value=get_available_checkpoints()[0],
                                    label="Base Checkpoint"
                                )
                            with gr.Row():
                                run_pretrain_btn = gr.Button("🚀 Run Imitation Pretraining", variant="primary")
                                stop_pretrain_btn = gr.Button("⏹️ Stop Pretrainer", variant="stop")
                            pretrain_status_box = gr.HTML(
                                """
                                <div class="status-callout-box" style="border-left-color: #64748b;">
                                    <span style="color: #94a3b8; font-weight: 700; margin-right: 8px;">IDLE:</span>
                                    <span>Ready to train initial policy weights on parsed replay data.</span>
                                </div>
                                """
                            )

            # =========================================================
            # TAB 4: 🔬 DIAGNOSTICS & EVALUATION (FLATTENED HUB)
            # =========================================================
            with gr.TabItem("🔬 Diagnostics & Evaluation"):
                gr.Markdown(
                    """
                    ### 🔬 Unified Diagnostic & Evaluation Hub
                    Single-pane-of-glass workspace for full system health, automated unit test verification, 2D match simulation replays, behavioral bias heatmaps, and AI assistant snapshot export.
                    """
                )

                # SECTION 1: Automated Unit Tests & Health
                with gr.Group():
                    with gr.Row():
                        gr.Markdown("### 🧪 Subsystem Unit Tests & Health Verification")
                        run_unit_tests_btn = gr.Button("🧪 Run All Unit Tests", variant="primary", scale=1)
                    with gr.Row():
                        with gr.Column(scale=1):
                            unit_tests_overview_md = gr.Markdown(value=format_test_results_markdown(get_cached_or_run_tests()))
                        with gr.Column(scale=1):
                            unit_tests_stdout = gr.Code(
                                label="Test Runner Output Stream",
                                language="markdown",
                                lines=10,
                                interactive=False
                            )

                gr.Markdown("---")

                # SECTION 2: 2D Pitch Match Visualizer & Simulation
                with gr.Group():
                    gr.Markdown("### 🎮 2D Pitch Match Visualizer & Simulation Replay")
                    with gr.Row():
                        with gr.Column(scale=4):
                            ckpt_dropdown = gr.Dropdown(
                                choices=get_available_checkpoints(),
                                value=get_available_checkpoints()[0],
                                label="Select Blue Team Checkpoint",
                                info="Trained PyTorch model checkpoint (.pt) for Blue Team."
                            )
                            opponent_mode = gr.Radio(
                                ["Self-Play (Bot vs Itself)", "Baseline Bot (Chase Ball Heuristic)", "Another Checkpoint"],
                                value="Self-Play (Bot vs Itself)",
                                label="Opponent Matchup Type"
                            )
                            orange_ckpt_dropdown = gr.Dropdown(
                                choices=get_available_checkpoints(),
                                value=get_available_checkpoints()[0],
                                label="Select Orange Team Checkpoint",
                                visible=False,
                                info="Select a different checkpoint for Orange Team."
                            )
                            refresh_ckpts_btn = gr.Button("🔄 Scan Checkpoints")
                            sim_steps_slider = gr.Slider(
                                100, 1000, value=400, step=50,
                                label="Simulation Steps",
                                info="Duration of match simulation (400 steps ≈ 26s)."
                            )
                            run_sim_btn = gr.Button("🕹️ Simulate Match & Render Replay", variant="primary")
                            sim_stats_box = gr.Markdown("#### Match Results: Click 'Simulate Match' to evaluate.")

                        with gr.Column(scale=7):
                            visualizer_plot = gr.Plot(label="🗺️ 2D Pitch Trajectories")
                            reward_breakdown_plot = gr.Plot(label="📊 Match Reward Breakdown")

                gr.Markdown("---")

                # SECTION 3: Behavioral Biases & AI Coach
                with gr.Group():
                    with gr.Row():
                        gr.Markdown("### 🧠 Behavioral Biases & AI Behavioral Coach")
                        diag_window_slider = gr.Slider(
                            1, 25, value=8, step=1,
                            label="Rolling Average Window (Iterations)",
                            scale=2
                        )
                        refresh_diag_btn = gr.Button("🔄 Refresh AI Coach Analysis", variant="primary", scale=1)
                    with gr.Row():
                        with gr.Column(scale=1):
                            diag_coach_report = gr.Markdown(value="*Click 'Refresh AI Coach Analysis' or run training to view live AI coach analysis.*")
                        with gr.Column(scale=1):
                            diag_action_plot = gr.Plot(label="Action & Control Distributions")
                            diag_position_plot = gr.Plot(label="Pitch Positioning & Vehicle State Radar")

                gr.Markdown("---")

                # SECTION 4: TrueSkill Bayesian Rating & Tournament Leaderboard
                with gr.Group():
                    gr.Markdown("### 🏆 TrueSkill Bayesian Rating & Tournament Leaderboard")
                    gr.Markdown(
                        """
                        > **⚔️ Competitive Evaluation Hub:**
                        > Run symmetric home/away matches between saved checkpoints and benchmark anchors with golden-goal sudden-death overtime.
                        > Models are ranked by conservative rating ($\\mu - 3\\sigma$).
                        """
                    )
                    with gr.Row():
                        with gr.Column(scale=5):
                            all_ckpts = get_available_checkpoints()
                            init_selected = [c for c in all_ckpts if "latest_model" in c or "pretrained" in c]
                            ts_ckpt_multiselect = gr.Dropdown(
                                choices=all_ckpts,
                                value=init_selected if init_selected else (all_ckpts[:2] if len(all_ckpts) >= 2 else all_ckpts),
                                multiselect=True,
                                label="Select Checkpoints to Evaluate",
                                info="Choose trained models to participate in the tournament."
                            )
                            with gr.Row():
                                ts_select_all_btn = gr.Button("✅ Select All", size="sm")
                                ts_clear_all_btn = gr.Button("🧹 Clear All", size="sm")
                                ts_refresh_ckpts_btn = gr.Button("🔄 Scan Checkpoints", size="sm")

                            anchor_choices = ["Baseline Chaser (Heuristic)"]
                            if os.path.exists("checkpoints/pretrained_baseline.pt"):
                                anchor_choices.append("Pretrained Baseline (BC)")
                            if os.path.exists("checkpoints/necto-model.pt"):
                                anchor_choices.append("Necto (EARL TorchScript)")
                            if os.path.exists("checkpoints/nexto-model.pt"):
                                anchor_choices.append("Nexto (EARL TorchScript)")

                            ts_anchors_checkbox = gr.CheckboxGroup(
                                choices=anchor_choices,
                                value=["Baseline Chaser (Heuristic)"],
                                label="Include Reference Benchmarks & Anchors"
                            )

                            with gr.Row():
                                ts_matches_slider = gr.Slider(2, 6, value=2, step=2, label="Matches per Pair", info="Symmetric home/away (even #).")
                                ts_steps_slider = gr.Slider(100, 600, value=350, step=50, label="Simulation Steps", info="Match length.")

                            ts_overtime_check = gr.Checkbox(value=True, label="Sudden-Death Golden Goal Overtime (break ties)")

                            with gr.Row():
                                run_tournament_btn = gr.Button("⚔️ Run TrueSkill Tournament", variant="primary", scale=2)
                                reset_leaderboard_btn = gr.Button("🗑️ Reset Leaderboard", variant="secondary", scale=1)

                            ts_status_md = gr.Markdown("#### 🏁 Tournament Status: Ready. Select models and click 'Run TrueSkill Tournament'.")

                        with gr.Column(scale=7):
                            ts_leaderboard_table = gr.Dataframe(
                                value=ts_evaluator.get_leaderboard_dataframe(),
                                label="🏆 Ranked Model Standings",
                                interactive=False
                            )
                            ts_leaderboard_plot = gr.Plot(
                                value=ts_evaluator.render_leaderboard_plot(),
                                label="📊 TrueSkill Rating Distribution (μ ± 2σ Confidence Intervals)"
                            )

                gr.Markdown("---")

                # SECTION 5: Comprehensive System Snapshot & AI Assistant Export
                with gr.Group():
                    with gr.Row():
                        gr.Markdown("### 📋 System Snapshot & AI Assistant Export")
                        refresh_snapshot_btn = gr.Button("🔄 Refresh Diagnostic Snapshot", variant="primary", scale=1)
                    with gr.Row():
                        with gr.Column(scale=1):
                            diag_overview_md = gr.Markdown(value="*Click 'Refresh Diagnostic Snapshot' to generate live overview.*")
                        with gr.Column(scale=1):
                            diag_export_raw = gr.Code(
                                label="📋 Complete Diagnostic Snapshot (Copy & Paste to Assistant)",
                                language="markdown",
                                lines=20,
                                interactive=False
                            )

        # -------------------------------------------------------------
        # EVENT HANDLERS & CALLBACKS
        # -------------------------------------------------------------

        # Dynamic State Synchronizer
        def sync_ui_state(feedback_msg: str = ""):
            status = mgr.get_status_info()
            card_html = build_status_card_html(status, feedback_msg)
            running = status.get("running", False)
            paused = status.get("paused", False)

            start_btn_update = gr.update(
                value="🚀 Start Training" if not running else "🟢 Training Active",
                variant="primary" if not running else "secondary",
                interactive=not running
            )
            pause_btn_update = gr.update(
                value="▶️ Resume Training" if paused else "⏸️ Pause Training",
                variant="primary" if paused else "secondary",
                interactive=running
            )
            stop_btn_update = gr.update(
                value="🛑 Stop Training",
                variant="stop" if running else "secondary",
                interactive=running
            )
            return card_html, start_btn_update, pause_btn_update, stop_btn_update

        # Training Controls
        def on_start(resume_latest: bool = True):
            ckpt = None
            if resume_latest:
                # Find the highest iteration checkpoint available
                candidates = []
                if os.path.exists("checkpoints/latest_model.pt"):
                    candidates.append("checkpoints/latest_model.pt")
                candidates.extend(glob.glob("checkpoints/checkpoint_iter_*.pt"))
                candidates.extend(glob.glob("checkpoints/**/*.pt", recursive=True))
                
                best_ckpt = None
                best_iter = -1
                for c_path in candidates:
                    try:
                        data = torch.load(c_path, map_location="cpu", weights_only=False)
                        if isinstance(data, dict):
                            it = int(data.get("iteration", 0))
                            if it > best_iter:
                                best_iter = it
                                best_ckpt = c_path
                    except Exception:
                        pass
                ckpt = best_ckpt if best_ckpt else ("checkpoints/latest_model.pt" if os.path.exists("checkpoints/latest_model.pt") else None)

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

        # -------------------------------------------------------------
        # LIVE HYPERPARAMETERS & DIALS (TAB 1)
        # -------------------------------------------------------------
        def on_apply_live_hyperparams(lr_val, ent_val, clip_val):
            payload = {
                "learning_rate": float(lr_val),
                "ent_coef": float(ent_val),
                "clip_range": float(clip_val),
            }
            mgr.update_live_config(payload)
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                if "hyperparameters" not in base_cfg:
                    base_cfg["hyperparameters"] = {}
                base_cfg["hyperparameters"]["learning_rate"] = float(lr_val)
                base_cfg["hyperparameters"]["ent_coef"] = float(ent_val)
                base_cfg["hyperparameters"]["clip_range"] = float(clip_val)
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            return f"✅ **Live Hyperparameters Applied:** LR=`{float(lr_val):.2e}`, Ent Coef=`{float(ent_val):.4f}`, Clip=`{float(clip_val):.2f}` at {time.strftime('%H:%M:%S')}"

        live_hp_btn.click(
            fn=on_apply_live_hyperparams,
            inputs=[lr_input, ent_coef_slider, clip_range_slider],
            outputs=[live_hp_msg]
        )

        def on_apply_opponent_mix(opp_list, opp_envs):
            # The slider is in environments; the league stores a ratio. Converting here
            # rather than there keeps every stored value one the scheduler can honour
            # exactly, because the slider can only land on a worker-block boundary.
            opp_envs = max(0, min(ui_num_envs, int(opp_envs or 0)))
            opp_ratio = opp_envs / float(ui_num_envs)
            selected = []
            for item in (opp_list or []):
                text = str(item).strip()
                if not text:
                    continue
                selected.append("heuristic" if text.startswith("Heuristic") else text)
            # Preserve order while dropping duplicates: the share is split evenly, so a
            # repeated entry would quietly receive double weight.
            selected = list(dict.fromkeys(selected))
            payload = {
                "training_opponents": selected,
                "training_opponent_ratio": float(opp_ratio),
            }
            mgr.update_live_config(payload)
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                if "environment" not in base_cfg:
                    base_cfg["environment"] = {}
                if "league" not in base_cfg:
                    base_cfg["league"] = {}
                base_cfg["league"]["training_opponents"] = selected
                base_cfg["league"]["training_opponent_ratio"] = float(opp_ratio)
                # Retire the single-model control this replaces, so the two cannot drift.
                base_cfg.get("environment", {}).pop("baseline_opponent_type", None)
                base_cfg.get("environment", {}).pop("baseline_opponent_ratio", None)
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            if not selected or opp_envs <= 0:
                return (f"✅ **Fixed opponents cleared** — league picks opponents on its own "
                        f"(50% self-play / 25% King / 25% pool) at {time.strftime('%H:%M:%S')}")
            each = opp_envs // len(selected)
            spare = opp_envs - each * len(selected)
            names = ", ".join(f"`{os.path.basename(x)}`" for x in selected)
            share = (f"{each} env{'s' if each != 1 else ''} each"
                     + (f", {spare} rotating between them" if spare else ""))
            return (f"✅ **Fixed opponents applied:** {names} — {opp_envs}/{ui_num_envs} environments "
                    f"({share}) at {time.strftime('%H:%M:%S')}")

        apply_opp_btn.click(
            fn=on_apply_opponent_mix,
            inputs=[training_opponents_select, baseline_opp_slider],
            outputs=[opp_apply_msg]
        )

        def on_apply_gauntlet_budget(series_per_step):
            step = max(1, int(series_per_step or 1))
            # Live config reaches the trainer, which copies it into the dict handed to
            # each grading child; the yaml keeps it across restarts.
            mgr.update_live_config({"contender_series_per_step": step})
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                base_cfg.setdefault("league", {})["contender_series_per_step"] = step
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            st = mgr.get_status_info()
            est = gauntlet_budget_estimate(
                step,
                load_yaml_config("config/default_config.yaml").get("league", {}) or {},
                default_cfg.get("logging", {}) or {},
                default_cfg.get("hyperparameters", {}) or {},
                float((st.get("metrics") or {}).get("sps") or 0.0),
            )
            return (
                f"✅ **Evaluation budget applied:** {step} series per trial — "
                f"~{est['minutes_to_rank']:.0f} min to rank a contender, "
                f"~{est['duty_pct']:.0f}% evaluation duty at {time.strftime('%H:%M:%S')}"
            )

        apply_budget_btn.click(
            fn=on_apply_gauntlet_budget,
            inputs=[gauntlet_budget_slider],
            outputs=[budget_apply_msg]
        )

        refresh_opponent_btn.click(
            fn=lambda: gr.Dropdown(choices=get_available_opponent_options()),
            outputs=[training_opponents_select]
        )

        # Quick Live Rewards (Tab 1)
        def on_apply_quick_rewards(g_w, c_w, sv_w, b2g_w, p2b_w, tch_w, bg_w, bl_w, bp_th):
            rewards = {
                "goal_weight": float(g_w),
                "concede_weight": float(c_w),
                "save_weight": float(sv_w),
                "ball_to_goal_weight": float(b2g_w),
                "player_to_ball_weight": float(p2b_w),
                "touch_weight": float(tch_w),
                "boost_gain_weight": float(bg_w),
                "boost_lose_weight": float(bl_w),
                "boost_pathing_threshold": float(bp_th)
            }
            mgr.update_live_config({"rewards": rewards})
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                if "rewards" not in base_cfg:
                    base_cfg["rewards"] = {}
                base_cfg["rewards"].update(rewards)
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            return f"✅ **Live Rewards Applied at {time.strftime('%H:%M:%S')}!**"

        apply_live_rewards_btn.click(
            fn=on_apply_quick_rewards,
            inputs=[
                goal_slider, concede_slider, save_slider,
                ball_to_goal_slider, player_to_ball_slider, touch_slider,
                boost_gain_slider, boost_lose_slider, boost_pathing_slider
            ],
            outputs=[live_rewards_msg]
        )

        # -------------------------------------------------------------
        # FULL CURRICULUM & REWARD DIALS (TAB 2)
        # -------------------------------------------------------------
        def on_apply_curriculum(
            g_w, c_w, sv_w,
            b2g_w, p2b_w, jb_w, ar_w, pw_w, tch_w,
            bg_w, bl_w,
            k_p, r_p, a_p, c_p, tr_p, w_p, wr_p, s_p, df_p,
            bc_w, bc_dec
        ):
            rewards = {
                "goal_weight": float(g_w),
                "concede_weight": float(c_w),
                "save_weight": float(sv_w),
                "ball_to_goal_weight": float(b2g_w),
                "player_to_ball_weight": float(p2b_w),
                "jump_bridge_weight": float(jb_w),
                "air_roll_recovery_weight": float(ar_w),
                "powerslide_weight": float(pw_w),
                "touch_weight": float(tch_w),
                "boost_gain_weight": float(bg_w),
                "boost_lose_weight": float(bl_w)
            }
            scenarios = {
                "kickoff_prob": float(k_p),
                "replay_prob": float(r_p),
                "aerial_prob": float(a_p),
                "custom_prob": float(c_p),
                "turnaround_prob": float(tr_p),
                "wall_prob": float(w_p),
                "wall_rebound_prob": float(wr_p),
                "save_prob": float(s_p),
                "dribble_flick_prob": float(df_p)
            }

            payload = {
                "rewards": rewards,
                "scenarios": scenarios,
                "bc_regularization_weight": float(bc_w),
                "bc_decay_steps": int(bc_dec)
            }
            mgr.update_live_config(payload)
            try:
                base_cfg = load_yaml_config("config/default_config.yaml")
                # Merge, never replace: this dict is built from the sliders on this page, so
                # assigning it would delete every reward weight that has no slider.
                base_cfg.setdefault("rewards", {}).update(rewards)
                base_cfg.setdefault("scenarios", {}).update(scenarios)
                if "hyperparameters" not in base_cfg:
                    base_cfg["hyperparameters"] = {}
                base_cfg["hyperparameters"]["bc_regularization_weight"] = float(bc_w)
                base_cfg["hyperparameters"]["bc_decay_steps"] = int(bc_dec)
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass
            return f"✅ **All Curriculum & Reward Dials Applied at {time.strftime('%H:%M:%S')}!**"

        apply_all_curriculum_btn.click(
            fn=on_apply_curriculum,
            inputs=[
                goal_slider, concede_slider, save_slider,
                ball_to_goal_slider, player_to_ball_slider, jump_bridge_slider, air_roll_recovery_slider, powerslide_slider, touch_slider,
                boost_gain_slider, boost_lose_slider,
                kickoff_prob_slider, replay_prob_slider, aerial_prob_slider, custom_prob_slider,
                turnaround_prob_slider, wall_prob_slider, wall_rebound_prob_slider, save_prob_slider, dribble_flick_prob_slider,
                bc_weight_slider, bc_decay_input
            ],
            outputs=[curriculum_apply_msg]
        )

        # Dynamic 100% Normalized Scenario Rebalancing Handler (9 Scenario Mix)
        def rebalance_scenarios_handler(changed_idx, new_val, k, r, a, c, tr, w, wr, s, df):
            current_vals = [float(k), float(r), float(a), float(c), float(tr), float(w), float(wr), float(s), float(df)]
            new_val = round(max(0.0, min(1.0, float(new_val))), 2)
            vals = list(current_vals)
            vals[changed_idx] = new_val

            rem = round(1.0 - new_val, 4)
            other_indices = [i for i in range(9) if i != changed_idx]
            other_sum = sum(current_vals[i] for i in other_indices)

            if other_sum > 0.0001:
                scale = rem / other_sum
                for i in other_indices:
                    vals[i] = round(current_vals[i] * scale, 2)
            else:
                even = round(rem / len(other_indices), 2)
                for i in other_indices:
                    vals[i] = even

            # Snap rounding error to first available other index
            tot = sum(vals)
            diff = round(1.0 - tot, 2)
            if abs(diff) > 0.0001:
                for idx in other_indices:
                    if vals[idx] + diff >= 0:
                        vals[idx] = round(vals[idx] + diff, 2)
                        break

            pct_total = int(round(sum(vals) * 100))
            badge_html = f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                <span class="status-badge-running" style="font-size: 1.0em; padding: 6px 16px;">● Total Mix: {pct_total}%</span>
            </div>
            """

            # Save dynamically to live_config.json so active training updates without lag
            try:
                sc_dict = {
                    "kickoff_prob": vals[0],
                    "replay_prob": vals[1],
                    "aerial_prob": vals[2],
                    "custom_prob": vals[3],
                    "turnaround_prob": vals[4],
                    "wall_prob": vals[5],
                    "wall_rebound_prob": vals[6],
                    "save_prob": vals[7],
                    "dribble_flick_prob": vals[8]
                }
                TrainingProcessManager.get_instance().update_live_config({"scenarios": sc_dict})
            except Exception:
                pass

            return tuple(vals) + (badge_html,)

        scenario_sliders = [
            kickoff_prob_slider, replay_prob_slider, aerial_prob_slider, custom_prob_slider,
            turnaround_prob_slider, wall_prob_slider, wall_rebound_prob_slider, save_prob_slider,
            dribble_flick_prob_slider
        ]
        rebalance_outputs = scenario_sliders + [scenario_total_badge]

        # Use .release() instead of .change() so dragging sliders is instant in-browser without processing flicker!
        for i, sld in enumerate(scenario_sliders):
            sld.release(
                fn=lambda *args, idx=i: rebalance_scenarios_handler(idx, args[0], *args[1:]),
                inputs=[sld] + scenario_sliders,
                outputs=rebalance_outputs
            )

        # Popover Auto-Balance Confirm Handler
        def on_popover_confirm(
            lock_k, val_k, lock_r, val_r, lock_a, val_a, lock_c, val_c,
            lock_tr, val_tr, lock_w, val_w, lock_wr, val_wr, lock_s, val_s,
            lock_df, val_df
        ):
            locks = [bool(lock_k), bool(lock_r), bool(lock_a), bool(lock_c), bool(lock_tr), bool(lock_w), bool(lock_wr), bool(lock_s), bool(lock_df)]
            raw_vals = [
                float(val_k or 0) / 100.0, float(val_r or 0) / 100.0, float(val_a or 0) / 100.0, float(val_c or 0) / 100.0,
                float(val_tr or 0) / 100.0, float(val_w or 0) / 100.0, float(val_wr or 0) / 100.0, float(val_s or 0) / 100.0,
                float(val_df or 0) / 100.0
            ]
            names = ["Kickoff", "Replay", "High Aerial", "Custom", "Turnaround", "Wall Play", "Wall Rebound", "Goalie Save", "Dribble & Flick"]

            locked_sum = sum(raw_vals[i] for i in range(9) if locks[i])
            unlocked_indices = [i for i in range(9) if not locks[i]]

            final_vals = list(raw_vals)
            if locked_sum > 1.0:
                scale = 1.0 / locked_sum
                for i in range(9):
                    final_vals[i] = round(raw_vals[i] * scale, 2) if locks[i] else 0.0
                tot = sum(final_vals)
                diff = round(1.0 - tot, 2)
                if abs(diff) > 0.0001:
                    first_l = next(i for i in range(9) if locks[i])
                    final_vals[first_l] = round(final_vals[first_l] + diff, 2)
                note = f"⚠️ Locked weights exceeded 100% (was {int(round(locked_sum * 100))}%)! Scaled down proportionally to 100%."
            else:
                rem = round(1.0 - locked_sum, 4)
                if unlocked_indices:
                    even = round(rem / len(unlocked_indices), 2)
                    for i in unlocked_indices:
                        final_vals[i] = even
                    tot = sum(final_vals)
                    diff = round(1.0 - tot, 2)
                    if abs(diff) > 0.0001:
                        final_vals[unlocked_indices[0]] = round(final_vals[unlocked_indices[0]] + diff, 2)

                    locked_names = [f"**{names[i]} ({int(round(final_vals[i] * 100))}%)**" for i in range(9) if locks[i]]
                    unlocked_names = [f"{names[i]} ({int(round(final_vals[i] * 100))}%)" for i in unlocked_indices]
                    if locked_names:
                        note = f"✅ **Auto-Balanced!** Locked: {', '.join(locked_names)}. Remaining **{int(round(rem * 100))}%** evenly distributed across: {', '.join(unlocked_names)}."
                    else:
                        note = f"✅ **Auto-Balanced!** No locks checked — distributed equally across all 9 scenarios ({int(round(100 / 9))}% each)."
                else:
                    tot = sum(final_vals)
                    diff = round(1.0 - tot, 2)
                    if abs(diff) > 0.0001:
                        final_vals[0] = round(final_vals[0] + diff, 2)
                    note = "✅ **All 9 scenarios locked** (Total: 100%)."

            pct_total = int(round(sum(final_vals) * 100))
            badge_html = f"""
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                <span class="status-badge-running" style="font-size: 1.0em; padding: 6px 16px;">● Total Mix: {pct_total}%</span>
            </div>
            """

            # Save dynamically to live config and default config
            try:
                sc_dict = {
                    "kickoff_prob": final_vals[0],
                    "replay_prob": final_vals[1],
                    "aerial_prob": final_vals[2],
                    "custom_prob": final_vals[3],
                    "turnaround_prob": final_vals[4],
                    "wall_prob": final_vals[5],
                    "wall_rebound_prob": final_vals[6],
                    "save_prob": final_vals[7],
                    "dribble_flick_prob": final_vals[8]
                }
                TrainingProcessManager.get_instance().update_live_config({"scenarios": sc_dict})
                base_cfg = load_yaml_config("config/default_config.yaml")
                base_cfg["scenarios"] = sc_dict
                save_yaml_config(base_cfg, "config/default_config.yaml")
            except Exception:
                pass

            new_pop_nums = [int(round(v * 100)) for v in final_vals]
            return tuple(final_vals) + (badge_html,) + tuple(new_pop_nums) + (note,)

        pop_inputs = [
            pop_lock_k, pop_val_k,
            pop_lock_r, pop_val_r,
            pop_lock_a, pop_val_a,
            pop_lock_c, pop_val_c,
            pop_lock_tr, pop_val_tr,
            pop_lock_w, pop_val_w,
            pop_lock_wr, pop_val_wr,
            pop_lock_s, pop_val_s,
            pop_lock_df, pop_val_df
        ]
        pop_val_outputs = [
            pop_val_k, pop_val_r, pop_val_a, pop_val_c,
            pop_val_tr, pop_val_w, pop_val_wr, pop_val_s,
            pop_val_df
        ]

        pop_sync_btn.click(
            fn=lambda *sl_vals: tuple(int(round(float(v) * 100)) for v in sl_vals),
            inputs=scenario_sliders,
            outputs=pop_val_outputs
        )

        pop_confirm_btn.click(
            fn=on_popover_confirm,
            inputs=pop_inputs,
            outputs=scenario_sliders + [scenario_total_badge] + pop_val_outputs + [popover_status_msg]
        )

        # Reset Rewards to Balanced Defaults
        def on_reset_rewards():
            def_cfg = load_yaml_config("config/default_config.yaml")
            rew = def_cfg.get("rewards", {})
            sc = def_cfg.get("scenarios", {})
            badge_html = """
            <div style="display: flex; justify-content: flex-end; align-items: center; height: 100%;">
                <span class="status-badge-running" style="font-size: 1.0em; padding: 6px 16px;">● Total Mix: 100%</span>
            </div>
            """
            return (
                rew.get("goal_weight", 20.0),
                rew.get("concede_weight", -20.0),
                rew.get("save_weight", 3.0),
                rew.get("ball_to_goal_weight", 1.5),
                rew.get("player_to_ball_weight", 0.6),
                rew.get("jump_bridge_weight", 0.35),
                rew.get("air_roll_recovery_weight", 0.10),
                rew.get("powerslide_weight", 0.20),
                rew.get("touch_weight", 1.2),
                rew.get("boost_gain_weight", 0.6),
                rew.get("boost_lose_weight", 0.3),
                sc.get("kickoff_prob", 0.20),
                sc.get("replay_prob", 0.15),
                sc.get("aerial_prob", 0.11),
                sc.get("custom_prob", 0.15),
                sc.get("turnaround_prob", 0.13),
                sc.get("wall_prob", 0.07),
                sc.get("wall_rebound_prob", 0.08),
                sc.get("save_prob", 0.07),
                sc.get("dribble_flick_prob", 0.08),
                badge_html,
                "🔄 **Reset dials to balanced standard configuration.**"
            )

        reset_curriculum_btn.click(
            fn=on_reset_rewards,
            outputs=[
                goal_slider, concede_slider, save_slider,
                ball_to_goal_slider, player_to_ball_slider, jump_bridge_slider, air_roll_recovery_slider, powerslide_slider, touch_slider,
                boost_gain_slider, boost_lose_slider,
                kickoff_prob_slider, replay_prob_slider, aerial_prob_slider, custom_prob_slider,
                turnaround_prob_slider, wall_prob_slider, wall_rebound_prob_slider, save_prob_slider, dribble_flick_prob_slider,
                scenario_total_badge,
                curriculum_apply_msg
            ]
        )

        # -------------------------------------------------------------
        # CUSTOM SCENARIO GENERATOR HANDLERS
        # -------------------------------------------------------------
        def assemble_scenario_payload(
            s_id, s_name, s_enabled, s_desc,
            c_x, c_y, c_z, c_yaw, c_spd, c_boost,
            b_x, b_y, b_z, b_vx, b_vy, b_vz,
            o_mode, o_boost, o_x, o_y, o_yaw,
            p_jit, v_jit, mirror
        ) -> dict:
            yaw_rad = math.radians(float(c_yaw))
            spd = float(c_spd)
            car_vel = [spd * math.cos(yaw_rad), spd * math.sin(yaw_rad), 0.0]

            opp_dict = {
                "mode": str(o_mode),
                "boost": float(o_boost)
            }
            if o_mode == "custom":
                opp_dict["pos"] = [float(o_x), float(o_y), 17.0]
                opp_dict["yaw"] = float(o_yaw)
                opp_dict["vel"] = [0.0, 0.0, 0.0]

            return {
                "id": str(s_id).strip(),
                "name": str(s_name).strip(),
                "enabled": bool(s_enabled),
                "description": str(s_desc).strip(),
                "car": {
                    "pos": [float(c_x), float(c_y), float(c_z)],
                    "vel": car_vel,
                    "yaw": float(c_yaw),
                    "boost": float(c_boost)
                },
                "ball": {
                    "pos": [float(b_x), float(b_y), float(b_z)],
                    "vel": [float(b_vx), float(b_vy), float(b_vz)]
                },
                "opponent": opp_dict,
                "variance": {
                    "pos_jitter": float(p_jit),
                    "vel_jitter": float(v_jit),
                    "mirror_symmetry": bool(mirror)
                }
            }

        def on_update_visual_preview(*args):
            sc = assemble_scenario_payload(*args)
            return render_scenario_visual_guide(sc)

        all_sc_inputs = [
            sc_id_input, sc_name_input, sc_enabled_cb, sc_desc_input,
            car_pos_x, car_pos_y, car_pos_z, car_yaw, car_speed, car_boost,
            ball_pos_x, ball_pos_y, ball_pos_z, ball_vel_x, ball_vel_y, ball_vel_z,
            opp_mode_radio, opp_boost, opp_pos_x, opp_pos_y, opp_yaw,
            pos_jitter, vel_jitter, mirror_symmetry
        ]

        for comp in [car_pos_x, car_pos_y, car_pos_z, car_yaw, car_speed, car_boost,
                     ball_pos_x, ball_pos_y, ball_pos_z, ball_vel_x, ball_vel_y, ball_vel_z,
                     opp_mode_radio, opp_pos_x, opp_pos_y, opp_yaw]:
            comp.change(fn=on_update_visual_preview, inputs=all_sc_inputs, outputs=[sc_preview_plot])

        refresh_preview_btn.click(fn=on_update_visual_preview, inputs=all_sc_inputs, outputs=[sc_preview_plot])

        def on_scenario_opp_mode_change(mode):
            return gr.Row(visible=(mode == "custom"))

        opp_mode_radio.change(fn=on_scenario_opp_mode_change, inputs=[opp_mode_radio], outputs=[opp_custom_row])

        # Preset Selector Callback
        def on_select_preset_template(preset_name):
            if not preset_name or preset_name == "(Select Template Preset...)":
                return (gr.update(),) * 23
            match = next((s for s in DEFAULT_CUSTOM_SCENARIOS if s["name"] == preset_name), None)
            if not match:
                return (gr.update(),) * 23

            c = match["car"]
            b = match["ball"]
            o = match.get("opponent", {})
            v = match.get("variance", {})

            spd = math.hypot(c["vel"][0], c["vel"][1])
            o_pos = o.get("pos", [0, 4800, 17])

            return (
                f"{match['id']}_{int(time.time()) % 1000}",
                f"{match['name']} (Custom)",
                True,
                match.get("description", ""),
                c["pos"][0], c["pos"][1], c["pos"][2],
                c.get("yaw", 90.0), spd, c.get("boost", 50.0),
                b["pos"][0], b["pos"][1], b["pos"][2],
                b["vel"][0], b["vel"][1], b["vel"][2],
                o.get("mode", "goalie"), o.get("boost", 60.0),
                o_pos[0], o_pos[1], o.get("yaw", -90.0),
                v.get("pos_jitter", 80.0), v.get("vel_jitter", 60.0), v.get("mirror_symmetry", True)
            )

        preset_dropdown.change(
            fn=on_select_preset_template,
            inputs=[preset_dropdown],
            outputs=all_sc_inputs
        )

        # Save Scenario Callback
        def on_save_custom_scenario(*args):
            sc = assemble_scenario_payload(*args)
            if not sc["id"]:
                return "❌ Error: Scenario ID cannot be empty.", build_scenarios_table(), gr.Dropdown()
            sc_mgr.save_scenario(sc)
            all_scs = sc_mgr.get_all_scenarios()
            choices = [f"{s['name']} ({s['id']})" for s in all_scs]
            sel = f"{sc['name']} ({sc['id']})"
            return (
                f"✅ **Saved custom scenario '{sc['name']}' ({sc['id']})!** Added to active training pool.",
                build_scenarios_table(),
                gr.Dropdown(choices=choices, value=sel)
            )

        save_scenario_btn.click(
            fn=on_save_custom_scenario,
            inputs=all_sc_inputs,
            outputs=[scenario_action_msg, saved_scenarios_table, load_scenario_dropdown]
        )

        # New Scenario Form Callback
        def on_new_scenario_form():
            nid = f"custom_drill_{int(time.time()) % 10000}"
            return (
                nid, "New Custom Drill", True, "User custom drill description.",
                0.0, -2500.0, 17.0, 90.0, 500.0, 50.0,
                0.0, 0.0, 93.15, 0.0, 0.0, 0.0,
                "goalie", 50.0, 0.0, 4800.0, -90.0,
                80.0, 60.0, True,
                "✨ Cleared form. Design your scenario and click **Save Custom Scenario**."
            )

        new_scenario_btn.click(
            fn=on_new_scenario_form,
            outputs=all_sc_inputs + [scenario_action_msg]
        )

        # Delete Scenario Callback
        def on_delete_custom_scenario(sc_id):
            if not sc_id:
                return "⚠️ No scenario selected to delete.", build_scenarios_table(), gr.Dropdown()
            success = sc_mgr.delete_scenario(str(sc_id).strip())
            all_scs = sc_mgr.get_all_scenarios()
            choices = [f"{s['name']} ({s['id']})" for s in all_scs]
            sel = choices[0] if choices else None
            msg = f"🗑️ **Deleted scenario '{sc_id}'.**" if success else f"⚠️ Scenario '{sc_id}' could not be deleted."
            return msg, build_scenarios_table(), gr.Dropdown(choices=choices, value=sel)

        delete_scenario_btn.click(
            fn=on_delete_custom_scenario,
            inputs=[sc_id_input],
            outputs=[scenario_action_msg, saved_scenarios_table, load_scenario_dropdown]
        )

        # Load Scenario from Library Callback
        def on_load_scenario_from_library(selected_choice):
            if not selected_choice:
                return (gr.update(),) * 23
            try:
                sc_id = selected_choice.split("(")[-1].rstrip(")").strip()
            except Exception:
                sc_id = selected_choice
            match = sc_mgr.get_scenario(sc_id)
            if not match:
                return (gr.update(),) * 23

            c = match["car"]
            b = match["ball"]
            o = match.get("opponent", {})
            v = match.get("variance", {})
            spd = math.hypot(c["vel"][0], c["vel"][1])
            o_pos = o.get("pos", [0, 4800, 17])

            return (
                match["id"],
                match["name"],
                match.get("enabled", True),
                match.get("description", ""),
                c["pos"][0], c["pos"][1], c["pos"][2],
                c.get("yaw", 90.0), spd, c.get("boost", 50.0),
                b["pos"][0], b["pos"][1], b["pos"][2],
                b["vel"][0], b["vel"][1], b["vel"][2],
                o.get("mode", "goalie"), o.get("boost", 60.0),
                o_pos[0], o_pos[1], o.get("yaw", -90.0),
                v.get("pos_jitter", 80.0), v.get("vel_jitter", 60.0), v.get("mirror_symmetry", True)
            )

        load_scenario_btn.click(
            fn=on_load_scenario_from_library,
            inputs=[load_scenario_dropdown],
            outputs=all_sc_inputs
        )

        refresh_library_btn.click(
            fn=lambda: (build_scenarios_table(), gr.Dropdown(choices=[f"{s['name']} ({s['id']})" for s in sc_mgr.get_all_scenarios()])),
            outputs=[saved_scenarios_table, load_scenario_dropdown]
        )

        # 2-Second Trajectory Rollout Simulation Callback
        def on_run_scenario_simulation(*args):
            sc = assemble_scenario_payload(*args)
            pts = get_available_checkpoints()
            active_ckpt = pts[0] if pts and not pts[0].startswith("checkpoints/latest_model.pt (none") else None
            res = simulate_custom_scenario(sc, model_path=active_ckpt, num_steps=150)
            if isinstance(res, dict):
                return res.get("plot"), res.get("stats")
            elif isinstance(res, (tuple, list)):
                return res[0], res[1]
            return None, {}

        sim_scenario_btn.click(
            fn=on_run_scenario_simulation,
            inputs=all_sc_inputs,
            outputs=[sc_sim_plot, sc_sim_stats]
        )

        # -------------------------------------------------------------
        # TAB 3 CONFIG & PRETRAINER HANDLERS
        # -------------------------------------------------------------
        def on_save_yaml(lr, ent, clip, gamma, gae, bs, mbs, n_ep, n_env, t_skip, m_steps, g_mode, autosave_int, ckpt_int, archive_str):
            base_cfg = load_yaml_config("config/default_config.yaml")
            base_cfg["hyperparameters"] = {
                "learning_rate": float(lr),
                "ent_coef": float(ent),
                "clip_range": float(clip),
                "gamma": float(gamma),
                "gae_lambda": float(gae),
                "batch_size": int(bs),
                "mini_batch_size": int(mbs),
                "n_epochs": int(n_ep),
            }
            base_cfg["environment"] = {
                "num_envs": int(n_env),
                "tick_skip": int(t_skip),
                "max_episode_steps": int(m_steps),
                "game_mode": str(g_mode),
            }
            if "logging" not in base_cfg:
                base_cfg["logging"] = {}
            base_cfg["logging"]["tensorboard"] = True
            base_cfg["logging"]["save_dir"] = "checkpoints"
            base_cfg["logging"]["log_dir"] = "logs"
            base_cfg["logging"]["autosave_interval"] = max(1, int(autosave_int))
            base_cfg["logging"]["checkpoint_interval"] = max(1, int(ckpt_int))
            base_cfg["logging"]["archive_stride"] = max(0, int(archive_str))

            save_yaml_config(base_cfg)
            return f"✅ **Saved configuration to config/default_config.yaml**"

        save_cfg_btn.click(
            fn=on_save_yaml,
            inputs=[
                lr_input, ent_coef_slider, clip_range_slider, gamma_slider,
                gae_lambda_slider, batch_size_input, mini_batch_input, n_epochs_input,
                num_envs_slider, tick_skip_slider, max_steps_input, game_mode_dropdown,
                autosave_interval_input, checkpoint_interval_input, archive_stride_input
            ],
            outputs=[cfg_save_msg]
        )

        # Replay Scanner Callbacks
        def on_scan_demos(demo_dir, max_replays, sort_mode):
            p = ReplayParser(demo_dir=str(demo_dir).strip())
            files = p.scan_demos(max_replays=int(max_replays), sort=str(sort_mode))
            rows = []
            for fp in files:
                try:
                    sz = round(os.path.getsize(fp) / 1024, 1)
                    mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(fp)))
                    rows.append([os.path.basename(fp), sz, mtime])
                except Exception:
                    rows.append([os.path.basename(fp), 0.0, "Unknown"])
            status_txt = f"🔍 Discovered **{len(rows)}** `.replay` files in `{demo_dir}`."
            return rows, status_txt

        scan_demos_btn.click(
            fn=on_scan_demos,
            inputs=[demos_dir_input, max_replays_slider, sort_replays_radio],
            outputs=[demos_table, replays_status_box]
        )

        def on_ingest_replays(demo_dir, max_replays, sort_mode):
            p = ReplayParser(demo_dir=str(demo_dir).strip())
            res = p.ingest_directory(max_replays=int(max_replays), sort=str(sort_mode))
            stats_md = build_replay_stats_md()
            if res['total_frames'] > 0:
                msg = f"⚡ Ingested **{res['parsed_files']}** replays ({res['total_frames']:,} frames) into dataset pool in {res['elapsed_seconds']:.2f}s."
            else:
                rep = getattr(p, "last_ingest_report", {})
                rej = rep.get("rejected_files", [])
                if rej:
                    msg = f"⚠️ Ingest scanned {rep.get('total_files', 0)} files in {res['elapsed_seconds']:.2f}s, but 0 frames were ingested. {len(rej)} file(s) could not be decoded (e.g. corrupt or incompatible .replay format)."
                else:
                    msg = f"⚠️ No valid replay files (.replay, .npz, .json) found in `{demo_dir}`."
            return stats_md, msg

        ingest_selected_btn.click(
            fn=on_ingest_replays,
            inputs=[demos_dir_input, max_replays_slider, sort_replays_radio],
            outputs=[replay_stats_box, replays_status_box]
        )

        def on_ingest_all_replays(demo_dir):
            p = ReplayParser(demo_dir=str(demo_dir).strip())
            res = p.ingest_directory(max_replays=999999, sort="newest")
            stats_md = build_replay_stats_md()
            if res['total_frames'] > 0:
                msg = f"📥 Ingested ALL **{res['parsed_files']}** replays ({res['total_frames']:,} frames) in {res['elapsed_seconds']:.2f}s."
            else:
                rep = getattr(p, "last_ingest_report", {})
                rej = rep.get("rejected_files", [])
                if rej:
                    msg = f"⚠️ Ingest scanned {rep.get('total_files', 0)} files in {res['elapsed_seconds']:.2f}s, but 0 frames were ingested. {len(rej)} file(s) failed decoding."
                else:
                    msg = f"⚠️ No valid replay files found in `{demo_dir}`."
            return stats_md, msg

        ingest_all_btn.click(
            fn=on_ingest_all_replays,
            inputs=[demos_dir_input],
            outputs=[replay_stats_box, replays_status_box]
        )

        def on_clear_replays():
            p = ReplayParser()
            p.clear_pool()
            stats_md = build_replay_stats_md()
            return stats_md, "🗑️ Replay dataset pool cleared."

        clear_pool_btn.click(
            fn=on_clear_replays,
            outputs=[replay_stats_box, replays_status_box]
        )

        # Upload Ingest Callback
        def on_upload_ingest(uploaded_files):
            if not uploaded_files:
                return build_replay_stats_md(), "⚠️ No files uploaded."
            p = ReplayParser()
            total_added = 0
            total_frames = 0
            file_paths = [f.name if hasattr(f, "name") else str(f) for f in uploaded_files]
            for fp in file_paths:
                ext = os.path.splitext(fp)[1].lower()
                if ext == ".zip":
                    parsed_count, frames_count = p.ingest_zip(fp)
                    total_added += parsed_count
                    total_frames += frames_count
                else:
                    dest = os.path.join(p.demo_dir, os.path.basename(fp))
                    try:
                        import shutil
                        shutil.copy2(fp, dest)
                        total_added += 1
                    except Exception:
                        pass
            if total_frames == 0 and total_added > 0:
                res = p.ingest_directory(max_replays=total_added, sort="newest")
                total_frames = res.get("total_frames", 0)
            stats_md = build_replay_stats_md()
            if total_frames > 0:
                msg = f"📤 Successfully Ingested **{total_added}** replay(s) (**{total_frames:,}** genuine frames) into dataset pool."
            else:
                rep = getattr(p, "last_ingest_report", {})
                rej = rep.get("rejected_files", [])
                if rej:
                    rej_sample = ", ".join(rej[:3])
                    if len(rej) > 3:
                        rej_sample += f" (+{len(rej)-3} more)"
                    msg = f"⚠️ Uploaded files processed, but **0 frames** could be extracted. {len(rej)} file(s) failed decoding ({rej_sample}). Check that the files are uncorrupted Rocket League replays."
                else:
                    msg = "⚠️ Uploaded files yielded 0 frames. Please upload valid Rocket League match replays (.replay, .npz, or .json) or archives (.zip)."
            return stats_md, msg

        replay_uploader.upload(
            fn=on_upload_ingest,
            inputs=[replay_uploader],
            outputs=[replay_stats_box, upload_status_box]
        )

        # BC Pretrainer Callbacks
        def on_run_pretraining(epochs, lr, batch_size, base_ckpt):
            chosen_ckpt = base_ckpt.split(" ")[0] if base_ckpt and not base_ckpt.startswith("checkpoints/latest_model.pt (none") else None
            res = bc_trainer.train(
                epochs=int(epochs),
                batch_size=int(batch_size),
                lr=float(lr),
                base_checkpoint=chosen_ckpt
            )
            raw_msg = res.get("message", "Pretraining finished.")
            success = res.get("success", True)
            color = "#4ade80" if success else "#f87171"
            title = "COMPLETED" if success else "FAILED"
            return f"""
            <div class="status-callout-box" style="border-left-color: {color};">
                <span style="color: {color}; font-weight: 700; margin-right: 8px;">{title}:</span>
                <span>{raw_msg}</span>
            </div>
            """

        def on_stop_pretraining():
            if bc_trainer.is_running():
                bc_trainer.request_stop()
                return """
                <div class="status-callout-box" style="border-left-color: #f87171;">
                    <span style="color: #f87171; font-weight: 700; margin-right: 8px;">STOPPED:</span>
                    <span>Imitation pretrainer stop requested.</span>
                </div>
                """
            return """
            <div class="status-callout-box" style="border-left-color: #94a3b8;">
                <span style="color: #94a3b8; font-weight: 700; margin-right: 8px;">IDLE:</span>
                <span>Pretrainer is not currently running.</span>
            </div>
            """

        run_pretrain_btn.click(
            fn=on_run_pretraining,
            inputs=[pretrain_epochs_slider, pretrain_lr_input, pretrain_batch_dropdown, pretrain_base_dropdown],
            outputs=[pretrain_status_box]
        )
        stop_pretrain_btn.click(
            fn=on_stop_pretraining,
            outputs=[pretrain_status_box]
        )

        # -------------------------------------------------------------
        # TAB 4: DIAGNOSTICS & EVALUATION HANDLERS
        # -------------------------------------------------------------
        def on_run_unit_tests():
            res = run_all_unit_tests(verbose=True)
            res_md = format_test_results_markdown(res)
            return res_md, res.get("raw_output", "")

        run_unit_tests_btn.click(
            fn=on_run_unit_tests,
            outputs=[unit_tests_overview_md, unit_tests_stdout]
        )

        def on_scan_checkpoints():
            ckpts = get_available_checkpoints()
            return gr.Dropdown(choices=ckpts, value=ckpts[0] if ckpts else None), gr.Dropdown(choices=ckpts, value=ckpts[0] if ckpts else None)

        refresh_ckpts_btn.click(fn=on_scan_checkpoints, outputs=[ckpt_dropdown, orange_ckpt_dropdown])

        def on_opp_mode_change(mode):
            return gr.Dropdown(visible=(mode == "Another Checkpoint"))

        opponent_mode.change(fn=on_opp_mode_change, inputs=[opponent_mode], outputs=[orange_ckpt_dropdown])

        def on_run_simulation(blue_choice, opp_mode, orange_choice, steps):
            blue_path = blue_choice.split(" ")[0] if blue_choice else None
            if not blue_path or not os.path.exists(blue_path):
                blue_path = "checkpoints/latest_model.pt" if os.path.exists("checkpoints/latest_model.pt") else None

            orange_path = "same_as_blue"
            if opp_mode == "Self-Play (Bot vs Itself)":
                orange_path = "same_as_blue"
            elif opp_mode == "Another Checkpoint":
                orange_path = orange_choice.split(" ")[0] if orange_choice else None
                if not orange_path or not os.path.exists(orange_path):
                    orange_path = "checkpoints/latest_model.pt" if os.path.exists("checkpoints/latest_model.pt") else "baseline"
            elif opp_mode == "Baseline Bot (Chase Ball Heuristic)":
                orange_path = "baseline"

            res = simulate_match(
                blue_model_path=blue_path,
                orange_model_path=orange_path,
                max_steps=int(steps)
            )

            if isinstance(res, dict):
                p_fig = res.get("plot")
                r_fig = res.get("reward_plot")
                stats = res.get("stats", {})
            else:
                p_fig, r_fig, stats = res

            total_s = stats.get("simulation_steps", stats.get("total_steps", int(steps)))
            b_goals = stats.get("blue_goals", stats.get("goals_blue", 0))
            o_goals = stats.get("orange_goals", stats.get("goals_orange", 0))
            b_touches = stats.get("blue_touches", stats.get("touches_blue", 0))
            o_touches = stats.get("orange_touches", stats.get("touches_orange", 0))
            b_rew = stats.get("blue_total_reward", stats.get("rewards_blue", 0.0))
            o_rew = stats.get("orange_total_reward", stats.get("rewards_orange", 0.0))

            summary_md = f"""
            #### 📊 Headless Match Simulation Results
            * **Simulated Duration:** `{total_s}` steps ({total_s/15.0:.1f}s match time)
            * **Score:** Blue **{b_goals}** - **{o_goals}** Orange
            * **Blue Ball Touches:** **{b_touches}** | **Orange Ball Touches:** **{o_touches}**
            * **Blue Net Reward:** `{b_rew:+.2f}` | **Orange Net Reward:** `{o_rew:+.2f}`
            """
            return p_fig, r_fig, summary_md

        run_sim_btn.click(
            fn=on_run_simulation,
            inputs=[ckpt_dropdown, opponent_mode, orange_ckpt_dropdown, sim_steps_slider],
            outputs=[visualizer_plot, reward_breakdown_plot, sim_stats_box]
        )

        def on_refresh_diagnostics(window_size):
            telem = extract_rolling_telemetry("logs/history.jsonl", window=int(window_size))
            live_cfg = mgr.get_live_config() if hasattr(mgr, "get_live_config") else {}
            if not live_cfg and os.path.exists("config/live_config.json"):
                try:
                    with open("config/live_config.json", "r") as f:
                        live_cfg = json.load(f)
                except Exception:
                    live_cfg = {}
            active_rewards = live_cfg.get("rewards", {})
            coach_md = generate_ai_coach_diagnostics(telem, active_rewards=active_rewards)
            action_fig = render_action_biases_plot(telem)
            pos_fig = render_positional_biases_plot(telem)
            return coach_md, action_fig, pos_fig

        refresh_diag_btn.click(
            fn=on_refresh_diagnostics,
            inputs=[diag_window_slider],
            outputs=[diag_coach_report, diag_action_plot, diag_position_plot]
        )

        def on_refresh_full_diagnostics():
            overview_md, export_box = build_full_diagnostic_export()
            return overview_md, export_box

        refresh_snapshot_btn.click(
            fn=on_refresh_full_diagnostics,
            outputs=[diag_overview_md, diag_export_raw]
        )

        # -------------------------------------------------------------
        # TRUESKILL TOURNAMENT & LEADERBOARD HANDLERS
        # -------------------------------------------------------------
        def on_ts_select_all():
            ckpts = get_available_checkpoints()
            return gr.Dropdown(value=ckpts)

        def on_ts_clear_all():
            return gr.Dropdown(value=[])

        def on_ts_refresh_ckpts():
            ckpts = get_available_checkpoints()
            return gr.Dropdown(choices=ckpts)

        def on_ts_reset_leaderboard():
            ts_evaluator.reset_leaderboard()
            return (
                "#### 🗑️ TrueSkill leaderboard reset successfully.",
                ts_evaluator.get_leaderboard_dataframe(),
                ts_evaluator.render_leaderboard_plot()
            )

        def on_run_ts_tournament(ckpts, anchors, series_per_pair, steps, enable_ot):
            model_list = list(ckpts or [])
            for a in (anchors or []):
                if "heuristic" in a.lower():
                    model_list.append("heuristic")
                elif "pretrained" in a.lower() and os.path.exists("checkpoints/pretrained_baseline.pt"):
                    model_list.append("checkpoints/pretrained_baseline.pt")
                elif "necto" in a.lower() and os.path.exists("checkpoints/necto-model.pt"):
                    model_list.append("checkpoints/necto-model.pt")
                elif "nexto" in a.lower() and os.path.exists("checkpoints/nexto-model.pt"):
                    model_list.append("checkpoints/nexto-model.pt")

            model_list = list(dict.fromkeys([os.path.normpath(m).replace("\\", "/") for m in model_list if m]))
            if len(model_list) < 2:
                yield (
                    "#### ⚠️ Error: Please select at least 2 contestants for the tournament.",
                    ts_evaluator.get_leaderboard_dataframe(),
                    ts_evaluator.render_leaderboard_plot()
                )
                return

            yield (
                f"#### ⚔️ Initializing tournament with {len(model_list)} models...",
                ts_evaluator.get_leaderboard_dataframe(),
                ts_evaluator.render_leaderboard_plot()
            )

            for update in ts_evaluator.run_tournament(
                model_paths=model_list,
                series_per_pair=max(1, int(series_per_pair)),
                max_steps=int(steps),
                device="cpu"
            ):
                p_idx = update["pairing_index"]
                total_p = update["total_pairings"]
                mA = update["model_a"]
                mB = update["model_b"]
                res_list = update["results"]

                summary_parts = []
                for r in res_list:
                    ot = " (OT)" if r["overtime"] else ""
                    summary_parts.append(f"{r['blue_name']} {r['blue_goals']}-{r['orange_goals']} {r['orange_name']}{ot}")

                status_md = f"""
                #### ⚔️ Tournament Progress: Matchup {p_idx}/{total_p}
                * **Pairing:** `{mA}` vs `{mB}`
                * **Results:** {', '.join(summary_parts)}
                """
                df = ts_evaluator.get_leaderboard_dataframe()
                fig = ts_evaluator.render_leaderboard_plot()
                yield status_md, df, fig

            final_md = f"#### 🏆 Tournament Complete! All {len(model_list)} models ranked."
            yield final_md, ts_evaluator.get_leaderboard_dataframe(), ts_evaluator.render_leaderboard_plot()

        ts_select_all_btn.click(fn=on_ts_select_all, outputs=[ts_ckpt_multiselect])
        ts_clear_all_btn.click(fn=on_ts_clear_all, outputs=[ts_ckpt_multiselect])
        ts_refresh_ckpts_btn.click(fn=on_ts_refresh_ckpts, outputs=[ts_ckpt_multiselect])
        reset_leaderboard_btn.click(
            fn=on_ts_reset_leaderboard,
            outputs=[ts_status_md, ts_leaderboard_table, ts_leaderboard_plot]
        )
        run_tournament_btn.click(
            fn=on_run_ts_tournament,
            inputs=[ts_ckpt_multiselect, ts_anchors_checkbox, ts_matches_slider, ts_steps_slider, ts_overtime_check],
            outputs=[ts_status_md, ts_leaderboard_table, ts_leaderboard_plot]
        )

        def on_refresh_cockpit_leaderboard():
            ts_evaluator.load_leaderboard()
            return (
                build_cockpit_leaderboard_summary_html(ts_evaluator),
                build_league_wire_and_queue_html(ts_evaluator),
                get_cockpit_leaderboard_df(ts_evaluator)
            )

        refresh_cockpit_lb_btn.click(
            fn=on_refresh_cockpit_leaderboard,
            outputs=[cockpit_lb_summary, cockpit_league_ticker, cockpit_lb_table]
        )

        # -------------------------------------------------------------
        # REAL-TIME BACKGROUND REFRESH TIMER & INITIAL LOAD
        # -------------------------------------------------------------
        _last_history_mtime = [0.0]
        _last_history_size = [0]
        _last_log_str = [""]
        _last_view_mode = ["Recent 100"]
        _last_leaderboard_mtime = [0.0]
        _last_league_mtime = [0.0]

        def on_timer_tick(view_mode: str = "Recent 100"):
            status = mgr.get_status_info()
            card_html = build_status_card_html(status)
            running = status.get("running", False)
            paused = status.get("paused", False)
            start_btn_update = gr.update(
                value="🚀 Start Training" if not running else "🟢 Training Active",
                variant="primary" if not running else "secondary",
                interactive=not running
            )
            pause_btn_update = gr.update(
                value="▶️ Resume Training" if paused else "⏸️ Pause Training",
                variant="primary" if paused else "secondary",
                interactive=running
            )
            stop_btn_update = gr.update(
                value="🛑 Stop Training",
                variant="stop" if running else "secondary",
                interactive=running
            )

            # Smart Log Update: zero network overhead if logs haven't changed
            logs = mgr.get_logs()
            if logs == _last_log_str[0]:
                logs_update = gr.update()
            else:
                _last_log_str[0] = logs
                logs_update = logs

            # Smart Plot Update: zero CPU / zero memory overhead if history hasn't been appended to
            history_file = "logs/history.jsonl"
            curr_mtime = os.path.getmtime(history_file) if os.path.exists(history_file) else 0.0
            curr_size = os.path.getsize(history_file) if os.path.exists(history_file) else 0
            mode_param = "full" if "full" in str(view_mode).lower() else "recent"

            if (curr_mtime == _last_history_mtime[0] and
                curr_size == _last_history_size[0] and
                view_mode == _last_view_mode[0]):
                plot_update = gr.update()
            else:
                _last_history_mtime[0] = curr_mtime
                _last_history_size[0] = curr_size
                _last_view_mode[0] = view_mode
                plot_update = render_training_curves_plot(history_file=history_file, mode=mode_param)

            # Smart Leaderboard Update: zero overhead if leaderboard JSON and league state haven't changed
            lb_file = "logs/trueskill_leaderboard.json"
            league_file = "logs/league_state.json"
            curr_lb_mtime = os.path.getmtime(lb_file) if os.path.exists(lb_file) else 0.0
            curr_league_mtime = os.path.getmtime(league_file) if os.path.exists(league_file) else 0.0

            if curr_lb_mtime == _last_leaderboard_mtime[0] and curr_league_mtime == _last_league_mtime[0]:
                lb_summary_update = gr.update()
                lb_wire_update = gr.update()
                lb_table_update = gr.update()
            else:
                _last_leaderboard_mtime[0] = curr_lb_mtime
                _last_league_mtime[0] = curr_league_mtime
                ts_evaluator.load_leaderboard()
                lb_summary_update = build_cockpit_leaderboard_summary_html(ts_evaluator)
                lb_wire_update = build_league_wire_and_queue_html(ts_evaluator)
                lb_table_update = get_cockpit_leaderboard_df(ts_evaluator)

            return (
                card_html, start_btn_update, pause_btn_update, stop_btn_update,
                logs_update, plot_update, lb_summary_update, lb_wire_update, lb_table_update
            )

        def on_change_view_mode(mode_val):
            mode_param = "full" if "full" in str(mode_val).lower() else "recent"
            _last_view_mode[0] = mode_val
            return render_training_curves_plot(mode=mode_param)

        metrics_window_radio.change(fn=on_change_view_mode, inputs=[metrics_window_radio], outputs=[live_metrics_plot])
        refresh_metrics_btn.click(fn=on_change_view_mode, inputs=[metrics_window_radio], outputs=[live_metrics_plot])
        refresh_logs_btn.click(fn=mgr.get_logs, outputs=[console_output])
        clear_logs_btn.click(fn=lambda: "", outputs=[console_output])

        status_timer = gr.Timer(3.0, active=True)
        status_timer.tick(
            fn=on_timer_tick,
            inputs=[metrics_window_radio],
            outputs=[
                status_card, start_btn, pause_btn, stop_btn,
                console_output, live_metrics_plot,
                cockpit_lb_summary, cockpit_league_ticker, cockpit_lb_table
            ]
        )

        # Initialize UI on page load
        demo.load(
            fn=on_timer_tick,
            inputs=[metrics_window_radio],
            outputs=[
                status_card, start_btn, pause_btn, stop_btn,
                console_output, live_metrics_plot,
                cockpit_lb_summary, cockpit_league_ticker, cockpit_lb_table
            ]
        )

    return demo
