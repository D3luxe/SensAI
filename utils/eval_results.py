"""
Eval-suite results (evals/*.json, written by scripts/eval_suite.py): loading, comparing and the
views the UI's Evaluation page shows.

Comparison uses the suite's own noise rule, the one --compare prints: a match metric differs when
the two results' ranges over seeds do not overlap; a scenario rate differs when the gap exceeds two
binomial standard errors. Everything else is inside the noise and is shown as such.

HEADLINE lists the metrics docs/reward_v3_spec.md section 6 judges a run on, grouped as the spec
groups them (primary, guardrails, the problems v2 had), each with the direction that is better.
"""
from __future__ import annotations

import glob
import html
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

EVAL_DIR = "evals"

# (group, metric, label, better: +1 higher / -1 lower)
HEADLINE: Dict[str, List[Tuple[str, str, str, int]]] = {
    "Primary": [
        ("necto", "goal_diff_per_10min", "Goal difference vs Necto (per 10 min)", +1),
    ],
    "Guardrails": [
        ("necto", "touches_per_min", "Touches per minute", +1),
        ("necto", "kickoff_first_touch_pct", "Kickoff first touch", +1),
        ("necto", "shot_conversion_pct", "Shot conversion", +1),
    ],
    "The problems v2 had": [
        ("necto", "retreat_boosting_pct", "Boosting while retreating", +1),
        ("necto", "retreat_dodges_per_100_touches", "Retreat dodges per 100 touches", -1),
        ("necto", "retreat_dodge_not_wheels_down_pct", "Retreat dodges not landed on wheels", -1),
        ("scenario_drop", "ran_under_ball_pct", "Ran under a dropping ball", -1),
        ("scenario_bounce", "sensai_first_touch_pct", "First touch on a bounce", +1),
        ("scenario_retreat", "conceded_pct", "Conceded in the retreat scenario", -1),
        ("necto", "ga_cause_back_wall_climb_pct", "Goals against: back-wall climb", -1),
        ("necto", "ga_cause_caught_upfield_pct", "Goals against: caught upfield", -1),
    ],
}

GROUP_LABELS = {
    "necto": "Matches vs Necto", "nexto": "Matches vs Nexto", "reference": "Matches vs reference",
    "scenario_drop": "Scenario: dropping ball", "scenario_bounce": "Scenario: bouncing ball",
    "scenario_wall": "Scenario: wall", "scenario_retreat": "Scenario: retreat",
}

# Direction for metrics outside HEADLINE; anything unlisted is shown without a better/worse colour.
_BETTER = {
    "goals_for": +1, "goals_against": -1, "goal_diff_per_10min": +1, "goals_for_per_10min": +1,
    "goals_against_per_10min": -1, "touches_per_min": +1, "on_target_per_100_touches": +1,
    "shot_conversion_pct": +1, "whiffs_per_100_touches": -1, "aimed_whiffs_per_100_touches": -1,
    "overshoots_aimed_per_100_touches": -1, "kickoff_first_touch_pct": +1, "kickoff_goals_for": +1,
    "kickoff_goals_against": -1, "retreat_dodges_per_100_touches": -1,
    "retreat_dodge_not_wheels_down_pct": -1, "backwall_climbs_per_100_touches": -1,
    "boost_empty_pct": -1, "retreat_boosting_pct": +1, "landing_not_wheels_down_pct": -1,
    "landing_speed_kept_median": +1, "sensai_first_touch_pct": +1, "opp_first_touch_pct": -1,
    "time_to_first_touch_s": -1, "ran_under_ball_pct": -1, "touch_goalward_pct": +1,
    "opp_next_touch_pct": -1, "conceded_pct": -1, "scored_pct": +1, "bad_landing_pct": -1,
    "reached_goalside_pct": +1, "time_to_goalside_s": -1, "conceded_pct_facing_home": -1,
    "conceded_pct_facing_away": -1,
}
_GA_CAUSES = ("caught_upfield", "back_wall_climb", "goal_side_far", "beaten_goal_side", "own_touch_last", "off_the_kickoff")
for _c in _GA_CAUSES:
    _BETTER[f"ga_cause_{_c}_pct"] = -1


def better_direction(metric: str) -> int:
    return _BETTER.get(metric, 0)


def humanize(metric: str) -> str:
    s = metric.replace("_per_10min", " per 10 min").replace("_per_100_touches", " per 100 touches")
    s = s.replace("_per_min", " per min").replace("_pct", " %").replace("ga_cause_", "goals against: ")
    if s.endswith("_s"):
        s = s[:-2] + " (s)"
    return s.replace("_", " ").strip().capitalize()


# ---------------------------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------------------------
def list_results(eval_dir: str = EVAL_DIR) -> List[Dict[str, Any]]:
    """Every result file with its metadata, newest first."""
    out = []
    for p in glob.glob(os.path.join(eval_dir, "*.json")) + glob.glob(os.path.join(eval_dir, "baselines", "*.json")):
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        stamp = d.get("reward_identity") if isinstance(d.get("reward_identity"), dict) else {}
        start = d.get("reward_run_start_step")
        steps = d.get("global_step")
        out.append({
            "path": p.replace("\\", "/"),
            "name": ("baseline " if os.path.basename(os.path.dirname(p)) == "baselines" else "")
                    + os.path.splitext(os.path.basename(p))[0],
            "version": stamp.get("version") or "unstamped",
            "iteration": d.get("iteration"),
            "run_steps_m": (steps - start) / 1e6 if (steps is not None and start is not None) else None,
            "quick": bool(d.get("quick")),
            "created": d.get("created", ""),
            "suite_version": d.get("suite_version"),
            "mtime": os.path.getmtime(p),
        })
    return sorted(out, key=lambda r: r["mtime"], reverse=True)


def load(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def label_for(entry: Dict[str, Any]) -> str:
    parts = [entry["name"]]
    if entry.get("run_steps_m") is not None:
        parts.append(f"{entry['version']} +{entry['run_steps_m']:.0f}M")
    elif entry.get("iteration") is not None:
        parts.append(f"iter {entry['iteration']:,}")
    if entry.get("quick"):
        parts.append("quick")
    return " · ".join(parts)


# ---------------------------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------------------------
def compare_metric(group: str, metric: str, sa: Dict[str, Any], sb: Dict[str, Any], n: Optional[float]) -> Dict[str, Any]:
    """One metric, A -> B, with the suite's noise rule."""
    a, b = sa["mean"], sb["mean"]
    d = b - a
    clear = False
    if len(sa.get("per_seed", [])) > 1 and len(sb.get("per_seed", [])) > 1:
        clear = sb["min"] > sa["max"] or sb["max"] < sa["min"]
    elif metric.endswith("_pct") and n:
        pa, pb = a / 100.0, b / 100.0
        if not (math.isnan(pa) or math.isnan(pb)):
            se = math.sqrt(max(1e-9, pa * (1 - pa) / n + pb * (1 - pb) / n)) * 100.0
            clear = not math.isnan(d) and abs(d) > 2.0 * se
    direction = better_direction(metric)
    verdict = "noise"
    if clear and not math.isnan(d) and d != 0:
        verdict = "neutral" if direction == 0 else ("better" if d * direction > 0 else "worse")
    return {"group": group, "metric": metric, "a": a, "b": b, "delta": d, "clear": clear, "verdict": verdict,
            "a_range": (sa.get("min"), sa.get("max")), "b_range": (sb.get("min"), sb.get("max")),
            "seeds": max(len(sa.get("per_seed", [])), len(sb.get("per_seed", [])))}


def compare_results(a: Dict[str, Any], b: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every metric both results carry, A -> B."""
    rows = []
    for g, ma in a.get("results", {}).items():
        mb = b.get("results", {}).get(g)
        if not mb:
            continue
        n = (ma.get("n") or {}).get("mean")
        for k, sa in ma.items():
            sb = mb.get(k)
            if sb is None or k == "n":
                continue
            rows.append(compare_metric(g, k, sa, sb, n))
    return rows


def headline_rows(a: Optional[Dict[str, Any]], b: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """HEADLINE metrics of B, compared with A when given."""
    out = {}
    for section, items in HEADLINE.items():
        rows = []
        for g, k, label, _ in items:
            sb = b.get("results", {}).get(g, {}).get(k)
            if sb is None:
                continue
            if a is not None and a.get("results", {}).get(g, {}).get(k) is not None:
                n = (b["results"][g].get("n") or {}).get("mean")
                row = compare_metric(g, k, a["results"][g][k], sb, n)
            else:
                row = {"group": g, "metric": k, "a": None, "b": sb["mean"], "delta": None, "clear": False,
                       "verdict": "none", "b_range": (sb.get("min"), sb.get("max")), "seeds": len(sb.get("per_seed", []))}
            row["label"] = label
            rows.append(row)
        out[section] = rows
    return out


# ---------------------------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------------------------
def _num(v: Optional[float], metric: str = "") -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "–"
    if metric.endswith("_pct"):
        return f"{v:.0f}%" if abs(v) >= 10 else f"{v:.1f}%"
    return f"{v:.0f}" if abs(v) >= 100 else (f"{v:.1f}" if abs(v) >= 10 else f"{v:.2f}")


def _delta(v: Optional[float], metric: str) -> str:
    if v is None or math.isnan(v):
        return ""
    s = _num(abs(v), metric).rstrip("%")
    unit = " pts" if metric.endswith("_pct") else ""
    return f"{'+' if v >= 0 else '−'}{s}{unit}"


VERDICT_TEXT = {"better": "better", "worse": "worse", "noise": "within noise", "neutral": "changed", "none": ""}


def scorecard_html(result: Dict[str, Any], baseline: Optional[Dict[str, Any]], result_name: str,
                   baseline_name: Optional[str]) -> str:
    """The spec's judging view: primary metric, guardrails and v2's problem metrics as cards."""
    sections = headline_rows(baseline, result)
    stamp = result.get("reward_identity") if isinstance(result.get("reward_identity"), dict) else {}
    meta = [f"<b>{html.escape(result_name)}</b>",
            f"reward {html.escape(str(stamp.get('version', 'unstamped')))}",
            f"iteration {result.get('iteration', '?'):,}" if isinstance(result.get("iteration"), int) else "",
            f"{len(result.get('seeds', []))} match seeds, {result.get('scenario_trials', '?')} trials per scenario",
            "<span class='ev-warn'>quick run: a smoke test, not a result</span>" if result.get("quick") else ""]
    head = " · ".join(m for m in meta if m)
    vs = (f"compared with <b>{html.escape(baseline_name)}</b>" if baseline is not None
          else "no baseline selected")
    if baseline is not None and baseline.get("suite_version") != result.get("suite_version"):
        vs += " <span class='ev-warn'>(different suite versions: not comparable)</span>"
    parts = [f"<div class='ev-meta'>{head}<br><span class='ev-sub'>{vs}. Differences inside the "
             f"seed spread or two binomial standard errors are shown as noise.</span></div>"]
    for section, rows in sections.items():
        cards = []
        for r in rows:
            m = r["metric"]
            verdict = r["verdict"]
            rng = r.get("b_range") or (None, None)
            spread = (f"<div class='ev-spread'>seeds {_num(rng[0], m)} – {_num(rng[1], m)}</div>"
                      if r.get("seeds", 0) > 1 else "")
            base = (f"<div class='ev-base'>was {_num(r['a'], m)} <span class='ev-d ev-{verdict}'>{_delta(r['delta'], m)} · "
                    f"{VERDICT_TEXT[verdict]}</span></div>" if r["a"] is not None else "")
            big = " ev-card-primary" if section == "Primary" else ""
            cards.append(f"<div class='ev-card ev-edge-{verdict}{big}'><div class='ev-label'>{html.escape(r['label'])}</div>"
                         f"<div class='ev-value'>{_num(r['b'], m)}</div>{base}{spread}</div>")
        parts.append(f"<div class='ev-section'><div class='ev-section-title'>{html.escape(section)}</div>"
                     f"<div class='ev-grid'>{''.join(cards)}</div></div>")
    return "<div class='ev-scorecard'>" + "".join(parts) + "</div>"


def comparison_table_html(a: Dict[str, Any], b: Dict[str, Any], only_clear: bool = False) -> str:
    rows = compare_results(a, b)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        if only_clear and not r["clear"]:
            continue
        groups.setdefault(r["group"], []).append(r)
    if not groups:
        return "<div class='ev-empty'>No metric differs beyond the noise.</div>" if only_clear else \
               "<div class='ev-empty'>These results share no metrics.</div>"
    out = ["<div class='ev-tables'>"]
    for g, rs in groups.items():
        body = "".join(
            f"<tr class='ev-row-{r['verdict']}'><td>{html.escape(humanize(r['metric']))}</td>"
            f"<td class='ev-n'>{_num(r['a'], r['metric'])}</td><td class='ev-n'>{_num(r['b'], r['metric'])}</td>"
            f"<td class='ev-n ev-d ev-{r['verdict']}'>{_delta(r['delta'], r['metric'])}</td>"
            f"<td class='ev-v ev-{r['verdict']}'>{VERDICT_TEXT[r['verdict']]}</td></tr>" for r in rs)
        out.append(f"<div class='ev-table-wrap'><div class='ev-section-title'>{html.escape(GROUP_LABELS.get(g, g))}</div>"
                   f"<table class='ev-table'><thead><tr><th>Metric</th><th>A</th><th>B</th><th>Δ</th><th></th></tr></thead>"
                   f"<tbody>{body}</tbody></table></div>")
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------------------------------------
# Trend over a run
# ---------------------------------------------------------------------------------------------
TREND = [
    ("necto", "goal_diff_per_10min", "Goal difference vs Necto / 10 min  (higher is better)"),
    ("necto", "kickoff_first_touch_pct", "Kickoff first touch %  (higher is better)"),
    ("necto", "touches_per_min", "Touches per minute  (higher is better)"),
    ("scenario_retreat", "conceded_pct", "Retreat scenario: conceded %  (lower is better)"),
    ("necto", "retreat_boosting_pct", "Boosting while retreating %  (higher is better)"),
    ("scenario_drop", "ran_under_ball_pct", "Ran under a dropping ball %  (lower is better)"),
]
DECISION_POINTS_M = (250, 500)


def trend_figure(version: str, baseline: Optional[Dict[str, Any]], eval_dir: str = EVAL_DIR):
    """Headline metrics of every full eval of `version`, by millions of steps into its run."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts = []
    for e in list_results(eval_dir):
        if e["version"] == version and e["run_steps_m"] is not None and not e["quick"]:
            pts.append((e["run_steps_m"], load(e["path"])))
    pts.sort(key=lambda x: x[0])

    fig, axes = plt.subplots(2, 3, figsize=(12, 5.4), dpi=100)
    fig.patch.set_facecolor("#0f172a")
    for ax, (g, k, title) in zip(axes.flat, TREND):
        ax.set_facecolor("#111c2e")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color("#334155")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.grid(True, linestyle=":", alpha=0.3, color="#64748b")
        ax.set_title(title, color="#e2e8f0", fontsize=9.5, loc="left")
        xmax = max([p[0] for p in pts] + [DECISION_POINTS_M[-1] * 1.05])
        for dp in DECISION_POINTS_M:
            ax.axvline(dp, color="#a78bfa", linewidth=1, linestyle="--", alpha=0.6)
        if baseline is not None and baseline.get("results", {}).get(g, {}).get(k):
            bv = baseline["results"][g][k]
            ax.axhline(bv["mean"], color="#94a3b8", linewidth=1.2, linestyle="-", alpha=0.8)
            if bv.get("min") is not None and len(bv.get("per_seed", [])) > 1:
                ax.axhspan(bv["min"], bv["max"], color="#94a3b8", alpha=0.10)
        xs = [p[0] for p in pts if p[1].get("results", {}).get(g, {}).get(k)]
        ys = [p[1]["results"][g][k] for p in pts if p[1].get("results", {}).get(g, {}).get(k)]
        if xs:
            means = [y["mean"] for y in ys]
            lo = [y["mean"] - y["min"] if y.get("min") is not None else 0 for y in ys]
            hi = [y["max"] - y["mean"] if y.get("max") is not None else 0 for y in ys]
            ax.errorbar(xs, means, yerr=[lo, hi], color="#38bdf8", ecolor="#38bdf8", elinewidth=1, capsize=2,
                        marker="o", markersize=4, linewidth=1.6)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("M steps into the run", color="#64748b", fontsize=8)
    if not pts:
        fig.text(0.5, 0.5, f"No full evals of reward {version} yet.\nRun one every ~50M steps from the panel on the left.",
                 ha="center", va="center", color="#94a3b8", fontsize=11,
                 bbox=dict(facecolor="#0f172a", edgecolor="#334155", boxstyle="round,pad=0.8"))
    fig.text(0.995, 0.005, "grey: baseline (band = seed range) · purple: decision points (250M, 500M)",
             ha="right", va="bottom", color="#64748b", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return fig
