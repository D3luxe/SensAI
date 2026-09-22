"""
Pooling eval results at a decision point: the table every judgement in docs/ is read off.

A single checkpoint's reading is not a decision (docs/reward_v5_gamma_spec.md §4): head to head has
a per-seed sd near 6.7, and adjacent checkpoints have differed by 25 goals per 10 min. Pooling takes
every seed of every result in a window, so the standard error shrinks with the number of checkpoints
rather than being hidden by the choice of one.

Goals-against causes are pooled as goals per 10 min, not as shares: a share moves when the total
conceded moves, which reads as a change in the cause that never happened.

  pick_results(...)    the results of a run in a window (last N, or around a decision point)
  row_values(...)      one result's columns, causes already scaled to goals per 10 min
  pooled(...)          the pooled row, the head-to-head seeds, and their standard error
  text_report(...)     what scripts/pool_evals.py prints
  table_html(...)      the same table for the Evaluation tab
"""
from __future__ import annotations

import html
import math
import statistics as st
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.eval_results import list_results, load

# (group, metric, header, scale by goals against -> goals per 10 min)
COLUMNS: List[Tuple[str, str, str, bool]] = [
    ("necto", "goals_for_per_10min", "GF", False),
    ("necto", "goals_against_per_10min", "GA", False),
    ("necto", "goal_diff_per_10min", "nGD", False),
    ("nexto", "goal_diff_per_10min", "xGD", False),
    ("necto", "touches_per_min", "tpm", False),
    ("necto", "on_target_per_100_touches", "onT", False),
    ("necto", "kickoff_goals_against", "koGA", False),
    ("necto", "ga_cause_caught_upfield_pct", "upfld", True),
    ("necto", "ga_cause_back_wall_climb_pct", "bwGA", True),
    ("necto", "ga_cause_beaten_goal_side_pct", "beatGS", True),
    ("scenario_retreat", "conceded_pct", "retC", False),
    ("scenario_retreat", "reached_goalside_pct", "retGS", False),
]
BOOST_COLUMNS: List[Tuple[str, str, str, bool]] = [
    ("necto", "boost_empty_pct", "empty%", False),
    ("necto", "boost_collected_per_min", "collect", False),
    ("necto", "big_pads_per_min", "bigPad", False),
    ("necto", "retreat_starts_low_boost_pct", "retLow", False),
]
# Columns whose header reads better with a direction reminder in the UI
TITLES = {"GF": "goals for /10 min", "GA": "goals against /10 min", "nGD": "Necto goal difference /10 min",
          "xGD": "Nexto goal difference /10 min", "tpm": "touches per minute",
          "onT": "on target per 100 touches", "koGA": "kickoff goals against",
          "upfld": "goals against, caught upfield /10 min", "bwGA": "goals against, back-wall climb /10 min",
          "beatGS": "goals against while goal-side /10 min", "retC": "retreat scenario: conceded %",
          "retGS": "retreat scenario: reached goal-side %", "empty%": "time on empty boost %",
          "collect": "boost collected per minute", "bigPad": "big pads per minute",
          "retLow": "retreats begun under 12 boost %"}


def columns_for(show_boost: bool) -> List[Tuple[str, str, str, bool]]:
    return COLUMNS + (BOOST_COLUMNS if show_boost else [])


def value(res: Dict[str, Any], group: str, metric: str) -> Optional[float]:
    s = (res.get("results") or {}).get(group, {}).get(metric)
    if s is None:
        return None
    return s["mean"] if isinstance(s, dict) else s


def row_values(res: Dict[str, Any], columns: Sequence[Tuple[str, str, str, bool]]) -> List[Optional[float]]:
    out = []
    for group, metric, _, scaled in columns:
        v = value(res, group, metric)
        if v is not None and scaled:
            ga = value(res, group, "goals_against_per_10min")
            v = None if ga is None else v * ga / 100.0
        out.append(v)
    return out


def head_to_head_seeds(res: Dict[str, Any]) -> List[float]:
    s = (res.get("results") or {}).get("reference", {}).get("goal_diff_per_10min")
    return list(s.get("per_seed", [])) if s else []


def pick_results(version: Optional[str] = None, eval_dir: str = "evals", last: Optional[int] = None,
                 around: Optional[float] = None, window: float = 10.0,
                 paths: Optional[Sequence[str]] = None) -> List[Tuple[str, Dict[str, Any]]]:
    """(name, result) in step order: explicit paths, or a version's run filtered by --last / --around."""
    if paths:
        return [(p.replace("\\", "/").split("/")[-1], load(p)) for p in paths]
    entries = [e for e in list_results(eval_dir)
               if e["version"] == version and not e["quick"] and e["run_steps_m"] is not None]
    entries.sort(key=lambda e: e["run_steps_m"])
    if around is not None:
        entries = [e for e in entries if abs(e["run_steps_m"] - around) <= window]
    if last:
        entries = entries[-last:]
    return [(e["name"], load(e["path"])) for e in entries]


def pooled(chosen: Sequence[Tuple[str, Dict[str, Any]]], columns: Sequence[Tuple[str, str, str, bool]]) -> Dict[str, Any]:
    """The pooled row over every result, and the head-to-head seeds of all of them together."""
    seeds: List[float] = []
    stacks: List[List[float]] = [[] for _ in columns]
    rows = []
    for name, res in chosen:
        s = head_to_head_seeds(res)
        seeds += s
        vals = row_values(res, columns)
        for i, v in enumerate(vals):
            if v is not None:
                stacks[i].append(v)
        h2h = (res.get("results") or {}).get("reference", {}).get("goal_diff_per_10min")
        rows.append({"name": name, "iteration": res.get("iteration"), "seeds": s,
                     "h2h": h2h["mean"] if h2h else None, "values": vals})
    se = st.stdev(seeds) / math.sqrt(len(seeds)) if len(seeds) > 1 else float("nan")
    mean = st.mean(seeds) if seeds else float("nan")
    return {"rows": rows, "values": [st.mean(v) if v else None for v in stacks], "seeds": seeds,
            "h2h": mean, "se": se, "positive": sum(x > 0 for x in seeds),
            "sigmas": (abs(mean / se) if seeds and se and not math.isnan(se) and se > 0 else float("nan"))}


def _fmt(v: Optional[float], width: int = 6) -> str:
    return f"{v:{width}.1f}" if v is not None else " " * (width - 1) + "-"


def text_report(title: str, chosen, show_boost: bool = False) -> str:
    columns = columns_for(show_boost)
    p = pooled(chosen, columns)
    out = [f"== {title}: {len(chosen)} result(s)",
           f"{'result':>14} {'iter':>7} {'h2h':>7} {'seeds':>26} " + " ".join(f"{h:>6}" for _, _, h, _ in columns)]
    for r in p["rows"]:
        out.append(f"{r['name'][:14]:>14} {r['iteration'] or 0:>7} "
                   f"{(r['h2h'] if r['h2h'] is not None else float('nan')):7.2f} "
                   f"{','.join(f'{x:.1f}' for x in r['seeds']):>26} "
                   + " ".join(_fmt(v) for v in r["values"]))
    if chosen:
        note = f"n={len(p['seeds'])} se={p['se']:.2f} +ve={p['positive']}"
        out.append(f"{'POOLED':>14} {'':>7} {p['h2h']:7.2f} {note:>26} " + " ".join(_fmt(v) for v in p["values"]))
        if not math.isnan(p["sigmas"]):
            out.append(f"{'':>14} head to head is {p['sigmas']:.1f} standard errors "
                       f"{'above' if p['h2h'] > 0 else 'below'} zero")
    return "\n".join(out)


def table_html(title: str, chosen, show_boost: bool = False, note: str = "") -> str:
    """The pooled table for the Evaluation tab. Pooled row last, head to head first after the name."""
    columns = columns_for(show_boost)
    if not chosen:
        return (f"<div class='pool-wrap'><div class='pool-empty'>No results in this window yet. "
                f"Run the eval suite, or follow the run from the panel on the left.</div></div>")
    p = pooled(chosen, columns)
    head = "".join(f"<th title='{html.escape(TITLES.get(h, h))}'>{html.escape(h)}</th>" for _, _, h, _ in columns)
    body = []
    for r in p["rows"]:
        seeds = ", ".join(f"{x:+.1f}" for x in r["seeds"]) or "–"
        h2h = f"{r['h2h']:+.2f}" if r["h2h"] is not None else "–"
        cls = "pos" if (r["h2h"] or 0) > 0 else ("neg" if r["h2h"] is not None and r["h2h"] < 0 else "")
        body.append(f"<tr><td class='pool-name'>{html.escape(r['name'])}</td>"
                    f"<td class='pool-h2h {cls}'>{h2h}</td><td class='pool-seeds'>{html.escape(seeds)}</td>"
                    + "".join(f"<td>{_fmt(v).strip()}</td>" for v in r["values"]) + "</tr>")
    cls = "pos" if p["h2h"] > 0 else "neg"
    sigmas = "" if math.isnan(p["sigmas"]) else (f"{p['sigmas']:.1f} SE {'above' if p['h2h'] > 0 else 'below'} zero")
    body.append(f"<tr class='pool-total'><td class='pool-name'>Pooled</td>"
                f"<td class='pool-h2h {cls}'>{p['h2h']:+.2f}</td>"
                f"<td class='pool-seeds'>{len(p['seeds'])} seeds · {p['positive']} positive · "
                f"se {p['se']:.2f}{' · ' + sigmas if sigmas else ''}</td>"
                + "".join(f"<td>{_fmt(v).strip()}</td>" for v in p["values"]) + "</tr>")
    return (f"<div class='pool-wrap'><div class='pool-title'>{html.escape(title)}"
            f"<span class='pool-note'>{html.escape(note)}</span></div>"
            f"<table class='pool-table'><thead><tr><th>Result</th><th>h2h</th><th>seeds</th>{head}</tr></thead>"
            f"<tbody>{''.join(body)}</tbody></table>"
            f"<div class='pool-foot'>Head to head is goal difference per 10 min against the version's start "
            f"checkpoint; every seed of every result is pooled. Goals-against causes are goals per 10 min, not "
            f"shares. Hover a column for what it measures.</div></div>")


POOL_CSS = """
.pool-wrap { background: #111c2e; border: 1px solid #1e293b; border-radius: 10px; padding: 10px 12px; }
.pool-title { color: #e2e8f0; font-weight: 600; font-size: 13px; margin-bottom: 6px; }
.pool-note { color: #64748b; font-weight: 400; margin-left: 8px; font-size: 12px; }
.pool-table { width: 100%; border-collapse: collapse; font-size: 12px; color: #cbd5e1; }
.pool-table th { text-align: right; color: #94a3b8; font-weight: 500; padding: 4px 6px; border-bottom: 1px solid #1e293b; cursor: help; }
.pool-table td { text-align: right; padding: 4px 6px; border-bottom: 1px solid #16233a; font-variant-numeric: tabular-nums; }
.pool-table th:first-child, .pool-table td.pool-name { text-align: left; }
.pool-table td.pool-seeds, .pool-table th:nth-child(3) { text-align: left; color: #64748b; white-space: nowrap; }
.pool-h2h.pos { color: #34d399; } .pool-h2h.neg { color: #f87171; }
.pool-total td { border-top: 1px solid #334155; border-bottom: none; font-weight: 600; color: #e2e8f0; }
.pool-total td.pool-seeds { font-weight: 400; color: #94a3b8; }
.pool-empty { color: #94a3b8; font-size: 13px; padding: 18px 6px; }
.pool-foot { color: #64748b; font-size: 11px; margin-top: 8px; line-height: 1.5; }
"""
