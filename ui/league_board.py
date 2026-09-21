"""
The League tab: King banner, rating ladder, standings, pipeline, head to head, recent series.

Everything is server-rendered HTML and SVG from three files -- the leaderboard (fitted ratings and
tallies), the league state (king, pool, contenders, events, benchmarks) and the results log (every
series, the data the ratings are fitted from). No JavaScript; animations are transform/opacity
only and all of them stop under prefers-reduced-motion.

Rebuilt 2026-09-21 with the batch-fit ratings (utils/rating_fit.py). What changed and why:
  - The scrolling ticker is gone. A marquee is the wrong shape for a feed you want to read; the
    events and the series now sit in two static, newest-first lists.
  - The Gauntlet is shown as a pipeline stage with live counts rather than a panel of trial cards,
    because trials run the moment a checkpoint is graded and the queue is empty nearly all the time.
  - The ladder chart is new: every rated checkpoint by iteration, with its uncertainty, against the
    pinned v3 king and the fitted Necto and Nexto. It is the one view that shows a run's progress.
  - Win probabilities come from the fit's own model, P = Phi(dmu / sqrt(2 beta^2 + s1^2 + s2^2)).
"""
from __future__ import annotations

import datetime
import html
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from utils import rating_fit
from utils.trueskill_evaluator import get_anchor_calibration

BETA = 25.0 / 6.0
RUN_COLOURS = ["#38bdf8", "#a78bfa", "#94a3b8", "#64748b"]     # current run first, then older
KIND_LABEL = {"league": "League", "calibration": "Calibration", "benchmark": "Benchmark",
              "migrated": "Migrated"}


# ---------------------------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------------------------

def _norm(p: str) -> str:
    return str(p or "").replace("\\", "/").lower()


def iteration_of(rec) -> int:
    m = re.search(r"checkpoint_iter_(\d+)", f"{getattr(rec, 'path', '')} {getattr(rec, 'name', '')}".lower())
    return int(m.group(1)) if m else -1


def run_of(rec, current: str) -> str:
    """Which run a checkpoint came from: its archive folder, or the active version if live."""
    path = _norm(getattr(rec, "path", ""))
    m = re.search(r"archive/([^/]+)/", path)
    if m:
        return m.group(1).replace("_run", "")
    if "checkpoint_iter_" in path:
        return current
    return ""


def short_name(rec) -> str:
    it = iteration_of(rec)
    if it >= 0:
        return f"{it:,}".replace(",", " ")
    name = str(getattr(rec, "name", "?"))
    if get_anchor_calibration(getattr(rec, "path", "")) is not None:
        return "v3 king"
    return name.split(" (")[0]


def p_beats(a, b) -> float:
    """The fit's own series win probability for a over b."""
    s = math.sqrt(2 * BETA * BETA + a.sigma ** 2 + b.sigma ** 2)
    return 0.5 * (1.0 + math.erf((a.mu - b.mu) / s / math.sqrt(2.0)))


def ago(ts: Any, now: Optional[datetime.datetime] = None) -> str:
    try:
        t = datetime.datetime.fromisoformat(str(ts))
    except (TypeError, ValueError):
        return ""
    secs = ((now or datetime.datetime.now()) - t).total_seconds()
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


class BoardData:
    """Everything the board reads, loaded once per render."""

    def __init__(self, evaluator, state: Dict[str, Any], current_version: str = ""):
        self.ev = evaluator
        self.state = state or {}
        self.current = current_version
        self.ratings = [r for r in evaluator.ratings.values()
                        if "latest_model" not in _norm(r.path) and "latest_model" not in r.name.lower()]
        self.by_path = {_norm(r.path): r for r in self.ratings}
        path = getattr(evaluator, "results_path", "")
        self.results: List[Dict[str, Any]] = rating_fit.load_results(path) if path else []

        self.pinned = next((r for r in self.ratings if get_anchor_calibration(r.path) is not None), None)
        self.necto = self._ref("necto")
        self.nexto = self._ref("nexto")

        king_path = _norm(self.state.get("king_of_the_hill", ""))
        self.king = self.by_path.get(king_path)
        if self.king is not None and self.king.is_anchor:
            self.king = None

        elite = [self.by_path.get(_norm(p)) for p in self.state.get("elite_pool", [])]
        self.elite = [r for r in elite if r is not None and not r.is_anchor]
        if not self.elite:
            rated = [r for r in self.ratings if not r.is_anchor and iteration_of(r) >= 0
                     and evaluator.is_rank_eligible(r)]
            self.elite = sorted(rated, key=lambda r: -r.mu)[:10]
        self.elite.sort(key=lambda r: (evaluator.ranking_key(r), r.matches_played), reverse=True)

        runs = []
        for r in sorted((r for r in self.ratings if iteration_of(r) >= 0), key=iteration_of, reverse=True):
            run = run_of(r, self.current)
            if run and run not in runs:
                runs.append(run)
        if self.current in runs:
            runs.remove(self.current)
            runs.insert(0, self.current)
        self.runs = runs

    def _ref(self, key: str):
        for r in self.ratings:
            base = os.path.basename(_norm(r.path))
            if key in base and not (key == "necto" and "nexto" in base):
                return r
        return None

    def run_colour(self, run: str) -> str:
        i = self.runs.index(run) if run in self.runs else len(RUN_COLOURS) - 1
        return RUN_COLOURS[min(i, len(RUN_COLOURS) - 1)]

    def form(self, rec, n: int = 5) -> List[Tuple[str, str]]:
        """Last n series results from rec's side, oldest first: ('W'|'L'|'D', tooltip)."""
        key = _norm(rec.path)
        out = []
        for r in reversed(self.results):
            a, b = _norm(r["a"]), _norm(r["b"])
            if key not in (a, b):
                continue
            mine, theirs = (r["a_score"], r["b_score"]) if a == key else (r["b_score"], r["a_score"])
            opp = self.by_path.get(b if a == key else a)
            res = "W" if mine > theirs else ("L" if mine < theirs else "D")
            tip = f"{res} {mine}-{theirs} vs {short_name(opp) if opp else os.path.basename(b if a == key else a)}"
            out.append((res, tip))
            if len(out) == n:
                break
        return list(reversed(out))

    def head_to_head(self, a, b) -> Tuple[float, int]:
        """(a's series score, series played) between two records."""
        ka, kb = _norm(a.path), _norm(b.path)
        score = games = 0.0
        for r in self.results:
            x, y = _norm(r["a"]), _norm(r["b"])
            if {x, y} != {ka, kb}:
                continue
            s = rating_fit.series_score(r["a_score"], r["b_score"])
            score += s if x == ka else 1.0 - s
            games += 1
        return score, int(games)


# ---------------------------------------------------------------------------------------------
# Small pieces
# ---------------------------------------------------------------------------------------------

def _chip(cls: str, text: str, tip: str = "") -> str:
    t = f' title="{html.escape(tip)}"' if tip else ""
    return f'<span class="lb-chip {cls}"{t}>{text}</span>'


def _run_tag(d: BoardData, rec) -> str:
    run = run_of(rec, d.current)
    if not run:
        return ""
    colour = d.run_colour(run)
    return f'<span class="lg-run" style="--run:{colour}">{html.escape(run)}</span>'


def _status_chip(d: BoardData, rec) -> str:
    if d.king is not None and rec is d.king:
        return _chip("lb-chip-king", "&#128081; King")
    if d.ev.is_rank_eligible(rec):
        return _chip("lb-chip-ranked", "Ranked", f"sigma {rec.sigma:.2f} is inside the {d.ev.eligibility_sigma} gate")
    return _chip("lb-chip-provisional", "Provisional", f"sigma {rec.sigma:.2f}: not yet converged")


def _signed(x: float, digits: int = 1) -> str:
    return f"{x:+.{digits}f}".replace("-", "&minus;")


def _pct(p: float) -> str:
    return f"{100 * p:.0f}%" if 0.005 <= p <= 0.995 else ("&lt;1%" if p < 0.005 else "&gt;99%")


# ---------------------------------------------------------------------------------------------
# Scale legend (the toolbar strip)
# ---------------------------------------------------------------------------------------------

def scale_legend_html(evaluator, explainer_html: str = "") -> str:
    d = BoardData(evaluator, {})
    chips = []
    if d.pinned is not None:
        chips.append(f'<span class="lb-anchor-chip lg-pin" title="Pinned: fixes the scale">'
                     f'&#128204; <b>v3 king</b><span class="lb-anchor-mu">&mu; {d.pinned.mu:.0f}</span></span>')
    for ref in (d.necto, d.nexto):
        if ref is None or ref.sigma > 8.0:
            continue
        chips.append(f'<span class="lb-anchor-chip" title="Fitted from every series against it">'
                     f'<b>{html.escape(short_name(ref))}</b>'
                     f'<span class="lb-anchor-mu">&mu; {ref.mu:.1f} &plusmn;{ref.sigma:.1f}</span></span>')
    return f"""
    <div class="lb-anchor-legend">
        {explainer_html}
        <span class="lb-anchor-legend-label">Scale</span>
        {''.join(chips)}
        <span class="lb-anchor-note">one pin, everything else fitted from every series &middot; &mu; 25 = the v3 king</span>
    </div>
    """


# ---------------------------------------------------------------------------------------------
# King banner
# ---------------------------------------------------------------------------------------------

def king_banner_html(evaluator, state: Dict[str, Any], current_version: str = "") -> str:
    d = BoardData(evaluator, state, current_version)
    if not d.ratings:
        return """
        <div class="lb-panel lg-standby">&#8505;&#65039; <b>League standby.</b> No checkpoints graded
        yet. Each numbered checkpoint is graded automatically on save.</div>"""

    series_total = len(d.results)
    now = datetime.datetime.now()
    last_day = sum(1 for r in _timed(d.results) if _within(r.get("at"), now, 86400))
    rated = sum(1 for r in d.ratings if not r.is_anchor and d.ev.is_rank_eligible(r))
    latest = max((iteration_of(r) for r in d.ratings), default=-1)
    strip = f"""
    <div class="lg-strip">
        <div><span>Rated checkpoints</span><b>{rated}</b></div>
        <div><span>Series in the fit</span><b>{series_total:,}</b></div>
        <div><span>Last 24 h</span><b>{last_day:,}</b></div>
        <div><span>Latest checkpoint</span><b>{latest if latest >= 0 else '&mdash;'}</b></div>
        <div><span>Runs on the board</span><b>{' &middot; '.join(html.escape(r) for r in d.runs) or '&mdash;'}</b></div>
    </div>"""

    k = d.king
    if k is None:
        return f"""
        <div class="lb-king lb-king-empty">
            <div class="lb-king-id"><span class="lb-crown lb-crown-dim">&#9876;&#65039;</span>
                <div><div class="lb-king-label">Throne vacant</div>
                <div class="lb-king-name">Awaiting a rated checkpoint</div>
                <div class="lb-king-sub">King and pool tiers run as self-play until one is graded</div></div>
            </div>
        </div>{strip}"""

    since = ""
    for ev in reversed(d.state.get("event_history", [])):
        if ev.get("type") == "coronation" and str(ev.get("model", "")) in (k.name, k.name.split("/")[-1]):
            since = f" &middot; crowned {ago(ev.get('timestamp'))}"
            break

    def stat(label, value, cls="", tip=""):
        t = f' title="{html.escape(tip)}"' if tip else ""
        return f'<div class="lb-stat"{t}><span class="lb-stat-label">{label}</span><span class="lb-stat-value {cls}">{value}</span></div>'

    stats = [stat("Rating", f"{k.mu:.1f}<small> &plusmn;{k.sigma:.1f}</small>", "warn")]
    if d.pinned is not None:
        stats.append(stat("vs v3 king", _signed(k.mu - d.pinned.mu), "good" if k.mu >= d.pinned.mu else "",
                          "mu above the pinned v3 king"))
        stats.append(stat("Beats v3 king", _pct(p_beats(k, d.pinned)), "accent",
                          "the fit's probability of winning a best-of-9 series"))
    if d.necto is not None and d.necto.sigma < 8.0:
        stats.append(stat("Beats Necto", _pct(p_beats(k, d.necto)), "",
                          f"Necto is fitted at {d.necto.mu:.1f} +/- {d.necto.sigma:.1f}; undefeated, so a lower bound"))
    stats.append(stat("Points", f"{k.points_rate:.0f}%", "good", "peer series only; calibration excluded"))

    return f"""
    <div class="lb-king">
        <div class="lb-king-id">
            <span class="lb-crown">&#128081;</span>
            <div style="min-width:0">
                <div class="lb-king-label">King of the Hill</div>
                <div class="lb-king-name">Iteration {short_name(k)} {_run_tag(d, k)}</div>
                <div class="lb-king-sub">{k.wins}W&ndash;{k.losses}L&ndash;{k.draws}D over {k.matches_played} series{since}</div>
            </div>
        </div>
        <div class="lb-king-stats">{''.join(stats)}</div>
    </div>{strip}"""


def _untimed_at(results: List[Dict[str, Any]]) -> set:
    """
    Timestamps that are not when a series was played. scripts/migrate_to_batch_fit.py wrote every
    recovered series -- league ones as "migrated", benchmark ones as "benchmark" -- with the one
    timestamp of the migration, so that timestamp marks them all.
    """
    return {r.get("at") for r in results if r.get("kind") == "migrated"}


def _timed(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Series with a real timestamp of their own."""
    skip = _untimed_at(results)
    return [r for r in results if r.get("at") not in skip]


def _within(ts: Any, now: datetime.datetime, secs: float) -> bool:
    try:
        return (now - datetime.datetime.fromisoformat(str(ts))).total_seconds() <= secs
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------------------------
# Rating ladder chart
# ---------------------------------------------------------------------------------------------

def ladder_chart_html(d: BoardData) -> str:
    pts = [r for r in d.ratings if iteration_of(r) >= 0 and not r.is_anchor and r.sigma < 6.0]
    if len(pts) < 2:
        return ('<div class="lb-empty">&#128200; <b>The ladder needs two rated checkpoints.</b> '
                'Each save is graded on arrival and appears here with its uncertainty.</div>')

    W, H = 1000.0, 300.0
    L, R, T, B = 46.0, 118.0, 14.0, 30.0
    xs = [iteration_of(r) for r in pts]
    x_lo, x_hi = min(xs), max(xs)
    x_span = max(x_hi - x_lo, 1)
    refs = [(d.pinned, "v3 king", "#c4b5fd", True), (d.necto, "Necto", "#fb7185", False),
            (d.nexto, "Nexto", "#34d399", False)]
    refs = [(r, n, c, p) for r, n, c, p in refs if r is not None and r.sigma < 8.0]
    # Provisional ratings carry sigmas of 3-6; letting them set the axis would squash the ranked
    # checkpoints, which are the ones the chart is for. They are drawn, hollow and unwhiskered.
    ranked = [r for r in pts if d.ev.is_rank_eligible(r)] or pts
    ys = ([r.mu - r.sigma for r in ranked] + [r.mu + r.sigma for r in ranked]
          + [r.mu for r in pts] + [r.mu for r, *_ in refs])
    y_lo, y_hi = math.floor(min(ys) - 1), math.ceil(max(ys) + 1)
    y_span = max(y_hi - y_lo, 1)

    def sx(it):
        return L + (W - L - R) * (it - x_lo) / x_span

    def sy(mu):
        return T + (H - T - B) * (1 - (mu - y_lo) / y_span)

    parts = []
    step = 5 if y_span > 20 else 2
    for v in range(int(math.ceil(y_lo / step) * step), int(y_hi) + 1, step):
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" class="bm-grid"/>'
                     f'<text x="{L - 7}" y="{sy(v) + 3:.1f}" text-anchor="end" class="bm-axis-label">{v}</text>')
    for it in (x_lo, (x_lo + x_hi) // 2, x_hi):
        parts.append(f'<text x="{sx(it):.1f}" y="{H - B + 15}" text-anchor="middle" class="bm-axis-label">{it:,}</text>')
    parts.append(f'<text x="{(L + W - R) / 2:.0f}" y="{H - 3}" text-anchor="middle" class="bm-axis-title">TRAINING ITERATION</text>')
    parts.append(f'<text x="12" y="{(T + H - B) / 2:.0f}" text-anchor="middle" class="bm-axis-title" '
                 f'transform="rotate(-90 12 {(T + H - B) / 2:.0f})">RATING &#956;</text>')

    for r, name, colour, pinned in refs:
        y = sy(r.mu)
        if not pinned:
            band = (sy(r.mu - r.sigma) - sy(r.mu + r.sigma))
            parts.append(f'<rect x="{L}" y="{sy(r.mu + r.sigma):.1f}" width="{W - L - R}" height="{band:.1f}" '
                         f'fill="{colour}" opacity="0.07"/>')
        dash = "" if pinned else ' stroke-dasharray="6 5"'
        parts.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="{colour}" '
                     f'stroke-width="{1.6 if pinned else 1.2}"{dash} opacity="0.85"/>')
        label = f"{name} {r.mu:.1f}" + (" &#128204;" if pinned else f" &#177;{r.sigma:.1f}")
        parts.append(f'<text x="{W - R + 8}" y="{y + 3.5:.1f}" class="lg-ref-label" fill="{colour}">{label}</text>')

    elite = {id(r) for r in d.elite}
    order = sorted(pts, key=lambda r: (r is d.king, id(r) in elite))
    for i, r in enumerate(order):
        x, y = sx(iteration_of(r)), sy(r.mu)
        colour = d.run_colour(run_of(r, d.current))
        ranked = d.ev.is_rank_eligible(r) or r is d.king
        tip = (f"Iteration {iteration_of(r):,} ({run_of(r, d.current)}) &#183; &#956; {r.mu:.2f} &#177;{r.sigma:.2f} "
               f"&#183; {r.wins}W-{r.losses}L-{r.draws}D")
        if ranked:
            parts.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{sy(r.mu - r.sigma):.1f}" y2="{sy(r.mu + r.sigma):.1f}" '
                         f'stroke="{colour}" stroke-width="1.2" opacity="0.45"/>')
        delay = f' style="animation-delay:{min(i, 60) * 12}ms"'
        if r is d.king:
            parts.append(f'<g class="lg-king-mark"><circle cx="{x:.1f}" cy="{y:.1f}" r="9" class="lg-king-halo"/>'
                         f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.2" fill="#facc15" stroke="#0b1220" stroke-width="1.2">'
                         f'<title>King &#183; {tip}</title></circle></g>')
        elif id(r) in elite:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.4" fill="{colour}" stroke="#f8fafc" '
                         f'stroke-width="1.2" class="lg-dot"{delay}><title>Elite &#183; {tip}</title></circle>')
        elif ranked:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{colour}" class="lg-dot"{delay}>'
                         f'<title>{tip}</title></circle>')
        else:
            parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="none" stroke="{colour}" '
                         f'stroke-width="1.2" class="lg-dot"{delay}><title>Provisional &#183; {tip}</title></circle>')

    legend = "".join(
        f'<span class="bm-legend-item"><span class="bm-legend-swatch" style="background:{d.run_colour(run)}"></span>'
        f'{html.escape(run)}{" (training)" if run == d.current else ""}</span>' for run in d.runs)
    return f"""
    <div class="bm-chart"><svg class="bm-svg" viewBox="0 0 {W:.0f} {H:.0f}" role="img"
        aria-label="Rating of every rated checkpoint by training iteration">{''.join(parts)}</svg></div>
    <div class="bm-legend">{legend}
        <span class="bm-legend-item"><span class="lg-key-king"></span>King</span>
        <span class="bm-legend-item"><span class="lg-key-elite"></span>Elite pool</span>
        <span class="bm-legend-item"><span class="lg-key-prov"></span>Provisional</span>
        <span class="lb-muted">whiskers &#177;1&#963; on ranked checkpoints &middot; numbering restarts each run</span>
    </div>"""


# ---------------------------------------------------------------------------------------------
# Standings
# ---------------------------------------------------------------------------------------------

def standings_html(d: BoardData) -> str:
    if not d.elite:
        return '<div class="lb-empty">&#128737;&#65039; <b>Pool initialising.</b> Checkpoints appear here as they graduate.</div>'
    lo = min(min(r.mu - r.sigma for r in d.elite), d.pinned.mu if d.pinned else 99) - 0.5
    hi = max(r.mu + r.sigma for r in d.elite) + 0.5
    span = max(hi - lo, 1.0)
    pos = lambda v: 100.0 * (v - lo) / span
    pin = (f'<span class="lg-pin-tick" style="left:{pos(d.pinned.mu):.1f}%" title="v3 king, mu {d.pinned.mu:.0f}"></span>'
           if d.pinned else "")

    head = ('<div class="lg-row lg-row-head"><span>#</span><span>Checkpoint</span><span>Status</span>'
            '<span>Rating &middot; &#177;1&#963; &middot; <i class="lg-pin-key"></i> v3 king</span><span>&#956;</span>'
            '<span class="lb-hide-sm">vs v3</span><span>Pts</span><span class="lb-hide-sm">Form</span>'
            '<span class="lb-hide-sm">GP</span></div>')
    rows = []
    for i, r in enumerate(d.elite, start=1):
        is_king = d.king is not None and r is d.king
        fill = "is-king" if is_king else ("" if d.ev.is_rank_eligible(r) else "is-provisional")
        form = "".join(f'<i class="lg-form lg-form-{res}" title="{html.escape(tip)}">{res}</i>'
                       for res, tip in d.form(r))
        pts = r.points_rate
        pts_colour = "#4ade80" if pts >= 50 else ("#facc15" if pts >= 40 else "#fb7185")
        vs = _signed(r.mu - d.pinned.mu) if d.pinned else "&mdash;"
        rows.append(f"""
        <div class="lg-row{' lg-row-king' if is_king else ''}">
            <span class="lb-rank{' lb-rank-top' if i <= 3 else ''}">{i}</span>
            <span class="lb-name">{short_name(r)} {_run_tag(d, r)}</span>
            <span>{_status_chip(d, r)}</span>
            <span class="lg-range" title="&#956; {r.mu:.2f} &#177;{r.sigma:.2f}">
                {pin}
                <span class="lg-range-ci {fill}" style="left:{pos(r.mu - r.sigma):.1f}%; width:{pos(r.mu + r.sigma) - pos(r.mu - r.sigma):.1f}%"></span>
                <span class="lg-range-mu {fill}" style="left:{pos(r.mu):.1f}%"></span>
            </span>
            <span class="lb-num">{r.mu:.1f}</span>
            <span class="lb-muted lb-hide-sm">{vs}</span>
            <span class="lb-num" style="color:{pts_colour}">{pts:.0f}%</span>
            <span class="lg-forms lb-hide-sm">{form or '<span class="lb-muted">&mdash;</span>'}</span>
            <span class="lb-muted lb-hide-sm">{r.matches_played}</span>
        </div>""")
    gate = d.ev.eligibility_sigma
    return f"""
    <div class="lb-face-note"><b>{len(d.elite)}</b> in the pool &middot; ranked by &#956; behind a &#963; &le; {gate:g} gate
        &middot; form is the last five series, newest right</div>
    <div class="lb-standings">{head}{''.join(rows)}</div>"""


# ---------------------------------------------------------------------------------------------
# Pipeline and events
# ---------------------------------------------------------------------------------------------

_EVENT_STYLE = {
    "coronation": ("&#128081;", "lg-ev-crown", "took the crown"),
    "promotion": ("&#127942;", "lg-ev-up", "graduated to the elite pool"),
    "admission": ("&#9876;&#65039;", "lg-ev-in", "entered the gauntlet"),
    "demotion": ("&#128317;", "lg-ev-down", "dropped out"),
    "preemption": ("&#128260;", "lg-ev-in", "took a gauntlet slot"),
}


def _event_line(ev: Dict[str, Any]) -> str:
    kind = str(ev.get("type", "")).lower()
    icon, cls, verb = _EVENT_STYLE.get(kind, ("&#9889;", "lg-ev-in", kind or "update"))
    extra = ""
    if kind == "promotion" and ev.get("matches"):
        extra = f' <span class="lb-muted">{ev.get("points_rate", 0):.0f}% over {ev["matches"]}</span>'
    elif kind == "coronation" and ev.get("old_king"):
        extra = f' <span class="lb-muted">from {html.escape(str(ev["old_king"]).replace("checkpoint_iter_", ""))}</span>'
    model = html.escape(str(ev.get("model", "?")).replace("checkpoint_iter_", ""))
    return (f'<li class="lg-ev {cls}" title="{html.escape(str(ev.get("detail", "")))}">'
            f'<span class="lg-ev-icon">{icon}</span><span><b>{model}</b> {verb}{extra}</span>'
            f'<span class="lg-ago">{ago(ev.get("timestamp"))}</span></li>')


def pipeline_html(d: BoardData) -> str:
    state = d.state
    queue = state.get("contenders") or []
    latest = max((r for r in d.ratings if iteration_of(r) >= 0 and run_of(r, d.current) == d.current),
                 key=iteration_of, default=None)
    events = state.get("event_history", [])
    last_admit = next((e for e in reversed(events) if e.get("type") == "admission"), None)

    stages = [
        ("Saved", short_name(latest) if latest else "&mdash;", "latest checkpoint this run"),
        ("Gauntlet", str(len(queue)), f"last entry {ago(last_admit.get('timestamp'))}" if last_admit else "no entries yet"),
        ("Elite", str(len(d.elite)), "sparring pool"),
        ("King", short_name(d.king) if d.king else "&mdash;", "25% of league opponents"),
    ]
    flow = "".join(
        f'<div class="lg-stage"><span class="lg-stage-k">{k}</span><b>{v}</b><span class="lg-stage-sub">{s}</span></div>'
        + ('<span class="lg-arrow">&#10140;</span>' if i < len(stages) - 1 else "")
        for i, (k, v, s) in enumerate(stages))

    trials = ""
    if queue:
        cards = []
        for c in queue[:3]:
            played, target = c.get("matches_played", 0), c.get("target_matches", 30)
            pct = min(100.0, 100.0 * played / max(1, target))
            cards.append(f"""
            <div class="lg-trial"><div class="lg-trial-top"><b>{html.escape(str(c.get('name', '?')))}</b>
                <span class="lb-muted">{played}/{target} series &middot; &#963; {c.get('sigma', 8.33):.2f}</span></div>
                <div class="lb-prog-track"><div class="lb-prog-fill" style="width:{pct:.0f}%"></div></div></div>""")
        trials = "".join(cards)
    else:
        trials = ('<div class="lg-note">Queue empty. Every save is graded on arrival and, if it holds up, '
                  'runs its trial straight away, so the gauntlet is usually empty between saves.</div>')

    # The league can log one event twice in a row (a coronation re-announced on the next
    # refresh); a feed that repeats itself reads as two events.
    distinct = []
    for e in reversed(events):
        key = (e.get("type"), e.get("model"), e.get("old_king"))
        if distinct and key == distinct[-1][0]:
            continue
        distinct.append((key, e))
    recent = "".join(_event_line(e) for _, e in distinct[:8])
    return f"""
    <div class="lb-panel">
        <div class="lb-panel-head"><span class="lb-panel-title">&#9876;&#65039; Pipeline</span>
            <span class="lb-panel-meta">save &rarr; trial &rarr; pool &rarr; crown</span></div>
        <div class="lg-flow">{flow}</div>
        <div class="lg-trials">{trials}</div>
        <div class="lg-sub-head">Recent league events</div>
        <ul class="lg-events">{recent or '<li class="lg-note">Nothing yet.</li>'}</ul>
    </div>"""


# ---------------------------------------------------------------------------------------------
# Head to head and recent series
# ---------------------------------------------------------------------------------------------

def head_to_head_html(d: BoardData, n: int = 6) -> str:
    rows = d.elite[:n]
    if len(rows) < 2:
        return ""
    cols = rows + [r for r in (d.pinned, d.necto) if r is not None]
    head = "".join(f'<th title="{html.escape(c.name)}">{short_name(c)}</th>' for c in cols)
    body = []
    for a in rows:
        cells = []
        for b in cols:
            if a is b:
                cells.append('<td class="lg-h2h-self"></td>')
                continue
            score, games = d.head_to_head(a, b)
            if not games:
                p = p_beats(a, b)
                cells.append(f'<td class="lg-h2h-none" title="not played; the fit expects {_pct(p)}">&middot;</td>')
                continue
            share = score / games
            hue = 142 if share >= 0.5 else 350
            alpha = 0.10 + 0.45 * abs(share - 0.5) * 2
            txt = f"{score:g}&ndash;{games - score:g}"
            cells.append(f'<td style="background:hsla({hue},70%,45%,{alpha:.2f})" '
                         f'title="{short_name(a)} vs {short_name(b)}: {txt} in {games} series; the fit expects {_pct(p_beats(a, b))}">{txt}</td>')
        body.append(f'<tr><th>{short_name(a)}</th>{"".join(cells)}</tr>')
    return f"""
    <div class="lb-panel">
        <div class="lb-panel-head"><span class="lb-panel-title">&#129354; Head to head</span>
            <span class="lb-panel-meta">series won&ndash;lost, row vs column</span></div>
        <div class="lg-h2h-wrap"><table class="lg-h2h"><tr><th></th>{head}</tr>{''.join(body)}</table></div>
    </div>"""


def recent_series_html(d: BoardData, n: int = 12) -> str:
    rows = []
    now = datetime.datetime.now()
    untimed = _untimed_at(d.results)
    for r in list(reversed(d.results))[:n]:
        a, b = d.by_path.get(_norm(r["a"])), d.by_path.get(_norm(r["b"]))
        na = short_name(a) if a else os.path.basename(r["a"])
        nb = short_name(b) if b else os.path.basename(r["b"])
        sa, sb = r["a_score"], r["b_score"]
        wa, wb = ("lg-win", "") if sa > sb else (("", "lg-win") if sb > sa else ("", ""))
        kind = r.get("kind", "league")
        rows.append(f"""
        <li><span class="lg-kind lg-kind-{kind}">{KIND_LABEL.get(kind, kind)}</span>
            <span class="lg-match"><span class="{wa}">{na}</span>
            <span class="lg-score">{sa}&ndash;{sb}</span><span class="{wb}">{nb}</span></span>
            <span class="lg-ago">{'migrated' if r.get('at') in untimed else ago(r.get('at'), now)}</span></li>""")
    hour = sum(1 for r in _timed(d.results) if _within(r.get("at"), now, 3600))
    return f"""
    <div class="lb-panel">
        <div class="lb-panel-head"><span class="lb-panel-title">&#127918; Recent series</span>
            <span class="lb-panel-meta"><b>{hour}</b> in the last hour</span></div>
        <ul class="lg-series">{''.join(rows) or '<li class="lg-note">No series played yet.</li>'}</ul>
    </div>"""


# ---------------------------------------------------------------------------------------------
# The board
# ---------------------------------------------------------------------------------------------

def board_html(evaluator, state: Dict[str, Any], current_version: str, benchmark_chart: str) -> str:
    d = BoardData(evaluator, state, current_version)
    return f"""
    <div class="league-board">
        <div class="lb-panel">
            <div class="lb-panel-head"><span class="lb-panel-title">&#128200; Rating ladder</span>
                <span class="lb-panel-meta">every rated checkpoint, against the v3 king and the references</span></div>
            {ladder_chart_html(d)}
        </div>
        <div class="lg-grid-main">
            <div class="lb-panel">
                <div class="lb-panel-head"><span class="lb-panel-title">&#128737;&#65039; Elite pool</span>
                    <span class="lb-panel-meta">the King and pool opponents are drawn from here</span></div>
                {standings_html(d)}
            </div>
            {pipeline_html(d)}
        </div>
        <div class="lg-grid-even">
            {head_to_head_html(d)}
            {recent_series_html(d)}
        </div>
        <div class="lb-panel">
            <div class="lb-panel-head"><span class="lb-panel-title">&#127919; Benchmarks</span>
                <span class="lb-panel-meta">goal margin per episode against Necto and Nexto, by iteration</span></div>
            {benchmark_chart}
        </div>
    </div>"""


LEAGUE_CSS = """
/* ---- League board (ui/league_board.py) ------------------------------------------------- */
.lg-standby { padding: 14px 20px; color: #7c8ba1; font-size: 0.88em; }
.lb-stat-value small { font-size: 0.6em; color: #94a3b8; font-weight: 700; }
.lg-strip {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1px;
    margin-top: 8px; border-radius: 9px; overflow: hidden; border: 1px solid #1f2a3d; background: #1f2a3d;
}
.lg-strip > div { background: rgba(11, 17, 29, 0.95); padding: 8px 14px; display: flex; flex-direction: column; gap: 1px; }
.lg-strip span { font-size: 0.66em; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; color: #64748b; }
.lg-strip b { font-size: 1.05em; color: #e2e8f0; font-variant-numeric: tabular-nums; }

.lg-run {
    display: inline-block; font-size: 0.62em; font-weight: 800; letter-spacing: 0.6px; text-transform: uppercase;
    padding: 1px 6px; border-radius: 4px; vertical-align: 2px; margin-left: 4px;
    color: var(--run); border: 1px solid var(--run); background: color-mix(in srgb, var(--run) 14%, transparent);
}
.lg-pin { border-color: rgba(196, 181, 253, 0.6) !important; }

.lg-ref-label { font-size: 10.5px; font-weight: 700; font-family: ui-monospace, monospace; }
.lg-dot { animation: lg-pop 0.45s cubic-bezier(0.34, 1.56, 0.64, 1) both; transform-box: fill-box; transform-origin: center; }
@keyframes lg-pop { from { transform: scale(0); opacity: 0; } to { transform: scale(1); opacity: 1; } }
.lg-king-halo { fill: none; stroke: #facc15; stroke-width: 1.5; transform-box: fill-box; transform-origin: center;
    animation: lg-halo 2.4s ease-out infinite; }
@keyframes lg-halo { 0% { transform: scale(0.6); opacity: 0.9; } 100% { transform: scale(1.8); opacity: 0; } }
.lg-key-king, .lg-key-elite, .lg-key-prov { display: inline-block; width: 9px; height: 9px; border-radius: 50%; }
.lg-key-king { background: #facc15; }
.lg-key-elite { background: #38bdf8; box-shadow: 0 0 0 1.5px #f8fafc inset; }
.lg-key-prov { border: 1.5px solid #94a3b8; }

.lg-grid-main { display: grid; grid-template-columns: minmax(0, 1.65fr) minmax(300px, 1fr); gap: 10px; }
.lg-grid-even { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 10px; }
@media (max-width: 1100px) { .lg-grid-main { grid-template-columns: 1fr; } }

.lg-row {
    display: grid; grid-template-columns: 30px minmax(110px, 1.2fr) 86px minmax(120px, 2fr) 44px 48px 44px 104px 40px;
    align-items: center; gap: 9px; padding: 7px 10px; border-radius: 7px; font-size: 0.86em; color: #cbd5e1;
    border: 1px solid transparent; transition: background-color 0.16s ease, border-color 0.16s ease;
}
.lg-row + .lg-row { margin-top: 2px; }
.lg-row:not(.lg-row-head):hover { background: rgba(56, 189, 248, 0.06); border-color: rgba(56, 189, 248, 0.22); }
.lg-row-head { font-size: 0.66em; font-weight: 800; letter-spacing: 0.9px; text-transform: uppercase; color: #64748b;
    border-bottom: 1px solid rgba(51, 65, 85, 0.5); border-radius: 0; padding-bottom: 6px; margin-bottom: 3px; }
.lg-row-head i { font-style: normal; }
.lg-row-king { background: rgba(234, 179, 8, 0.07); border-color: rgba(234, 179, 8, 0.32); }

.lg-range { position: relative; height: 14px; }
.lg-range::before { content: ""; position: absolute; left: 0; right: 0; top: 6px; height: 2px; background: rgba(51, 65, 85, 0.6); border-radius: 1px; }
.lg-range-ci { position: absolute; top: 4px; height: 6px; border-radius: 3px; background: rgba(56, 189, 248, 0.35);
    transform-origin: left center; animation: lb-bar-grow 0.5s cubic-bezier(0.22, 1, 0.36, 1) both; }
.lg-range-ci.is-king { background: rgba(250, 204, 21, 0.4); }
.lg-range-ci.is-provisional { background: rgba(148, 163, 184, 0.3); }
.lg-range-mu { position: absolute; top: 1px; width: 3px; height: 12px; margin-left: -1.5px; border-radius: 2px; background: #38bdf8; }
.lg-range-mu.is-king { background: #facc15; }
.lg-range-mu.is-provisional { background: #94a3b8; }
.lg-pin-tick { position: absolute; top: -1px; bottom: -1px; width: 0; border-left: 1.5px dashed #c4b5fd; opacity: 0.8; }
.lg-pin-key { display: inline-block; width: 0; height: 9px; border-left: 1.5px dashed #c4b5fd; vertical-align: -1px; }

.lg-forms { display: inline-flex; gap: 3px; }
.lg-form { display: inline-flex; align-items: center; justify-content: center; width: 17px; height: 17px;
    border-radius: 4px; font-size: 0.7em; font-weight: 900; font-style: normal; color: #0b1220; }
.lg-form-W { background: #4ade80; }
.lg-form-L { background: #fb7185; }
.lg-form-D { background: #94a3b8; }

.lg-flow { display: flex; align-items: stretch; gap: 4px; padding: 12px 14px 4px; }
.lg-stage { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 1px; padding: 8px 9px;
    border-radius: 8px; background: rgba(15, 23, 42, 0.8); border: 1px solid #24324a; }
.lg-stage-k { font-size: 0.64em; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; color: #7dd3fc; }
.lg-stage b { font-size: 1.05em; color: #f1f5f9; font-variant-numeric: tabular-nums; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.lg-stage-sub { font-size: 0.68em; color: #64748b; line-height: 1.3; }
.lg-arrow { align-self: center; color: #334155; font-size: 0.9em; }
.lg-trials { padding: 8px 14px 2px; display: flex; flex-direction: column; gap: 6px; }
.lg-trial { padding: 8px 10px; border-radius: 8px; border: 1px solid #24324a; background: rgba(10, 15, 30, 0.7); display: flex; flex-direction: column; gap: 6px; }
.lg-trial-top { display: flex; justify-content: space-between; gap: 8px; font-size: 0.82em; color: #e2e8f0; }
.lg-note { font-size: 0.8em; color: #7c8ba1; line-height: 1.45; padding: 4px 2px; list-style: none; }
.lg-sub-head { padding: 10px 16px 4px; font-size: 0.66em; font-weight: 800; letter-spacing: 1px; text-transform: uppercase; color: #64748b; }

.lg-events, .lg-series { list-style: none; margin: 0; padding: 2px 10px 10px; display: flex; flex-direction: column; gap: 2px; }
.lg-ev, .lg-series li { display: grid; align-items: center; gap: 8px; padding: 5px 6px; border-radius: 6px;
    font-size: 0.82em; color: #94a3b8; }
.lg-ev { grid-template-columns: 22px 1fr auto; border-left: 2px solid transparent; }
.lg-ev b { color: #f1f5f9; }
.lg-ev:hover, .lg-series li:hover { background: rgba(56, 189, 248, 0.05); }
.lg-ev-crown { border-left-color: #eab308; }
.lg-ev-up { border-left-color: #22c55e; }
.lg-ev-down { border-left-color: #f43f5e; }
.lg-ev-in { border-left-color: #38bdf8; }
.lg-ev-icon { text-align: center; }
.lg-ago { font-family: ui-monospace, monospace; font-size: 0.85em; color: #52627a; white-space: nowrap; }

.lg-series li { grid-template-columns: 92px 1fr auto; }
.lg-kind { font-size: 0.7em; font-weight: 800; letter-spacing: 0.5px; text-transform: uppercase; padding: 2px 6px;
    border-radius: 4px; text-align: center; }
.lg-kind-league { background: rgba(56, 189, 248, 0.12); color: #7dd3fc; }
.lg-kind-calibration { background: rgba(168, 85, 247, 0.14); color: #c4b5fd; }
.lg-kind-benchmark { background: rgba(244, 63, 94, 0.12); color: #fda4af; }
.lg-kind-migrated { background: rgba(100, 116, 139, 0.15); color: #94a3b8; }
.lg-match { display: flex; align-items: center; gap: 8px; min-width: 0; font-variant-numeric: tabular-nums; }
.lg-match > span:not(.lg-score) { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.lg-score { font-family: ui-monospace, monospace; color: #e2e8f0; padding: 1px 7px; border-radius: 4px; background: rgba(30, 41, 59, 0.9); }
.lg-win { color: #f1f5f9; font-weight: 800; }

.lg-h2h-wrap { padding: 10px 14px 14px; overflow-x: auto; }
.lg-h2h { border-collapse: separate !important; border-spacing: 3px; font-size: 0.78em; width: 100%; background: transparent !important; }
.lg-h2h th, .lg-h2h td { border: none !important; }
.lg-h2h th { color: #7c8ba1; font-weight: 700; font-family: ui-monospace, monospace; font-size: 0.92em; padding: 3px 5px; white-space: nowrap; background: transparent !important; }
.lg-h2h td { text-align: center; padding: 6px 4px; border-radius: 5px; color: #e2e8f0; font-weight: 700;
    font-variant-numeric: tabular-nums; background: rgba(30, 41, 59, 0.5); }
.lg-h2h-self { background: repeating-linear-gradient(45deg, rgba(51,65,85,0.35) 0 4px, transparent 4px 8px) !important; }
.lg-h2h-none { color: #475569 !important; font-weight: 400 !important; }

@media (max-width: 900px) {
    .lg-row { grid-template-columns: 26px minmax(100px, 1.2fr) 82px minmax(90px, 2fr) 42px 42px; }
    .lg-row > .lb-hide-sm { display: none; }
    .lg-flow { flex-wrap: wrap; }
    .lg-arrow { display: none; }
}
@media (prefers-reduced-motion: reduce) {
    .lg-dot, .lg-king-halo, .lg-range-ci { animation: none !important; }
    .lg-king-halo { opacity: 0.5; }
}
"""
