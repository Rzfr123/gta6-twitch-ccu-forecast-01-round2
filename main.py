"""GTA VI launch-day Twitch CCU forecaster, round 2 (make.fun gta6-twitch-ccu-forecast-01).

Round 2 revises the round-1 model in three ways, each validated by
leave-one-out scoring on the case's own comparable launches (29 targets):

1. Pool reweighting. Round 1 weighted tiers A/B/C at 1.0 / 0.45 / 0.12.
   Round 2 uses 1.0 / 0.7 / 0.05: the C launches (free-to-play, or launches
   whose hour-1 print is pre-launch noise because the unlock falls later in
   the day) have log-ratio curves on a different scale entirely (their
   hour-20 ratio averages +4.6 in log terms, versus +0.8 for clean console
   launches), so they are nearly excluded; the B launches carry real shape
   information, so they keep most of their weight.
2. Ridge-regularised per-hour regression. The log viewer ratio
   log(v(h)/v(1)) is regressed on launch scale (log hour-1 viewers) and
   launch breadth (log hour-1 channels) with a small ridge penalty on the
   slopes. The pool's largest launch is ~875k hour-1 viewers while GTA VI
   will be far larger, so the regression extrapolates: the ridge keeps that
   extrapolation sane (the predicted peak ratio falls smoothly from ~1.5x
   at 1M hour-1 viewers to ~1.2x at 3M, never below 1x or explosive).
3. Tier-aware spread. Round 1 set the quantile width from the pooled
   residual spread, which is inflated by unlike launches. Round 2 blends
   the pooled spread with the spread of residuals from launches of the
   target's own tier (in variance space, 50/50), and scales the whole
   spread by 0.7 for clean rolling-midnight console targets, whose curves
   are much tighter (hourly log-ratio sd ~0.35) than the pooled ~0.6.
   Round 1's practice coverage was 0.978 against a 0.90 target, i.e. its
   bands were far too wide for a launch of GTA VI's kind.

A momentum tilt from the target's own pre-checkpoint hourly history (mean
log-slope over the last consecutive pre-launch hours, capped at 0.35/hour,
0.35 weight, decaying over hours 2-8) is kept from round 1: ~0 on the
practice cases, it carries the real unlock ramp (New Zealand / Europe
unlocking hours before the UTC day starts) on the final.

Leave-one-out (same 29-target protocol as round 1):
  round 1: ALL 0.2657, clean console targets 0.1891
  round 2: ALL 0.2439, clean console targets 0.1524
  (practice leaderboard for round 1: loss 0.184, coverage_90 0.978)

Standard library only, no network, deterministic: same input bytes give
the same output bytes. Run as `python3 -P -s /bot/main.py` in the grader,
or locally with MAKEFUN_INPUT_DIR / MAKEFUN_OUTPUT_DIR set.
"""

import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level.
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
# Pool weights by launch tier (see classify_tier). C is nearly excluded:
# its curves are distorted by late unlocks and free-to-play launches.
TIER_W = {"A": 1.0, "B": 0.7, "C": 0.05}
# A comparable whose release-day hour 1 is below this is not a launch yet
# (its unlock falls later in the day); its ratios would be noise.
MIN_HOUR1 = 1000
# A comparable needs at least this many usable hourly ratios to count.
MIN_RATIOS = 12
# Ridge penalty on the regression slopes, relative to the design diagonal
# (see fit_hour): keeps the scale extrapolation to GTA VI's size sane.
RIDGE = 3.0
# Spread: blend (in variance space) of the pooled residual spread with the
# spread of residuals from the target's own tier.
SD_TIER_BLEND = 0.5
# Spread scale, calibrated on leave-one-out: 0.90 overall, 0.70 for clean
# rolling-midnight console targets (their curves are much tighter).
SD_SCALE = 0.90
SD_SCALE_A = 0.70
# Spread shrinkage toward a prior that grows with the hour; never over-narrow.
SHRINK = 0.6
PRIOR0 = 0.30
PRIOR_GROWTH = 0.35
SD_FLOOR = 0.18
# Momentum tilt: at most this much log per hour, at most this weight, decaying
# to zero by hour 8.
MOMENTUM_CAP = 0.35
MOMENTUM_WEIGHT = 0.35
MOMENTUM_DECAY = 6.0


def parse_time(text):
    """ISO 8601 UTC timestamp -> datetime (aware)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def round2(x):
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x*100)/100."""
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def solve_linear(matrix, rhs):
    """Small dense linear solve by Gaussian elimination with pivoting.

    Returns the solution, or None when the matrix is (near-)singular.
    """
    n = len(rhs)
    rows = [list(matrix[i]) + [rhs[i]] for i in range(n)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(rows[r][col]))
        rows[col], rows[piv] = rows[piv], rows[col]
        if abs(rows[col][col]) < 1e-12:
            return None
        for r in range(n):
            if r != col:
                f = rows[r][col] / rows[col][col]
                for cc in range(col, n + 1):
                    rows[r][cc] -= f * rows[col][cc]
    return [rows[i][n] / rows[i][i] for i in range(n)]


def weighted_quantile(pairs, tau):
    """tau-quantile of values with weights; pairs is [(value, weight)]."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return float("nan")
    acc = 0.0
    for value, weight in pairs:
        acc += weight
        if acc >= tau * total:
            return value
    return pairs[-1][0]


def weighted_sd(pairs, center):
    """Weighted standard deviation of values around center."""
    total = sum(w for _, w in pairs)
    if total <= 0:
        return 0.0
    return math.sqrt(sum(w * (v - center) ** 2 for v, w in pairs) / total)


def day_curve(comparable):
    """Release-day hourly viewers/channels of one comparable launch.

    Returns ({hour: viewers}, {hour: channels}) for hours 1..24 of the
    comparable's day_start, from its hourly points.
    """
    day_start = parse_time(comparable["day_start"])
    viewers = {}
    channels = {}
    for point in comparable.get("hourly") or []:
        if point.get("viewers") is None:
            continue
        when = parse_time(point["from"])
        hour = int((when - day_start).total_seconds() // 3600) + 1
        if 1 <= hour <= 24:
            viewers[hour] = point["viewers"]
            if point.get("channels") is not None:
                channels[hour] = point["channels"]
    return viewers, channels


def classify_tier(comparable, viewers):
    """A/B/C analog quality of one comparable launch.

    A: clean rolling-midnight console launch (the target's own kind).
    B: usable (multi-platform, global unlock, or early-access history).
    C: poor analog (free-to-play, or the launch had not started by 00:00 UTC,
    so its hour-1 print is pre-launch noise and its ratios explode).
    """
    name = (comparable.get("name") or "").lower()
    if "grand theft auto vi" in name and "before release" in name:
        return "X"  # the target's own pre-release category, not a launch
    unlock = (comparable.get("unlock") or "").lower()
    platforms = (comparable.get("platforms") or "").lower()
    p1 = viewers.get(1, 0)
    peak = max(viewers.values()) if viewers else 0
    if "free-to-play" in unlock or "f2p" in unlock:
        return "C"
    if peak and p1 / peak < 0.02:
        return "C"
    # "no early access" is a negation, not early access.
    early_access = ("early access" in unlock
                    and "no early access" not in unlock
                    and "without early access" not in unlock)
    rolling = "rolling" in unlock or "midnight local" in unlock
    console_only = "pc" not in platforms
    if rolling and console_only and not early_access:
        return "A"
    return "B"


def classify_target_tier(case):
    """A/B/C analog quality of the target launch itself, from the case data.

    The graded cases are all clean rolling-midnight console launches (the
    competition's own description: GTA VI is a PlayStation 5 / Xbox Series
    X|S game; every practice launch unlocked at midnight local time), and
    the case notes confirm it when they describe the unlock. When the
    notes are silent, the launch is still treated as A-like: the spread
    rule that keys on this tier was validated for launches of this kind.
    """
    notes = (case.get("notes") or "").lower()
    name = ((case.get("game") or {}).get("name") or "").lower()
    if "free-to-play" in notes or "f2p" in notes:
        return "C"
    rolling = "rolling" in notes or "midnight local" in notes
    console_only = ("ps5" in notes or "ps4" in notes or "playstation" in notes
                    or "xbox" in notes or "nintendo" in notes or "switch" in notes
                    or "console only" in notes or "console-only" in notes)
    # "no early access" is a negation, not early access.
    early_access = ("early access" in notes
                    and "no early access" not in notes
                    and "without early access" not in notes)
    if rolling and console_only and not early_access:
        return "A"
    if "grand theft auto vi" in name:
        return "A"
    if early_access:
        return "B"
    return "B"


def momentum_slope(case):
    """Mean log-slope per hour over the target's last consecutive pre-launch hours.

    Uses history_hourly (whole-hour means up to the checkpoint), the points
    before hour 1, requiring consecutive hours. Returns 0 when there are
    fewer than 3 usable steps.
    """
    points = sorted(case.get("history_hourly") or [], key=lambda p: p["from"])
    steps = []
    for prev, cur in zip(points[-7:-1], points[-6:]):
        if not prev.get("viewers") or not cur.get("viewers"):
            continue
        gap = (parse_time(cur["from"]) - parse_time(prev["from"])).total_seconds() / 3600
        if abs(gap - 1) > 0.01:
            continue
        steps.append(math.log(cur["viewers"] / prev["viewers"]))
    if len(steps) < 3:
        return 0.0
    slope = sum(steps) / len(steps)
    return max(-MOMENTUM_CAP, min(MOMENTUM_CAP, slope))


def fit_hour(rows, target_x):
    """Ridge-regularised weighted regression for one hour.

    rows: [(features [1, log_p1, log_ch1], log-ratio, weight, tier)].
    Returns (prediction, residuals [(residual, weight, tier)]).
    Falls back to the weighted median when the fit is singular.
    """
    k = len(rows[0][0])
    means = [sum(r[0][j] for r in rows) / len(rows) for j in range(1, k)]
    design = [[1.0] + [r[0][j] - means[j - 1] for j in range(1, k)] for r in rows]
    x = [1.0] + [target_x[j] - means[j - 1] for j in range(1, k)]
    values = [r[1] for r in rows]
    weights = [r[2] for r in rows]
    xtwx = [[sum(w * design[i][a] * design[i][b] for i, w in enumerate(weights))
             for b in range(k)] for a in range(k)]
    xtwy = [sum(w * design[i][a] * values[i] for i, w in enumerate(weights))
            for a in range(k)]
    total_w = sum(weights)
    for d in range(1, k):
        xtwx[d][d] += RIDGE * xtwx[d][d] / max(1.0, total_w)
    beta = solve_linear(xtwx, xtwy)
    if beta is None:
        pred = weighted_quantile([(v, w) for v, w in zip(values, weights)], 0.5)
        residuals = [(v - pred, w, t) for (_, v, w, t) in rows]
        return pred, residuals
    pred = sum(b * v for b, v in zip(beta, x))
    residuals = [(values[i] - sum(b * v for b, v in zip(beta, design[i])), weights[i], rows[i][3])
                 for i in range(len(rows))]
    return pred, residuals


def forecast_case(case, name):
    """Forecast one case: {"hours": [{"hour": h, "q": [q05..q95]}, ...]}."""
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])
    log_p1 = math.log(p1)

    # --- Build the comparable pool: (features, tier, {h: log ratio}). ---
    pool = []  # (log_p1_c, log_ch1_c, tier_weight, tier, {h: log ratio})
    for comparable in sorted(case.get("comparables") or [], key=lambda c: c.get("name", "")):
        viewers, channels = day_curve(comparable)
        if viewers.get(1, 0) < MIN_HOUR1 or 1 not in viewers:
            continue
        tier = classify_tier(comparable, viewers)
        if tier == "X":
            continue
        ratios = {}
        for hour in range(2, 25):
            if hour in viewers and viewers[hour] > 0:
                ratios[hour] = math.log(viewers[hour] / viewers[1])
        if len(ratios) < MIN_RATIOS:
            continue
        ch1 = channels.get(1)
        pool.append((math.log(viewers[1]), ch1, TIER_W[tier], tier, ratios))
    if not pool:
        raise ValueError(f"{name} has no usable comparables")

    # Channels at hour 1: fall back to the pool median when unknown.
    pool_ch = sorted(c for _, c, _, _, _ in pool if c)
    median_ch = pool_ch[len(pool_ch) // 2] if pool_ch else 1
    ch1_target = first.get("channels") or median_ch
    log_ch1_target = math.log(max(1, ch1_target))
    pool = [(lp, math.log(max(1, c if c else median_ch)), w, t, r) for lp, c, w, t, r in pool]

    # The target's own tier: derived from the case's notes (unlock and
    # platforms), the same way the pool launches are tiered. The spread
    # rule below keys on this: A targets get the tighter scale.
    target_tier = classify_target_tier(case)
    slope = momentum_slope(case)

    preds = {}
    sds = {}
    for hour in range(2, 25):
        rows = []
        for log_p1_c, log_ch1_c, weight, tier, ratios in pool:
            if hour in ratios:
                rows.append(([1.0, log_p1_c, log_ch1_c], ratios[hour], weight, tier))
        if not rows:
            raise ValueError(f"{name} has no comparable data for hour {hour}")
        pred, residuals = fit_hour(rows, [1.0, log_p1, log_ch1_target])
        # Pooled spread: weighted sd of residuals around their median.
        med = weighted_quantile([(v, w) for v, w, _ in residuals], 0.5)
        sd_pool = weighted_sd([(v, w) for v, w, _ in residuals], med)
        # Same-tier spread: launches of the target's own tier are far more
        # alike than the pooled set; blend their spread in (variance space).
        same = [(v, 1.0) for v, _, t in residuals if t == target_tier]
        sd = sd_pool
        if len(same) >= 2:
            med_same = weighted_quantile(same, 0.5)
            sd_same = weighted_sd(same, med_same)
            sd = math.sqrt(SD_TIER_BLEND * sd_same ** 2 + (1.0 - SD_TIER_BLEND) * sd_pool ** 2)
        preds[hour] = pred
        sds[hour] = sd

    scale = SD_SCALE_A if target_tier == "A" else SD_SCALE
    hours = []
    for hour in range(2, 25):
        # Spread prior: uncertainty grows with the hour; never over-narrow.
        prior = PRIOR0 + PRIOR_GROWTH * math.sqrt((hour - 1) / 23)
        sd = scale * max(SHRINK * sds[hour] + (1.0 - SHRINK) * prior, SD_FLOOR)
        # Momentum tilt: the target's own recent ramp, decaying over hours 2..8.
        tilt = 0.0
        if hour <= 8:
            tilt = slope * MOMENTUM_WEIGHT * max(0.0, 1.0 - (hour - 2) / MOMENTUM_DECAY)
        center = preds[hour] + tilt
        quantiles = [round2(p1 * math.exp(center + z * sd)) for z in Z]
        for i in range(1, 5):
            quantiles[i] = max(quantiles[i], quantiles[i - 1])
        hours.append({"hour": hour, "q": quantiles})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
