# Strategy: GTA VI launch-day Twitch CCU forecast (round 2)

## What is forecast

For each hour h = 2..24 of the UTC release day, five quantiles (5%, 25%, 50%,
75%, 95%) of that hour's mean concurrent viewers in Twitch's "Grand Theft
Auto VI" category, given the first hour's print and everything recorded up to
the end of hour 1. Scored by mean pinball loss on the log scale, so the
quantiles should be the forecaster's honest beliefs: a narrow forecast pays
for every hour outside it, a wide one pays at every hour.

## The model

The anchor is the target's own hour-1 print p1. Everything else is learned
from the case's comparables: past big game launches, each with its full
release-day hourly curve (14 days before to 3 days after release).

1. **Ratios, not levels.** For each comparable c and hour h, the log ratio
   l_c(h) = ln(v_c(h) / v_c(1)). Launch-day shapes are far more stable in
   ratio space than in levels: the median ratio dips in hours 2-5 (the
   post-midnight lull), then climbs to a peak around hours 20-21 (US
   afternoon/evening), then eases off.
2. **Tiered pool, reweighted.** Comparables are graded by how much they
   resemble the target (a console-only, rolling-midnight launch): tier A
   (clean rolling-midnight console launches, weight 1.0), tier B
   (multi-platform, global unlock, or early-access history, 0.7), tier C
   (free-to-play, or launches that had not started by 00:00 UTC so their
   hour-1 print is pre-launch noise, 0.05). Round 1 used 1.0 / 0.45 / 0.12;
   round 2 nearly excludes C because its curves live on a different scale
   (mean hour-20 log ratio +4.6, versus +0.8 for tier A), and trusts B
   more because B launches carry real shape information.
3. **Ridge-regularised per-hour regression.** For each hour, l(h) is
   regressed on log hour-1 viewers (launch scale) and log hour-1 channels
   (launch breadth) with the tier weights and a ridge penalty on the
   slopes. Scale matters strongly: bigger launches have flatter curves
   (correlation of log hour-1 viewers with log peak ratio is -0.77). The
   pool's largest launch is ~875k hour-1 viewers and GTA VI will be far
   larger, so the regression extrapolates; the ridge keeps that
   extrapolation sane (predicted peak ratio ~1.5x at 1M hour-1 viewers,
   ~1.2x at 3M, never below 1x).
4. **Tier-aware spread.** The residual spread around the regression sets
   the quantile width, but round 1's pooled spread is inflated by unlike
   launches. Round 2 blends the pooled spread with the spread of
   residuals from launches of the target's own tier (50/50 in variance
   space), shrinks 40% toward a prior that grows with the hour
   (0.30 + 0.35*sqrt((h-1)/23)), floors at 0.18, and scales by 0.90
   overall but 0.70 for clean console targets, whose curves are much
   tighter (hourly log-ratio sd ~0.35 versus ~0.6 pooled). Round 1's
   practice coverage was 0.978 against a 0.90 target: its bands were far
   too wide for a launch of GTA VI's kind.
5. **Momentum tilt.** The target's own pre-checkpoint hourly history gives
   a recent log-slope (mean over the last consecutive pre-launch hours,
   capped at 0.35/hour). It tilts the median for hours 2-8, decaying to
   zero, at 0.35 weight. On practice cases the slope is ~0 (flat
   pre-launch), so the tilt is negligible there; on the final it carries
   the real unlock ramp (New Zealand and Europe unlocking hours before the
   UTC day starts).

## Validation (leave-one-out on the 38 comparables)

Each comparable launch in turn is treated as the target (its hour-1 print
known, the other 37 as the pool) and scored by the same pinball loss:

- Round 1: mean loss 0.266 over all 29 usable targets, 0.189 over the 4
  clean rolling-midnight console targets.
- Round 2: mean loss 0.244 over all targets, 0.152 over the clean console
  targets; coverage_90 0.922 (target 0.90).
- House baseline (persistence, starter): 0.360 / 0.301 on the same targets.
- Practice leaderboard (4 practice cases): round 1 scored 0.184
  (coverage_90 0.978). The practice cases are 4 specific big console
  launches, so the leave-one-out numbers are a pessimistic guide; the
  final case is a single huge console launch whose pool includes the 4
  practice launches.

Ideas tested and rejected on leave-one-out: hype/jump features from the
pre-launch history (hurt clean targets), kernel weighting by launch scale
(marginal), k-nearest-neighbour medians (much worse), a rank-1 template
model (much worse), empirical residual quantiles instead of lognormal
(worse), tier-dummy features (worse), and a viewers-per-channel feature
(worse).

## Determinism and limits

Pure Python standard library, no network, no randomness: two runs on the
same input write byte-identical outputs. All 4 practice cases run in
~0.3 s total, far inside the 8 s per-case limit. Outputs are exactly 23
rows (hours 2-24), five non-decreasing quantiles each, within (0, 1e9].
