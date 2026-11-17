# gta6-twitch-ccu-forecast-01-round2

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Rzfr123/gta6-twitch-ccu-forecast-01-round2/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/Rzfr123/gta6-twitch-ccu-forecast-01-round2/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Round-2 entry: forecasts Grand Theft Auto VI's launch-day Twitch concurrent
viewers, hour by hour (hours 2-24 of 19 November 2026 UTC), from the first
hour's print and the release-day curves of past big game launches.

Revises the round-1 entry (tiered log-ratio regression, practice loss 0.184)
in three validated ways: near-exclusion of distorted C-tier launches, a
ridge-regularised per-hour regression that extrapolates sanely to GTA VI's
scale, and a tier-aware quantile spread that is tighter for clean console
launches. Leave-one-out on the case's comparables: ALL 0.2439 (round 1:
0.2657), clean console targets 0.1524 (round 1: 0.1891). See STRATEGY.md.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. Reads every `*.in` case in
the input directory and writes one `*.out` forecast per case.

---

<sub>Entered on [make.fun](https://make.fun/?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>
