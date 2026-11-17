"""Compare forecasts of next-six-hour realized variance on archived AUDUSD H1.

Fit the calendar profile in 2023-2024. All forecasts issued in 2025 use only
closed bars. QLIKE differences are relative to the constant variance forecast.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

import duckdb
import numpy as np

from audusd_h1_pilot import read_market, utc


HOUR = 3600
HORIZON = 6
DECAY = 0.94


def calendar_variance(squared, split):
    sums = np.zeros(24)
    counts = np.zeros(24)
    for t, value in squared.items():
        if t < split:
            hour = t // HOUR % 24
            sums[hour] += value
            counts[hour] += 1
    mean = sums.sum() / counts.sum()
    if not np.isfinite(mean) or mean <= 0:
        raise ValueError("Expected positive finite historical variance")
    return mean, (sums + 20 * mean) / (counts + 20)


def replay(closes, stop):
    squared = {t: math.log(price / closes[t-HOUR]) ** 2
               for t, price in closes.items() if t-HOUR in closes}
    mean, profile = calendar_variance(squared, utc(2025))
    ewma, normalized = mean, 1.0
    previous = None
    pending = {}
    scored = []
    issued = missing = 0
    for now in range(utc(2024, 12), stop + 1, HOUR):
        hour = now // HOUR % 24
        if now in squared:
            if previous != now - HOUR:
                ewma, normalized = mean, 1.0
            ewma = DECAY * ewma + (1 - DECAY) * squared[now]
            normalized = DECAY * normalized + (1 - DECAY) * squared[now] / profile[hour]
            previous = now
        forecast = pending.pop(now, None)
        if forecast is not None:
            target_times = [now - step * HOUR for step in range(HORIZON)]
            if all(t in squared for t in target_times):
                rv = sum(squared[t] for t in target_times)
                loss_diff = np.log(forecast / forecast[0]) + rv * (1 / forecast - 1 / forecast[0])
                scored.append((now - HORIZON * HOUR, loss_diff))
            else:
                missing += 1
        if utc(2025) <= now < stop and now in squared:
            seasonal = sum(profile[(hour + h) % 24] for h in range(1, HORIZON + 1))
            pending[now + HORIZON * HOUR] = np.array(
                [HORIZON * mean, seasonal, HORIZON * ewma, seasonal * normalized])
            issued += 1
    assert issued == len(scored) + missing + len(pending)
    return scored, dict(issued=issued, scored=len(scored), missing_label=missing, right_censored=len(pending))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    with duckdb.connect() as db:
        bars = read_market(db, args.cache, "AUDUSD", stop_year=2026)
    scored, counts = replay({t + HOUR: close for t, close in bars.items()}, utc(2026))
    weeks = defaultdict(list)
    for origin, diffs in scored:
        weeks[(origin // 86400 + 3) // 7].append(diffs)
    blocks = np.array([np.append(np.sum(values, axis=0), len(values)) for values in weeks.values()])
    ids = np.random.default_rng(37).integers(len(blocks), size=(2000, len(blocks)))
    sample = blocks[ids].sum(axis=1)
    sampled_means = sample[:, :-1] / sample[:, -1:]
    ci = np.quantile(sampled_means, (0.025, 0.975), axis=0)
    mean = blocks[:, :-1].sum(axis=0) / blocks[:, -1].sum()
    names = ("constant", "seasonal", "ewma", "seasonal_ewma")
    scores = {name: dict(delta=float(mean[i]), ci95=ci[:, i].tolist()) for i, name in enumerate(names)}
    paired = sampled_means[:, 3] - sampled_means[:, 1]
    hybrid_comparison = dict(delta=float(mean[3] - mean[1]), ci95=np.quantile(paired, (0.025, 0.975)).tolist())
    print(json.dumps(dict(**counts, weeks=len(weeks), qlike_vs_constant=scores,
                          hybrid_minus_seasonal=hybrid_comparison), indent=2))


if __name__ == "__main__":
    main()
