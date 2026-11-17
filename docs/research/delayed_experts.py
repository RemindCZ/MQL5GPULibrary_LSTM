"""Prequential AUDUSD direction experiment, six hours, fixed 2025 evaluation.

Reuses the two hash-checked H1 files and the reader from audusd_h1_pilot.py.
Windows/MT5, CUDA and PyTorch are not required.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import duckdb
import numpy as np

from audusd_h1_pilot import read_market, utc


HOUR = 3600
HORIZON = 6
ETA = 0.1
PRIOR_OBSERVATIONS = 20
NAMES = ("neutral", "session", "lag_signs", "equal", "hedge")


def input_groups(markets, origin):
    """Read only closes already available at origin."""
    signs = []
    for bars in markets:
        if any(origin - step * HOUR not in bars for step in range(HORIZON + 1)):
            return None
        signs.append(int(bars[origin] > bars[origin - HORIZON * HOUR]))
    return (origin // HOUR % 24 // 6, tuple(signs))


def realization(target, maturity):
    if any(maturity - step * HOUR not in target for step in range(HORIZON + 1)):
        return None
    return int(target[maturity] > target[maturity - HORIZON * HOUR])


def fit_tables(markets):
    totals = [Counter(), Counter()]
    positives = [Counter(), Counter()]
    examples = 0
    for origin in range(utc(2023), utc(2025) - HORIZON * HOUR, HOUR):
        groups = input_groups(markets, origin)
        outcome = realization(markets[0], origin + HORIZON * HOUR)
        if groups is None or outcome is None:
            continue
        examples += 1
        for i, key in enumerate(groups):
            totals[i][key] += 1
            positives[i][key] += outcome
    tables = [{key: (pos[key] + PRIOR_OBSERVATIONS / 2) / (n + PRIOR_OBSERVATIONS)
               for key, n in total.items()} for total, pos in zip(totals, positives)]
    return tables, examples


def forecast_year(markets, tables, stop):
    # Each UTC-hour lane has at most one outstanding prediction at a time.
    log_weights = np.zeros((HORIZON, 3))
    pending = {}
    settled = []
    issued = expired = 0
    for now in range(utc(2025), stop + 1, HOUR):
        lane = now // HOUR % HORIZON
        previous = pending.pop(now, None)
        if previous is not None:
            outcome = realization(markets[0], now)
            if outcome is None:
                expired += 1
            else:
                losses = (previous - outcome) ** 2
                log_weights[lane] -= ETA * losses[:3]
                settled.append((now - HORIZON * HOUR, losses))
        if now >= stop:
            break
        groups = input_groups(markets, now)
        if groups is None:
            continue
        experts = np.array([0.5, tables[0].get(groups[0], 0.5), tables[1].get(groups[1], 0.5)])
        weights = np.exp(log_weights[lane] - log_weights[lane].max())
        pooled = np.dot(experts, weights / weights.sum())
        pending[now + HORIZON * HOUR] = np.append(experts, [experts.mean(), pooled])
        issued += 1
    assert issued == len(settled) + expired + len(pending)
    return settled, dict(issued=issued, scored=len(settled), missing_label=expired, right_censored=len(pending))


def summarize(settled):
    loss = np.stack([row for _, row in settled])
    scores = dict(zip(NAMES, loss.mean(axis=0).round(8).tolist()))
    # Paired bootstrap of calendar weeks (Monday in UTC), weighted by their observation counts.
    weeks = defaultdict(list)
    for origin, values in settled:
        weeks[(origin // 86400 + 3) // 7].append(values[-1] - values[-2])
    blocks = np.array([[sum(values), len(values)] for values in weeks.values()])
    rng = np.random.default_rng(37)
    sample = rng.integers(len(blocks), size=(2000, len(blocks)))
    totals = blocks[sample].sum(axis=1)
    interval = np.quantile(totals[:, 0] / totals[:, 1], (0.025, 0.975))
    return dict(brier=scores, hedge_minus_equal=float(np.mean(loss[:, -1] - loss[:, -2])),
                weekly_ci95=interval.tolist(), weeks=len(blocks))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    args = parser.parse_args()
    with duckdb.connect() as db:
        markets = [{t + HOUR: price for t, price in read_market(db, args.cache, symbol, stop_year=2026).items()}
                   for symbol in ("AUDUSD", "NZDUSD")]
    tables, examples = fit_tables(markets)
    settled, counts = forecast_year(markets, tables, utc(2026))
    print(json.dumps(dict(training=examples, **counts, **summarize(settled)), indent=2))


if __name__ == "__main__":
    main()
