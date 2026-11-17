"""Reproduce the historical pilot described in ../NAVRH_PREDIKCI.cs.md.

Requires Python, NumPy and DuckDB. --dll adds the native recurrent reference. --tcn needs PyTorch.
The four H1 Parquet files come from the published Yllvar/fx-historical-data snapshot.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import duckdb
import numpy as np


SYMBOLS = ("AUDUSD", "NZDUSD", "USDJPY", "USDCAD")
SEQUENCE, HORIZON = 24, 6
ROOT = Path(__file__).resolve().parents[2]
FILE_HASH = {
    "AUDUSD": "ab50e8269f38482e86288ed3d405dc5e3d53d0a0deb9e7fdd04ec8a34fbcd855",
    "NZDUSD": "bee4e49de3ac0927ae8da91094b5d175d38d99def18ca9ae1920ab1acfbf1500",
    "USDJPY": "daa87eaec20ed680bb0830a753df4727865b334c084af779d4ab3e6bb28ad485",
    "USDCAD": "cfd54675c988da6e88843136f42d87dee136f6529a70f6a438d01d11d140952f",
}


def utc(year, month=1):
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp())


def read_market(db, cache, symbol, *, stop_year=2025):
    path = cache / f"{symbol}_H1.parquet"
    if hashlib.sha256(path.read_bytes()).hexdigest() != FILE_HASH[symbol]:
        raise ValueError(f"Different data revision for {symbol}")
    rows = db.execute("""
        SELECT timestamp_ms, Close FROM read_parquet(?)
        WHERE timestamp_ms >= ? AND timestamp_ms < ? AND Volume > 0
        ORDER BY timestamp_ms
        """, [str(path), utc(2023) * 1000, utc(stop_year) * 1000]).fetchall()
    bars = {}
    for ms, close in rows:
        stamp = ms // 1000
        if ms % 3_600_000 or stamp in bars or not np.isfinite(close) or close <= 0:
            raise ValueError(f"Invalid hourly record for {symbol} at {ms}")
        bars[stamp] = close
    if not bars:
        raise ValueError(f"No observations for {symbol}")
    return bars


def dataset(cache):
    with duckdb.connect() as db:
        markets = {s: read_market(db, cache, s) for s in SYMBOLS}
    stamps = np.array(sorted(markets[SYMBOLS[0]]))
    prices = np.array([[markets[s].get(t, np.nan) for s in SYMBOLS] for t in stamps], dtype=np.float64)
    returns = np.log(prices[1:] / prices[:-1])
    origins = np.arange(SEQUENCE, len(stamps) - HORIZON)
    continuous = stamps[origins + HORIZON] - stamps[origins - SEQUENCE] == 3600 * (SEQUENCE + HORIZON)
    ends = origins[continuous]
    # Future driver availability must never decide whether an origin is scored.
    usable = np.isfinite(returns).all(axis=1)
    missing = np.concatenate(([0], np.cumsum(~usable)))
    ends = ends[missing[ends] == missing[ends - SEQUENCE]]
    split = utc(2024)
    decision = stamps[ends] + 3600
    train = stamps[ends + HORIZON] + 3600 < split
    test = decision >= split
    x = np.stack([returns[i-SEQUENCE:i] for i in ends])
    y = np.stack([100 * np.log(prices[i+1:i+HORIZON+1, 0] / prices[i, 0]) for i in ends])
    # Fit one location and scale per market on unique input bars used by training windows.
    rows = np.unique((ends[train, None] - SEQUENCE + np.arange(SEQUENCE)).ravel())
    x = (x - returns[rows].mean(axis=0)) / np.maximum(returns[rows].std(axis=0), 1e-8)
    return x.reshape(len(ends), -1), y, decision, train, test


def ridge(x, y, fit, evaluate):
    center = x[fit].mean(axis=0)
    target_center = y[fit].mean(axis=0)
    design = x[fit] - center
    penalty = len(design) * np.eye(design.shape[1])
    weights = np.linalg.solve(design.T @ design + penalty, design.T @ (y[fit] - target_center))
    return (x[evaluate] - center) @ weights + target_center


def recurrent(x, y, fit, evaluate, dll):
    sys.path.insert(0, str(ROOT / "python"))
    from mql5gpu import Library, Model
    with Model(Library(dll)) as model:
        model.configure(len(SYMBOLS), SEQUENCE, HORIZON)
        model.fit(x[fit].tolist(), y[fit].tolist(), epochs=6, batch_size=128, timeout=1200)
        test_x = x[evaluate]
        return np.vstack([model.predict(test_x[i:i+64].tolist()) for i in range(0, len(test_x), 64)])


def tcn_predict(x, y, fit, evaluate, seed):
    import torch
    from torch import nn

    class CausalBlock(nn.Module):
        def __init__(self, dilation):
            super().__init__()
            self.left = nn.ConstantPad1d((2 * dilation, 0), 0.0)
            self.conv = nn.Conv1d(16, 16, kernel_size=3, dilation=dilation)

        def forward(self, inputs):
            return inputs + torch.relu(self.conv(self.left(inputs)))

    class PointTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(nn.Conv1d(len(SYMBOLS), 16, 1),
                                         *(CausalBlock(d) for d in (1, 2, 4, 8)))
            self.head = nn.Linear(16, HORIZON)
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

        def forward(self, inputs):
            return self.head(self.encoder(inputs)[:, :, -1])

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    inputs = torch.as_tensor(x.reshape(-1, SEQUENCE, len(SYMBOLS)), dtype=torch.float32)
    inputs = inputs.transpose(1, 2).contiguous()
    train_x = inputs[fit]
    scale = np.maximum(y[fit].std(axis=0), 1e-8)
    target = torch.as_tensor(y[fit] / scale, dtype=torch.float32)
    model = PointTCN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    for _ in range(6):
        for batch in torch.randperm(len(target)).split(128):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(train_x[batch]), target[batch])
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        return torch.cat([model(b) for b in inputs[evaluate].split(128)]).numpy() * scale


def metrics(prediction, target, decision):
    error = (prediction - target) ** 2
    reference = target ** 2
    ratio = error.sum(axis=0) / reference.sum(axis=0)
    quarters = [utc(2024, month) for month in (1, 4, 7, 10)] + [utc(2025)]
    quarterly = []
    for left, right in zip(quarters, quarters[1:]):
        mask = (decision >= left) & (decision < right)
        quarterly.append(round(float(error[mask, -1].sum() / reference[mask, -1].sum()), 4))
    # Monday-to-Sunday UTC weeks, resampled as paired model/reference clusters.
    weeks = (decision // 86400 + 3) // 7
    by_week = np.array([[error[weeks == week, -1].sum(), reference[weeks == week, -1].sum()]
                        for week in np.unique(weeks)])
    rng = np.random.default_rng(37)
    ids = rng.integers(len(by_week), size=(2000, len(by_week)))
    totals = by_week[ids].sum(axis=1)
    interval = np.quantile(totals[:, 0] / totals[:, 1], [.025, .975])
    return dict(mse_ratio_h1_to_h6=ratio.round(4).tolist(), h6_quarters=quarterly,
                h6_week_bootstrap_95=interval.round(4).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--dll", type=Path)
    parser.add_argument("--tcn", action="store_true")
    args = parser.parse_args()
    x, y, decision, train, test = dataset(args.cache)
    print(json.dumps(dict(train=int(train.sum()), test=int(test.sum()))), flush=True)
    for name, inputs in (("ridge_target", x[:, ::len(SYMBOLS)]), ("ridge_all", x)):
        pred = ridge(inputs, y, train, test)
        print(name, json.dumps(metrics(pred, y[test], decision[test])), flush=True)
    if args.dll:
        for run in range(1, 4):
            pred = recurrent(x, y, train, test, args.dll)
            print(f"recurrent_{run}", json.dumps(metrics(pred, y[test], decision[test])), flush=True)
    if args.tcn:
        for seed in (37, 38, 39):
            pred = tcn_predict(x, y, train, test, seed)
            print(f"tcn_{seed}", json.dumps(metrics(pred, y[test], decision[test])), flush=True)


if __name__ == "__main__":
    main()
