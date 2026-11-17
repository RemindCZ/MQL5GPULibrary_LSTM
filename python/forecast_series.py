"""Point-in-time input for replay, fitting and execution analysis."""
import csv
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Bar:
    close_time: int
    available: int
    prices: tuple


def read_bars(path, symbols):
    """One frozen snapshot per closed target bar. Blank driver means unavailable."""
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"close_time", "available_at", *symbols} <= set(reader.fieldnames or ()):
            raise ValueError("Missing timestamp or symbol columns")
        for line, r in enumerate(reader, 2):
            event, available = int(r["close_time"]), int(r["available_at"])
            prices = tuple(float(r[s]) if r[s] else 0. for s in symbols)
            if (event <= 0 or available < event or
                (rows and (event <= rows[-1].close_time or available < rows[-1].available)) or
                prices[0] <= 0 or any(p < 0 or not math.isfinite(p) for p in prices)):
                raise ValueError(f"Invalid frozen close snapshot at line {line}")
            rows.append(Bar(event, available, prices))
    return rows


def window(bars, end, steps):
    if end < steps or end >= len(bars):
        raise ValueError("Incomplete window")
    return [p for bar in bars[end-steps:end+1] for p in bar.prices]


def losses(f, realized, rv):
    delta = realized - f.point
    score = lambda q, v: (realized-v)*(q-(realized<v))
    return (delta*delta, (f.p_up-(realized>0))**2,
            (score(.1,f.q10)+score(.5,f.q50)+score(.9,f.q90))/3,
            math.log(f.variance)+rv/f.variance,
            float(f.q10<=realized<=f.q90), f.q90-f.q10)
