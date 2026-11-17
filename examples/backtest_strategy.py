"""Single-position, unlevered execution replay on recorded bid/ask quotes."""
import argparse
import csv
import json
import math
from pathlib import Path
import sqlite3
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from forecast_series import read_bars


def quotes(path):
    with open(path,encoding="utf-8-sig",newline="") as f:
        last = -1
        for r in csv.DictReader(f):
            ts, bid, ask = int(r["time_msc"]), float(r["bid"]), float(r["ask"])
            if ts < last or ts <= 0 or not (0 < bid <= ask and math.isfinite(ask)):
                raise ValueError("Invalid or unordered bid/ask quote")
            last = ts
            yield ts,bid,ask


def simulate(bars, signals, ticks, horizon, start, commission_bps, slippage_bps, edge_bps, max_wait_ms):
    if not (1 <= horizon <= 64 and max_wait_ms > 0 and
            all(math.isfinite(x) and 0 <= x < 1000 for x in (commission_bps,slippage_bps,edge_bps))):
        raise ValueError("Invalid execution assumptions")
    fee, slip, edge = [v/10000 for v in (commission_bps,slippage_bps,edge_bps)]
    by_time = {b.close_time:i for i,b in enumerate(bars)}
    stream = iter(signals)
    signal = next(stream,None)
    balance = peak = 1.
    drawdown = pos = entry = entry_fee = 0.
    due = None
    pending = previous = None
    trades = []
    opened = wins = expired = seen = 0
    total_fees = turnover = 0.
    final_equity = balance
    for now,bid,ask in ticks:
        if now < start*1000:
            previous=(now,bid,ask)
            continue
        # Only quotes already observed when issuing a signal can set its threshold.
        while signal is not None and signal[1]*1000 < now:
            origin, issued, point = signal
            seen+=1
            if origin not in by_time or not math.isfinite(point):
                raise ValueError("Signal and target series mismatch")
            i = by_time[origin]
            maturity = bars[i+horizon].available*1000 if i+horizon<len(bars) else None
            if not pos and pending is None and previous is not None and issued>=start:
                qt,qb,qa = previous
                if 0 <= issued*1000-qt <= max_wait_ms:
                    cost = math.log(qa/qb)+2*(fee+slip)+edge
                    if abs(point)>cost:
                        pending=(issued*1000,1 if point>0 else -1,maturity)
            signal=next(stream,None)
        if pos and due is not None and now>=due:
            exit_price = bid*(1-slip) if pos>0 else ask*(1+slip)
            exit_fee = abs(pos)*exit_price*fee
            change = pos*(exit_price-entry)-exit_fee
            balance += change
            total_fees += exit_fee
            turnover += abs(pos)*exit_price
            pnl = change-entry_fee
            trades.append(pnl)
            wins += int(pnl>0)
            pos=0.
        if pending is not None:
            issued,side,maturity=pending
            if now-issued > max_wait_ms or (maturity is not None and now>=maturity):
                expired+=1
            elif balance>0:
                entry=ask*(1+slip) if side>0 else bid*(1-slip)
                pos=side*balance/(entry*(1+fee))
                entry_fee=abs(pos)*entry*fee
                balance-=entry_fee
                total_fees+=entry_fee
                turnover+=abs(pos)*entry
                due=maturity
                opened+=1
            pending=None
        # Liquidation value includes the closing spread, slippage and commission.
        mark=bid*(1-slip) if pos>0 else ask*(1+slip)
        final_equity=balance+pos*(mark-entry)-abs(pos)*mark*fee
        peak=max(peak,final_equity)
        drawdown=max(drawdown,(peak-final_equity)/peak)
        previous=(now,bid,ask)
    return dict(net_return=final_equity-1,max_drawdown=drawdown,signals_observed=seen,opened=opened,closed=len(trades),
                open_at_end=bool(pos),wins=wins,expired_entries=expired,commission_paid=total_fees,
                turnover_initial_equity=turnover,flat_return=0.)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("ledger","bars","ticks"):
        p.add_argument(name,type=Path)
    p.add_argument("--series",required=True,help="Exact series ID printed by replay_forecast")
    p.add_argument("--start",type=int,required=True,help="First out-of-sample issuance time in Unix seconds")
    p.add_argument("--horizon",type=int,default=6)
    p.add_argument("--commission-bps",type=float,required=True,help="Per side, on executed notional")
    p.add_argument("--slippage-bps",type=float,required=True,help="Adverse per side")
    p.add_argument("--edge-bps",type=float,default=0)
    p.add_argument("--max-wait-ms",type=int,default=60000)
    a=p.parse_args()
    symbols=a.series.split("|")[3].split(",")
    with sqlite3.connect(f"{a.ledger.resolve().as_uri()}?mode=ro",uri=True) as db:
        if not db.execute("SELECT 1 FROM forecasts WHERE series=? AND h=? AND issued>=? LIMIT 1",
                          (a.series,a.horizon,a.start)).fetchone():
            p.error("Missing series, horizon or evaluation period")
        signals=db.execute("SELECT origin,issued,point FROM forecasts WHERE series=? AND h=? AND issued>=? ORDER BY issued,origin",
                           (a.series,a.horizon,a.start))
        print(json.dumps(simulate(read_bars(a.bars,symbols),signals,quotes(a.ticks),a.horizon,a.start,
                                  a.commission_bps,a.slippage_bps,a.edge_bps,a.max_wait_ms),indent=2))
