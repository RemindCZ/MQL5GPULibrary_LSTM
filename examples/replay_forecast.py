"""Replay frozen close snapshots into an idempotent forecast/outcome ledger."""
import argparse
import hashlib
import math
from pathlib import Path
import sqlite3
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
from dnforecast import Forecast, Library, Session, market_contract
from forecast_series import losses, read_bars, window


def replay(db, session, bars, contract, start):
    db.execute("""CREATE TABLE IF NOT EXISTS forecasts
        (series TEXT, origin INTEGER, issued INTEGER, h INTEGER, anchor REAL, point REAL,
         q10 REAL, q50 REAL, q90 REAL, p_up REAL, variance REAL, snapshot TEXT,
         PRIMARY KEY(series,origin,h))""")
    db.execute("""CREATE TABLE IF NOT EXISTS outcomes
        (series TEXT, origin INTEGER, h INTEGER, maturity INTEGER, observed INTEGER, price REAL,
         mse REAL, brier REAL, pinball REAL, qlike REAL, covered REAL, width REAL,
         PRIMARY KEY(series,origin,h))""")
    series = f"{contract}|{session.steps}|{session.horizons}|{session.generation}"
    for i in range(session.steps, len(bars)):
        b = bars[i]
        if b.close_time < start:
            continue
        prices = window(bars, i, session.steps)
        digest = hashlib.sha256(struct.pack(f"<{len(prices)}d", *prices)).hexdigest()
        with db:
            existing = db.execute("SELECT issued,snapshot FROM forecasts WHERE series=? AND origin=? AND h=1",
                                  (series,b.close_time)).fetchone()
            if existing and existing != (b.available, digest):
                raise ValueError("Revised origin: replay into a separate database")
            if not existing:
                predictions = session.predict(b.close_time, prices)
                db.executemany("INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    [(series,b.close_time,b.available,h,b.prices[0],*forecast,digest)
                     for h,forecast in enumerate(predictions,1)])
            # Only publish a realization after this target bar was available.
            rv = 0.
            for h in range(1,min(session.horizons, i)+1):
                rv += math.log(bars[i-h+1].prices[0]/bars[i-h].prices[0])**2
                origin = bars[i-h]
                row = db.execute("""SELECT issued,point,q10,q50,q90,p_up,variance FROM forecasts
                    WHERE series=? AND origin=? AND h=?""",(series,origin.close_time,h)).fetchone()
                if row is None or row[0] >= b.available:
                    continue
                r = math.log(b.prices[0]/origin.prices[0])
                outcome = (series,origin.close_time,h,b.close_time,b.available,b.prices[0],*losses(Forecast(*row[1:]),r,rv))
                existing_outcome = db.execute("SELECT * FROM outcomes WHERE series=? AND origin=? AND h=?",outcome[:3]).fetchone()
                if existing_outcome is None:
                    db.execute("INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",outcome)
                elif existing_outcome != outcome:
                    raise ValueError("Revised outcome: replay into a separate database")
    return series


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dll","bars","ledger"):
        p.add_argument(name, type=Path)
    p.add_argument("--symbols",required=True)
    p.add_argument("--seconds",type=int,default=3600)
    p.add_argument("--steps",type=int,default=96)
    p.add_argument("--horizons",type=int,default=6)
    p.add_argument("--start",type=int,default=0,help="Earliest origin, Unix seconds")
    p.add_argument("--model",type=Path)
    args = p.parse_args()
    symbols = args.symbols.split(",")
    bars = read_bars(args.bars,symbols)
    with Session(Library(args.dll),symbols,args.seconds,args.steps,args.horizons,model=args.model) as session:
        with sqlite3.connect(args.ledger) as db:
            series = replay(db,session,bars,market_contract(symbols,args.seconds),args.start)
            print(series)
            print("h, n_scored, mse, brier, pinball, qlike, coverage, width")
            for row in db.execute("""SELECT h,count(*),avg(mse),avg(brier),avg(pinball),avg(qlike),avg(covered),avg(width)
                     FROM outcomes WHERE series=? GROUP BY h ORDER BY h""",(series,)):
                print(*row,sep=",")
