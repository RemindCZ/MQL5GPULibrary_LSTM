"""Convert the four pinned pilot archives to DN3 bar snapshots (assumed close availability)."""
import argparse
import csv
from pathlib import Path
import duckdb
from audusd_h1_pilot import SYMBOLS, read_market

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("cache",type=Path)
    p.add_argument("output",type=Path)
    args=p.parse_args()
    with duckdb.connect() as db:
        markets=[read_market(db,args.cache,s,stop_year=2026) for s in SYMBOLS]
    with args.output.open("x",encoding="utf-8",newline="") as f:
        writer=csv.writer(f)
        writer.writerow(["close_time","available_at",*SYMBOLS])
        for stamp in sorted(markets[0]):
            writer.writerow([stamp+3600,stamp+3600,*(m.get(stamp,"") for m in markets)])
