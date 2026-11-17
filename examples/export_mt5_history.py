"""Read public market history from an existing connected MT5 terminal."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import time
import MetaTrader5 as mt


def export(a):
    start=datetime.fromisoformat(a.start).replace(tzinfo=timezone.utc)
    stop=datetime.fromisoformat(a.end).replace(tzinfo=timezone.utc)
    symbols=a.symbols.split(",")
    if not start<stop or len(set(symbols))!=len(symbols):raise ValueError("Invalid dates or symbols")
    if a.output.exists():raise FileExistsError(a.output)
    if not mt.initialize(path=str(a.terminal),timeout=15000):raise RuntimeError(f"Terminal unavailable: {mt.last_error()}")
    try:
        if not mt.terminal_info().connected:raise RuntimeError("Terminal is disconnected")
        rates=[]
        for s in symbols:
            for attempt in range(4):
                rows=mt.copy_rates_range(s,mt.TIMEFRAME_H1,start,stop)
                if rows is not None and len(rows):break
                mt.copy_rates_from_pos(s,mt.TIMEFRAME_H1,0,2000)
                time.sleep(1)
            if rows is None or not len(rows):raise ValueError(f"No H1 history for {s}: {mt.last_error()}")
            rates.append({int(r["time"])+3600:float(r["close"]) for r in rows
                          if start.timestamp()<=r["time"] and r["time"]+3600<=stop.timestamp()})
        a.output.parent.mkdir(exist_ok=True,parents=True)
        with tempfile.TemporaryDirectory(dir=a.output.parent) as tmp:
            dest=Path(tmp)/"history"
            dest.mkdir()
            with (dest/"bars.csv").open("w",newline="") as f:
                writer=csv.writer(f)
                writer.writerow(["close_time","available_at",*symbols])
                for t in sorted(rates[0]):
                    writer.writerow([t,t,*(m.get(t,"") for m in rates)])
            if a.ticks:
                count=0
                with (dest/"ticks.csv").open("w",newline="") as f:
                    writer=csv.writer(f)
                    writer.writerow(["time_msc","bid","ask"])
                    day=start
                    while day<stop:
                        end=min(day+timedelta(days=1),stop)
                        ticks=mt.copy_ticks_range(symbols[0],day,end,mt.COPY_TICKS_INFO)
                        if ticks is None:raise RuntimeError(f"Tick history unavailable {day}: {mt.last_error()}")
                        for t in ticks:
                            if day.timestamp()*1000<=t["time_msc"]<end.timestamp()*1000 and 0<t["bid"]<=t["ask"]:
                                writer.writerow([int(t["time_msc"]),float(t["bid"]),float(t["ask"])])
                                count+=1
                        day=end
                if not count:raise ValueError("No bid/ask quotes for the period")
                print("Bid/ask records:",count)
            if not rates[0]:raise ValueError("No completed target bars")
            dest.rename(a.output)
        print("Target H1 bars:",len(rates[0]),"UTC close availability assumed")
    finally:
        mt.shutdown()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("terminal",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--symbols",default="AUDUSD,NZDUSD,USDJPY,USDCAD")
    p.add_argument("--start",required=True,help="Inclusive UTC date")
    p.add_argument("--end",required=True,help="Exclusive UTC date")
    p.add_argument("--ticks",action="store_true")
    export(p.parse_args())
