"""Issue one forecast from currently closed H1 bars in a running MT5 terminal."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import MetaTrader5 as mt

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from dnforecast import Library,Session


def snapshot(terminal,symbols,steps,server,portable=False):
    if not mt.initialize(path=str(terminal),timeout=15000,portable=portable):
        raise RuntimeError(f"MT5 connection failed: {mt.last_error()}")
    try:
        info=mt.terminal_info()
        if not info or not info.connected:raise RuntimeError("Terminal is offline")
        active=mt.account_info().server
        if server and active!=server:raise ValueError(f"Active server is {active}, expected {server}")
        markets=[]
        for s in symbols:
            if not mt.symbol_select(s,True):raise RuntimeError(f"Cannot subscribe to {s}")
            for attempt in range(8):
                rows=mt.copy_rates_from_pos(s,mt.TIMEFRAME_H1,1,steps+1)
                if rows is not None and len(rows)>=steps+1:break
                time.sleep(.25)
            if rows is None or len(rows)==0:raise RuntimeError(f"Missing closed bars for {s}")
            markets.append({int(r["time"]):float(r["close"]) for r in rows})
        times=sorted(markets[0])
        if len(times)!=steps+1:raise ValueError("Incomplete target window")
        current=mt.copy_rates_from_pos(symbols[0],mt.TIMEFRAME_H1,0,2)
        for attempt in range(40):
            tick=mt.symbol_info_tick(symbols[0])
            if tick is not None and 0<tick.bid<=tick.ask:break
            time.sleep(.25)
        if tick is None or not (0<tick.bid<=tick.ask) or current is None or len(current)!=2:
            raise RuntimeError("Missing current quote or bar")
        if int(current[0]["time"])!=times[-1] or times[-1]+3600>tick.time:
            raise RuntimeError("Target bar rolled over during collection; retry")
        prices=[market.get(t,0.) for t in times for market in markets]
        return prices,dict(server=active,symbol=symbols[0],origin_close=times[-1]+3600,
                           observed_tick_msc=tick.time_msc,bid=tick.bid,ask=tick.ask,anchor=prices[-len(symbols)],
                           closes_available=[sum(t in m for t in times) for m in markets])
    finally:
        mt.shutdown()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("terminal",type=Path)
    p.add_argument("dll",type=Path)
    p.add_argument("--symbols",default="AUDUSD,NZDUSD,USDJPY,USDCAD")
    p.add_argument("--expected-server")
    p.add_argument("--portable",action="store_true")
    p.add_argument("--model",type=Path)
    a=p.parse_args()
    symbols=a.symbols.split(",")
    prices,report=snapshot(a.terminal,symbols,96,a.expected_server,a.portable)
    with Session(Library(a.dll),symbols,model=a.model) as session:
        forecasts=session.predict(report["origin_close"],prices)
        report.update(generation=session.generation,issued_wall_utc_ms=time.time_ns()//1000000,
                      forecast_close=[report["anchor"]*math.exp(r.point) for r in forecasts])
        print(json.dumps(report,indent=2,allow_nan=False))
