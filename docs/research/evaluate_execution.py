"""Fixed six-bar October 2026 IC Trading execution test. Input quotes stay local."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/"python"),str(ROOT/"examples")]
from dnforecast import Library,Session,market_contract
from forecast_series import read_bars
from replay_forecast import replay
from backtest_strategy import simulate,quotes

def evaluate(dll,data,model):
    symbols=["AUDUSD","NZDUSD","USDJPY","USDCAD"]
    start=1790812800
    bars=read_bars(data/"bars.csv",symbols)
    report={"start":start,"end_exclusive":1793491200,"horizon":6,
            "commission_bps_per_side":.5,"slippage_bps_per_side":.5,"max_wait_ms":60000,
            "availability":"assumed at bar close, prices and quotes on terminal time axis",
            "symbols":symbols,"model_sha256":hashlib.sha256(model.read_bytes()).hexdigest(),
            "target_bars_including_warmup":len(bars),
            "available_closes":[sum(bool(b.prices[m]) for b in bars) for m in range(4)]}
    for file in ("bars.csv","ticks.csv"):
        with (data/file).open("rb") as f:report[file+"_sha256"]=hashlib.file_digest(f,"sha256").hexdigest()
    lib=Library(dll)
    for name,path in (("reference",None),("tcn",model)):
        with Session(lib,symbols,model=path) as session,sqlite3.connect(":memory:") as db:
            series=replay(db,session,bars,market_contract(symbols,3600),start)
            cursor=db.execute("SELECT origin,issued,point FROM forecasts WHERE series=? AND h=6 ORDER BY issued,origin",(series,))
            result=simulate(bars,cursor,quotes(data/"ticks.csv"),6,start,.5,.5,0,60000)
            result["issued"],=db.execute("SELECT count(*) FROM forecasts WHERE h=6").fetchone()
            result["scored"],result["mse"],result["brier"],result["coverage"]=db.execute(
                "SELECT count(*),avg(mse),avg(brier),avg(covered) FROM outcomes WHERE h=6").fetchone()
            report[name]=result
    return report

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("dll","data","model","output"):p.add_argument(name,type=Path)
    a=p.parse_args()
    report=evaluate(a.dll,a.data,a.model)
    a.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report,indent=2))
