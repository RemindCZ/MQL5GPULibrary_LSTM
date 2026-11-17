"""Wrap a 2.3 financial model directory for DN3 asynchronous inference."""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from dnforecast import market_contract


def pack(directory,seconds):
    meta=json.loads((directory/"metadata.json").read_text(encoding="utf-8"))
    symbols=meta["symbols"]
    mean,scale=meta["mean"],meta["scale"]
    if (meta["schema"]!=1 or len(mean)!=len(symbols) or len(scale)!=len(symbols)
        or any(not math.isfinite(v) for v in (*mean,*scale,meta["target_scale"]))
        or min(*scale,meta["target_scale"])<=0):
        raise ValueError("Invalid version 2 model metadata")
    lines=["DN3V2",market_contract(symbols,seconds),str(meta["target_scale"]),
           " ".join(map(str,mean))," ".join(map(str,scale))]
    return ("\n".join(lines)+"\n").encode()+(directory/"weights.txt").read_bytes()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--seconds",type=int,default=3600)
    a=p.parse_args()
    data=pack(a.directory,a.seconds)
    with a.output.open("xb") as f:
        f.write(data)
