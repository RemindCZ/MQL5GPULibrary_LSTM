"""Measure submission and result observation through the C ABI in synchronized chart waves."""
import argparse
import ctypes as C
from contextlib import ExitStack
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"python"))
from dnforecast import D,ID,I,Library,Session


def bench(lib,model,clients,count):
    symbols=["AUDUSD","NZDUSD","USDJPY","USDCAD"]
    sample=(D*(97*4))(*(1.+.001*math.sin(i) for i in range(97*4)))
    out=(D*36)();needed=I();origin=ID()
    submits=[];ends=[]
    with ExitStack() as contexts:
        t=time.perf_counter_ns()
        sessions=[contexts.enter_context(Session(lib,symbols,model=model)) for _ in range(clients)]
        load_ms=(time.perf_counter_ns()-t)/1e6
        for s in sessions:s.predict(1,sample)
        start=time.perf_counter_ns()
        while len(ends)<count:
            outstanding=[]
            for j,s in enumerate(sessions[:count-len(ends)]):
                issued=len(ends)+j+10
                t=time.perf_counter_ns()
                job=lib.api.DN3_Submit(s.id,issued,sample,len(sample))
                submits.append((time.perf_counter_ns()-t)/1000)
                if job<=0:raise RuntimeError(f"Submit failed {job}")
                outstanding.append((s,issued,t,job))
            while outstanding:
                pending=[]
                for s,issued,t,job in outstanding:
                    state=lib.api.DN3_TryRead(s.id,job,out,len(out),C.byref(needed),C.byref(origin))
                    if state==1:
                        pending.append((s,issued,t,job))
                    elif state==0 and origin.value==issued:
                        ends.append((time.perf_counter_ns()-t)/1000)
                        s.release(job)
                    else:raise RuntimeError(f"Wrong result {state}, {origin.value}, {issued}")
                outstanding=pending
        elapsed=(time.perf_counter_ns()-start)/1e9
    def stats(xs):
        xs.sort()
        return {f"p{p}":round(xs[math.ceil(len(xs)*p/100)-1],2) for p in (50,95,99)}
    return dict(sessions=clients,requests=count,load_all_ms=round(load_ms,2),submit_us=stats(submits),
                observed_us=stats(ends),jobs_per_second=round(count/elapsed,1))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("dll",type=Path)
    p.add_argument("--model",type=Path)
    p.add_argument("--count",type=int,default=10000)
    a=p.parse_args()
    if a.count<1:p.error("Positive count required")
    lib=Library(a.dll)
    print(json.dumps([bench(lib,a.model,c,a.count) for c in (1,8,32)],indent=2))
