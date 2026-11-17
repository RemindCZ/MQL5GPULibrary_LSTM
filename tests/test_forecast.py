"""DN3 contract and event-time regressions, including the production shared library."""
import ctypes as C
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"python"),str(ROOT/"examples")]
from dnforecast import Library, Session, D, I, ID, market_contract
from forecast_series import Bar, window
from replay_forecast import replay
from backtest_strategy import simulate

DLL=Path(sys.argv.pop(1))
LIB=Library(DLL)
SYMBOLS=["AUDUSD","NZDUSD","USDJPY","USDCAD"]

def wait(session,job):
    limit=time.monotonic()+5
    while time.monotonic()<limit:
        result=session.read(job)
        if result is not None:return result
    raise TimeoutError(job)


class RuntimeTests(unittest.TestCase):
    def test_input_copy_and_retention(self):
        with Session(LIB,["A"],steps=16,horizons=2,capacity=1) as s:
            src=(D*17)(*([1.]*17))
            job=s.api.DN3_Submit(s.id,49,src,17)
            src[7]=0
            self.assertEqual(s.api.DN3_Submit(s.id,50,src,17),-2)
            self.assertEqual(wait(s,job)[0],49)
            n,o=I(),ID()
            self.assertEqual(s.api.DN3_TryRead(s.id,job,None,0,C.byref(n),C.byref(o)),-3)
            self.assertEqual((n.value,o.value),(12,49))
            self.assertEqual(s.read(job)[1][0],(0.,0.,0.,0.,.5,1e-16))
            self.assertEqual(s.release(job),0)
            self.assertEqual(s.release(job),0)
            invalid=s.api.DN3_Submit(s.id,50,src,17)
            with self.assertRaisesRegex(RuntimeError,"Invalid close"):wait(s,invalid)

    def test_masks(self):
        prices=[p for t in range(17) for p in (1.,0. if t in (5,6) else 2.)]
        x=LIB.encode(16,2,prices)
        at=lambda t:list(x[(t-1)*6+3:t*6])
        self.assertEqual(at(5),[0.,0.,1/16])
        self.assertEqual(at(6),[0.,0.,2/16])
        self.assertEqual(at(7),[0.,0.,0.])
        self.assertEqual(at(8),[0.,1.,0.])

    def test_parallel_lifecycle(self):
        def client(k):
            for r in range(4):
                with Session(LIB,["A"],steps=64,horizons=8,latest=True,capacity=32) as s:
                    jobs=[s.submit(k*100+i+1,[1.+k*.01]*65) for i in range(30)]
                    self.assertEqual(wait(s,jobs[-1])[0],k*100+30)
                    for job in jobs[:-1]:
                        rc=s.api.DN3_Cancel(s.id,job)
                        self.assertIn(rc,(0,-4,-5))
                        self.assertEqual(s.release(job),0)
                    old=s.id
                self.assertEqual(LIB.api.DN3_StatusOf(old),-7)
        with ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(client,range(8)))

    def test_onnx(self):
        if not LIB.api.DN3_GetCapabilities()&2:self.skipTest("SDK disabled")
        path=ROOT/"models/AUDUSD-H1-TCN.onnx"
        with Session(LIB,SYMBOLS,model=path) as s:
            out=s.predict(1,[1.]*388)
            self.assertEqual(len(out),6)
            self.assertTrue(all(r.q10<=r.q50<=r.q90 and r.variance>0 for r in out))
        with self.assertRaisesRegex(RuntimeError,"contract mismatch"):
            Session(LIB,SYMBOLS[::-1],model=path)
        with self.assertRaisesRegex(RuntimeError,"dimensions"):
            Session(LIB,SYMBOLS,steps=32,model=path)
        with tempfile.TemporaryDirectory() as folder:
            bad=Path(folder)/"bad.onnx";bad.write_bytes(b"invalid")
            with self.assertRaises(RuntimeError):Session(LIB,SYMBOLS,model=bad)

    def test_replay_and_revision(self):
        rows=[Bar(100+i,101+i,(1.+i*.01,)) for i in range(28)]
        with Session(LIB,["A"],steps=16,horizons=2) as s, sqlite3.connect(":memory:") as db:
            replay(db,s,rows[:22],"A",0)
            saved=db.execute("SELECT * FROM forecasts ORDER BY origin,h").fetchall()
            replay(db,s,rows,"A",0)
            self.assertEqual(saved,db.execute("SELECT * FROM forecasts WHERE origin<122 ORDER BY origin,h").fetchall())
            self.assertEqual(db.execute("SELECT count(*) FROM outcomes").fetchone()[0],21)
            replay(db,s,rows,"A",0)
            revised=rows.copy();revised[20]=Bar(120,121,(2.,))
            with self.assertRaisesRegex(ValueError,"Revised"):replay(db,s,revised,"A",0)

    def test_task_policy(self):
        if not LIB.api.DN3_GetCapabilities()&2:self.skipTest("SDK disabled")
        try:
            from select_tasks import select
        except ImportError:
            self.skipTest("Optional ONNX authoring tools unavailable")
        original=ROOT/"models/AUDUSD-H1-TCN.onnx"
        prices=[1.+.001*math.sin(i) for i in range(388)]
        masks=[0,1,2,4,8,15]
        with tempfile.TemporaryDirectory() as folder:
            frozen=Path(folder)/"policy.onnx"
            select(original,frozen,masks)
            with Session(LIB,SYMBOLS) as base, Session(LIB,SYMBOLS,model=original) as raw, Session(LIB,SYMBOLS,model=frozen) as active:
                baseline,neural,policy=[s.predict(100,prices) for s in (base,raw,active)]
                for h,mask in enumerate(masks):
                    self.assertEqual(policy[h],tuple((neural if mask&bit else baseline)[h][j]
                        for j,bit in enumerate((1,2,2,2,4,8))))

    def test_execution_timing(self):
        bars=[Bar(i,i,(100.,)) for i in (100,101,102,103,104)]
        ticks=[(100000,99,100),(100001,100,101),(101000,109,110),(102000,109,110)]
        result=simulate(bars,[(100,100,.1)],ticks,1,100,0,0,0,5000)
        self.assertAlmostEqual(result["net_return"],109/101-1)
        self.assertEqual(result["closed"],1)
        cost=simulate(bars,[(100,100,.1)],ticks,1,100,1,2,0,5000)
        self.assertLess(cost["net_return"],result["net_return"])
        short=simulate(bars,[(100,100,-.1)],ticks,1,100,0,0,0,5000)
        self.assertAlmostEqual(short["net_return"],-.10)
        self.assertEqual(simulate(bars,[(100,100,0)],ticks,1,100,0,0,0,100)["opened"],0)
        # No entry on the quote used to issue, or after the expiry of the signal.
        self.assertEqual(simulate(bars,[(100,100,.1)],ticks[::2],1,100,0,0,0,50)["opened"],0)


if __name__=="__main__":unittest.main()
