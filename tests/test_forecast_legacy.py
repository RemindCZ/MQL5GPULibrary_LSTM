"""Compare the legacy checkpoint adapter against synchronous CUDA inference."""
import json
import math
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/"python"),str(ROOT/"examples")]
from mql5gpu import Library as LegacyLibrary, Model
from dnforecast import Library, Session
from pack_legacy import pack

if __name__=="__main__":
    for path in (Path(sys.argv[3]),Path(sys.argv[3])/"x64"):
        if path.is_dir():
            # Keep registered directories until model destruction.
            globals().setdefault("dll_directories",[]).append(os.add_dll_directory(str(path.resolve())))
    lib=Library(sys.argv[1])
    s=24
    prices=[1+.001*math.sin(i) for i in range(s+1)]
    x=[math.log(b/a) for a,b in zip(prices,prices[1:])]
    with tempfile.TemporaryDirectory() as tmp:
        folder=Path(tmp)
        with Model(LegacyLibrary(sys.argv[2])) as old:
            old.configure(1,s,3,"lstm:4:linear:0:concat:-1")
            old.save(folder/"weights.txt")
            expected=old.predict([x])[0]
        meta=dict(schema=1,symbols=["A"],mean=[0],scale=[1],target_scale=100)
        (folder/"metadata.json").write_text(json.dumps(meta))
        bundle=folder/"legacy.dn2"
        bundle.write_bytes(pack(folder,3600))
        with Session(lib,["A"],steps=s,horizons=3,model=bundle) as session:
            for f,v in zip(session.predict(1,prices),expected):
                assert abs(f.point-v/100)<1e-8
    print("Legacy checkpoint: all horizons agree with synchronous CUDA")
