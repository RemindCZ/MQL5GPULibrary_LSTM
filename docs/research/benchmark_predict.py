"""Measure the host-side duration of DN_PredictBatch on the default 2.3 graph."""
import argparse
import ctypes as C
import hashlib
import math
from pathlib import Path
import statistics
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
from mql5gpu import Library, Model


def measure(model, batch):
    width, horizons = 96, 6
    x = (C.c_double * (batch * width))(*(math.sin(i * 0.13) for i in range(batch * width)))
    y = (C.c_double * (batch * horizons))()

    def timed_call():
        start = perf_counter_ns()
        ok = model.api.DN_PredictBatch(model.handle, x, batch, width, 0, y)
        duration_ms = (perf_counter_ns() - start) / 1e6
        model.library.check(ok)
        if not all(map(math.isfinite, y)):
            raise ValueError("Non-finite output")
        return duration_ms

    for _ in range(20):
        timed_call()
    for repeat in (1, 2):
        times = sorted(timed_call() for _ in range(100))
        throughput = batch * len(times) * 1000 / sum(times)
        print(f"{batch:5} {repeat:6} {statistics.median(times):8.3f} {times[94]:8.3f} {throughput:10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dll", type=Path)
    args = parser.parse_args()
    library = Library(args.dll)
    print("DLL", library.api.DN_GetVersion(), hashlib.sha256(args.dll.read_bytes()).hexdigest())
    print("batch series   p50_ms   p95_ms  windows/s")
    with Model(library) as model:
        model.configure(4, 24, 6)
        for batch_size in (1, 8, 32):
            measure(model, batch_size)
