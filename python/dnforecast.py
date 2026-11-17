"""DNForecast 3 C ABI. Requires Python 3.9+; inference has no pip dependencies."""
import ctypes as C
import hashlib
import os
from pathlib import Path
import time
from typing import NamedTuple

I, ID, D, F = C.c_int, C.c_int64, C.c_double, C.c_float
PI, PID, PD, PF = map(C.POINTER, (I, ID, D, F))
FIELDS = ("point", "q10", "q50", "q90", "p_up", "variance")
REFERENCE, ONNX_CPU, ONNX_CUDA = range(3)
PENDING, FULL, BUFFER, CANCELLED, SUPERSEDED = 1, -2, -3, -4, -5


class Forecast(NamedTuple):
    point: float
    q10: float
    q50: float
    q90: float
    p_up: float
    variance: float


def market_contract(symbols, seconds):
    symbols = tuple(symbols)
    if not 1 <= len(symbols) <= 16 or len(set(symbols)) != len(symbols) or seconds <= 0:
        raise ValueError("Invalid market list or timeframe")
    if any(not s or any(c in s for c in (",", "|", "\0")) for s in symbols):
        raise ValueError("Invalid market name")
    return f"dn3-v1|target-trading-bars|{seconds}|{','.join(symbols)}"


class Library:
    def __init__(self, path):
        path = Path(path).resolve(strict=True)
        if os.name == "nt":
            with os.add_dll_directory(str(path.parent)):
                self.api = C.CDLL(str(path))
        else:
            self.api = C.CDLL(str(path))
        signatures = {
            "GetAbiVersion": (I,), "GetCapabilities": (I,),
            "OpenSession": (ID, I, I, I, I, I, I, C.c_char_p, I, C.c_void_p, I),
            "StatusOf": (I, ID), "CloseSession": (I, ID),
            "Encode": (I, I, I, PD, I, PF, I), "Submit": (ID, ID, ID, PD, I),
            "TryRead": (I, ID, ID, PD, I, PI, PID), "Cancel": (I, ID, ID),
            "Release": (I, ID, ID), "Error": (I, ID, ID, C.c_void_p, I),
            "Metrics": (I, ID, PID, I),
        }
        for name, (result, *args) in signatures.items():
            function = getattr(self.api, "DN3_" + name)
            function.restype, function.argtypes = result, args
        if self.api.DN3_GetAbiVersion() != 30000:
            raise RuntimeError("Unsupported DNForecast ABI")

    def encode(self, steps, markets, prices):
        source = (D * len(prices))(*prices)
        out = (F * (steps * markets * 3))()
        if self.api.DN3_Encode(steps, markets, source, len(source), out, len(out)):
            raise ValueError("Invalid close window")
        return out


class Session:
    def __init__(self, library, symbols, seconds=3600, steps=96, horizons=6, capacity=8,
                 latest=False, model=None, cuda=False, timeout=60):
        self.api = library.api
        self.symbols = tuple(symbols)
        self.steps, self.horizons = steps, horizons
        contract = market_contract(self.symbols, seconds).encode()
        data = Path(model).read_bytes() if model is not None else b""
        if model is not None and not data:
            raise ValueError("Empty ONNX model")
        if cuda and model is None:
            raise ValueError("CUDA requires an ONNX model")
        self.generation = hashlib.sha256(data).hexdigest() if data else "reference-3.0"
        backend = ONNX_CUDA if cuda else ONNX_CPU if data else REFERENCE
        if data.startswith(b"DN3V2\n"):
            backend = 3
        self.id = self.api.DN3_OpenSession(steps, len(self.symbols), horizons, capacity, int(latest),
                                           backend, contract, len(contract), data, len(data))
        if self.id <= 0:
            raise ValueError(f"OpenSession failed ({self.id})")
        try:
            deadline = time.monotonic() + timeout
            while (status := self.api.DN3_StatusOf(self.id)) == PENDING:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Loading the model timed out")
                time.sleep(.001)
            if status != 0:
                raise RuntimeError(self.error(0))
        except BaseException:
            self.close()
            raise

    def error(self, job):
        size = self.api.DN3_Error(self.id, job, None, 0)
        if size <= 0:
            return f"Unknown session or job ({size})"
        buf = C.create_string_buffer(size)
        self.api.DN3_Error(self.id, job, buf, size)
        return buf.value.decode("utf8", "replace")

    def submit(self, origin, prices):
        source = (D * len(prices))(*prices)
        job = self.api.DN3_Submit(self.id, origin, source, len(source))
        if job <= 0:
            raise RuntimeError(f"Submit failed ({job})")
        return job

    def read(self, job):
        output = (D * (self.horizons * len(FIELDS)))()
        required, origin = I(), ID()
        code = self.api.DN3_TryRead(self.id, job, output, len(output), C.byref(required), C.byref(origin))
        if code == PENDING:
            return None
        if code != 0:
            raise RuntimeError(f"Job {job}: {code} {self.error(job)}")
        return origin.value, [Forecast(*output[i:i+6]) for i in range(0, required.value, 6)]

    def predict(self, origin, prices, timeout=10):
        job = self.submit(origin, prices)
        try:
            deadline = time.monotonic() + timeout
            while (result := self.read(job)) is None:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Prediction {job} timed out")
                time.sleep(.0001)
            return result[1]
        finally:
            self.release(job)

    def release(self, job):
        return self.api.DN3_Release(self.id, job)

    def close(self):
        if self.id > 0:
            self.api.DN3_CloseSession(self.id)
            self.id = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
