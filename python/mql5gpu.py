"""Standalone, standard-library ctypes client for the Windows CUDA DLL."""
import ctypes as C
import os
from pathlib import Path
import time

DEFAULT_TOPOLOGY = 'lstm:32:linear:0.10:concat:-1;gru:16:linear:0.10:concat:-1;dense:24:tanh:0:concat:0,1'


class Library:
    def __init__(self, path, cuda_bin=None):
        if os.name != 'nt':
            raise OSError('The DLL requires 64-bit Windows and an NVIDIA GPU')
        path = Path(path).resolve()
        dirs = [path.parent]
        if cuda_bin:
            dirs += [Path(cuda_bin), Path(cuda_bin) / 'x64']
        elif os.environ.get('CUDA_PATH'):
            dirs += [Path(os.environ['CUDA_PATH']) / 'bin', Path(os.environ['CUDA_PATH']) / 'bin/x64']
        self._directories = [os.add_dll_directory(str(p)) for p in dirs if p.is_dir()]
        self.api = C.WinDLL(str(path))
        i, d, ptr = C.c_int, C.c_double, C.POINTER
        definitions = {
            'DN_GetVersion': (i, []), 'DN_Create': (i, []), 'DN_Free': (None, [i]),
            'DN_GetError': (None, [ptr(C.c_wchar), i]),
            'DN_BeginGraph': (i, [i,i,i]),
            'DN_AddGraphNode': (i, [i,i,i,i,d,ptr(i),i,i]),
            'DN_SetGraphOutput': (i, [i,i,i]),
            'DN_SetMiniBatchSize': (i, [i,i]), 'DN_SetGradClip': (i, [i,d]),
            'DN_GetInputDim': (i, [i]), 'DN_GetOutputDim': (i, [i]), 'DN_GetSequenceLength': (i, [i]),
            'DN_LoadBatch': (i, [i,ptr(d),ptr(d),i,i,i,i]),
            'DN_PredictBatch': (i, [i,ptr(d),i,i,i,ptr(d)]),
            'DN_TrainAsync': (i, [i,i,d,d,d]), 'DN_GetTrainingStatus': (i, [i]),
            'DN_StopTraining': (None, [i]), 'DN_GetTrainingResult': (None, [i,ptr(d),ptr(i)]),
            'DN_SaveState': (i, [i]), 'DN_GetState': (i, [i,ptr(C.c_char),i]),
            'DN_LoadState': (i, [i,C.c_char_p]),
        }
        for name, (result, args) in definitions.items():
            fn = getattr(self.api, name)
            fn.restype, fn.argtypes = result, args
        if self.api.DN_GetVersion() < 20300:
            raise RuntimeError('DLL 2.3.0 or newer required')

    def error(self):
        buf = C.create_unicode_buffer(1024)
        self.api.DN_GetError(buf, len(buf))
        return buf.value

    def check(self, result):
        if result != 1:
            raise RuntimeError(self.error() or 'DLL operation failed')


class Model:
    def __init__(self, library):
        self.library = library
        self.api = library.api
        self.handle = self.api.DN_Create()
        if self.handle <= 0:
            raise RuntimeError(library.error() or 'Cannot initialize CUDA model')

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

    def close(self):
        if self.handle:
            self.api.DN_Free(self.handle)
            self.handle = 0

    def configure(self, features, sequence, outputs, topology=DEFAULT_TOPOLOGY):
        nodes = []
        kinds = dict(lstm=0, gru=1, rnn=2, dense=3, identity=4)
        activations = dict(linear=0, tanh=1, relu=2, sigmoid=3)
        merges = dict(concat=0, sum=1)
        # Parse before changing the model so syntax errors preserve it.
        for entry in topology.split(';'):
            kind, width, activation, drop, merge, parents = [x.strip() for x in entry.split(':')]
            nodes.append((kinds[kind], int(width), activations[activation], float(drop),
                          [int(x) for x in parents.split(',')], merges[merge]))
        self.library.check(self.api.DN_BeginGraph(self.handle, features, sequence))
        last = -1
        for kind, width, activation, drop, parents, merge in nodes:
            ids = (C.c_int * len(parents))(*parents)
            last = self.api.DN_AddGraphNode(self.handle, kind, width, activation, drop, ids, len(ids), merge)
            if last < 0:
                raise RuntimeError(self.library.error())
        self.library.check(self.api.DN_SetGraphOutput(self.handle, last, outputs))
        return self

    @property
    def dimensions(self):
        values = (self.api.DN_GetInputDim(self.handle), self.api.DN_GetSequenceLength(self.handle),
                  self.api.DN_GetOutputDim(self.handle))
        if min(values) <= 0:
            raise RuntimeError('Model is unconfigured, closed or busy')
        return values

    @staticmethod
    def _buffer(rows, width):
        if not rows or any(len(row) != width for row in rows):
            raise ValueError(f'Expected nonempty rows of width {width}')
        return (C.c_double * (len(rows)*width))(*(x for row in rows for x in row))

    def fit(self, x, targets, epochs=60, learning_rate=.001, batch_size=64, weight_decay=.0001, timeout=600):
        features, sequence, outputs = self.dimensions
        if len(x) != len(targets):
            raise ValueError('Input/target sample count mismatch')
        xb, tb = self._buffer(x, features*sequence), self._buffer(targets, outputs)
        self.library.check(self.api.DN_SetMiniBatchSize(self.handle, batch_size))
        self.library.check(self.api.DN_SetGradClip(self.handle, 5.0))
        self.library.check(self.api.DN_LoadBatch(self.handle, xb, tb, len(x), features*sequence, outputs, 0))
        self.library.check(self.api.DN_TrainAsync(self.handle, epochs, 0, learning_rate, weight_decay))
        deadline = time.monotonic() + timeout
        try:
            while self.api.DN_GetTrainingStatus(self.handle) == 1:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Training timeout; cancellation requested')
                time.sleep(.02)
        except BaseException:
            self.api.DN_StopTraining(self.handle)
            raise
        if self.api.DN_GetTrainingStatus(self.handle) != 2:
            raise RuntimeError(self.library.error())
        mse, completed = C.c_double(), C.c_int()
        self.api.DN_GetTrainingResult(self.handle, C.byref(mse), C.byref(completed))
        return {'mse': mse.value, 'epochs': completed.value}

    def predict(self, rows):
        features, sequence, outputs = self.dimensions
        x = self._buffer(rows, features*sequence)
        y = (C.c_double * (len(rows)*outputs))()
        self.library.check(self.api.DN_PredictBatch(self.handle, x, len(rows), features*sequence, 0, y))
        return [list(y[i*outputs:(i+1)*outputs]) for i in range(len(rows))]

    def save(self, path):
        size = self.api.DN_SaveState(self.handle)
        if size <= 0:
            raise RuntimeError(self.library.error())
        data = C.create_string_buffer(size)
        self.library.check(self.api.DN_GetState(self.handle, data, size))
        Path(path).write_bytes(data.value)

    def load(self, path):
        data = Path(path).read_bytes()
        if b'\0' in data:
            raise ValueError('Unexpected embedded NUL in checkpoint')
        self.library.check(self.api.DN_LoadState(self.handle, data))
        return self
