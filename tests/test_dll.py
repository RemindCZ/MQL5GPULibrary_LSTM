"""GPU integration regressions. Run in a separate process; requires Windows + CUDA.

python tests/test_dll.py build/Release/MQL5GPULibrary_LSTM.dll [CUDA bin directory]
"""
import ctypes as C
import math
import os
from pathlib import Path
import sys
import time
import unittest

DLL_PATH = Path(sys.argv[1]).resolve()
CUDA_BIN = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(os.environ['CUDA_PATH']) / 'bin'
DLL_DIRS = [os.add_dll_directory(str(p)) for p in (DLL_PATH.parent, CUDA_BIN, CUDA_BIN / 'x64') if p.is_dir()]
api = C.WinDLL(str(DLL_PATH))
I, D, P = C.c_int, C.c_double, C.POINTER


def bind(name, result, *args):
    fn = getattr(api, name)
    fn.restype, fn.argtypes = result, args
    return fn


bind('DN_Create', I)
bind('DN_Free', None, I)
for name in ('DN_SetSequenceLength', 'DN_SetMiniBatchSize', 'DN_SetOutputDim'):
    bind(name, I, I, I)
bind('DN_SetGradClip', I, I, D)
bind('DN_AddLayerEx', I, I, I, I, I, I, D)
bind('DN_LoadBatch', I, I, P(D), P(D), I, I, I, I)
bind('DN_PredictBatch', I, I, P(D), I, I, I, P(D))
for name in ('DN_Train', 'DN_TrainAsync'):
    bind(name, I, I, I, D, D, D)
bind('DN_GetTrainingStatus', I, I)
bind('DN_GetTrainingResult', None, I, P(D), P(I))
bind('DN_StopTraining', None, I)
bind('DN_SaveState', I, I)
bind('DN_GetState', I, I, P(C.c_char), I)
bind('DN_LoadState', I, I, C.c_char_p)
bind('DN_GetError', None, P(C.c_wchar), I)
bind('DN_SnapshotWeights', I, I)
bind('DN_RestoreWeights', I, I)
bind('DN_GetProgressLR', D, I)
bind('DN_GetGradNorm', D, I)


def scalar_checkpoint(kind, parameters):
    """Known one-unit model. Disk types: 0=RNN, 1=GRU, 2=LSTM."""
    gates = (1, 2, 4)[kind]
    n = 3 * gates
    text = f'SEQMODEL_V3 SEQ_LEN 3 LAYERS 1 {kind} 1 1 0 {2*gates} {gates} '
    text += ' '.join(map(str, parameters[:n])) + ' '
    if kind == 1:
        text += '1 1 1 1 ' + ' '.join(map(str, parameters[n:n+4])) + ' '
        n += 4
    return (text + 'OUTPUT 1 1 ' + ' '.join(map(str, parameters[n:]))).encode()


def scalar_reference(kind, parameters, samples, sequence=False):
    """Independent CPU recurrence with explicit scalar equations, no CUDA or numpy."""
    gates = (1, 2, 4)[kind]
    weights, biases = parameters[:2*gates], parameters[2*gates:3*gates]
    sigmoid = lambda v: 1 / (1 + math.exp(-v))
    result = []
    for sample in samples:
        h = cell = 0.0
        timeline = []
        for x in sample:
            raw = [weights[2*j]*h + weights[2*j+1]*x + biases[j] for j in range(gates)]
            if kind == 0:
                h = math.tanh(raw[0])
            elif kind == 1:
                wx, wh, bx, bh = parameters[6:10]
                z, r = map(sigmoid, raw)
                candidate = math.tanh(wx*x + bx + r*(wh*h + bh))
                h = (1-z)*h + z*candidate
            else:
                f, i, g, o = raw
                cell = sigmoid(f)*cell + sigmoid(i)*math.tanh(g)
                h = sigmoid(o)*math.tanh(cell)
            timeline.append(parameters[-2]*h + parameters[-1])
        result.append(timeline if sequence else timeline[-1])
    return result


def error():
    buf = C.create_unicode_buffer(1024)
    api.DN_GetError(buf, len(buf))
    return buf.value


class GPUApiTests(unittest.TestCase):
    def setUp(self):
        self.handles = []

    def tearDown(self):
        for h in self.handles:
            api.DN_Free(h)

    def ok(self, value):
        self.assertEqual(value, 1, error())

    def model(self, kind=0, stacked=False):
        h = api.DN_Create()
        self.assertGreater(h, 0, error())
        self.handles.append(h)
        self.ok(api.DN_SetSequenceLength(h, 3))
        self.ok(api.DN_SetMiniBatchSize(h, 4))
        self.ok(api.DN_AddLayerEx(h, 2, 5, kind, 0, 0))
        if stacked:
            self.ok(api.DN_AddLayerEx(h, 0, 3, 0, 0, 0))
        self.ok(api.DN_SetOutputDim(h, 2))
        return h

    def data(self, h, count=9):
        x = (D * (count * 6))(*(0.2 * math.sin(i * 0.3) for i in range(count * 6)))
        t = (D * (count * 2))(*(0.1 * math.cos(i * 0.2) for i in range(count * 2)))
        self.ok(api.DN_LoadBatch(h, x, t, count, 6, 2, 0))
        return x, t

    def predict(self, h, x, count=9):
        y = (D * (count * 2))()
        self.ok(api.DN_PredictBatch(h, x, count, 6, 0, y))
        self.assertTrue(all(math.isfinite(v) for v in y))
        return list(y)

    def save(self, h):
        size = api.DN_SaveState(h)
        self.assertGreater(size, 0, error())
        data = C.create_string_buffer(size)
        self.ok(api.DN_GetState(h, data, size))
        return data.value

    def test_roundtrip_all_layer_types_and_stacked(self):
        for kind, stacked in [(0, False), (1, False), (2, False), (0, True)]:
            with self.subTest(kind=kind, stacked=stacked):
                h = self.model(kind, stacked)
                x, _ = self.data(h)
                before = self.predict(h, x)
                state = self.save(h)
                self.ok(api.DN_LoadState(h, state))
                self.assertEqual(before, self.predict(h, x))
                self.assertEqual(state, self.save(h))
                self.assertEqual(api.DN_Train(h, 1, 0, 0.001, 0), 0)  # data invalidated

    def test_invalid_checkpoint_preserves_model(self):
        h = self.model()
        x, _ = self.data(h)
        before, state = self.predict(h, x), self.save(h)
        lines = state.decode().splitlines()
        oversized = lines.copy()
        oversized[4] = '2147483647 20'
        disconnected = lines.copy()
        out = next(i for i, line in enumerate(lines) if line.startswith('OUTPUT'))
        disconnected[out] = 'OUTPUT 999 2'
        bad = [b'', state[:len(state)//2], state + b' junk',
               '\n'.join(oversized).encode(), '\n'.join(disconnected).encode(),
               state.replace(b'SEQ_LEN 3', b'SEQ_LEN 0')]
        for payload in bad:
            self.assertEqual(api.DN_LoadState(h, payload), 0)
            self.assertEqual(before, self.predict(h, x))
            self.assertEqual(state, self.save(h))

    def test_invalid_inputs_preserve_weights(self):
        h = self.model()
        x, t = self.data(h)
        state = self.save(h)
        y = (D * 18)()
        for n, dim, layout in [(-1, 6, 0), (0, 6, 0), (9, 5, 0), (9, 9, 0), (9, 6, 1), (2147483647, 6, 0)]:
            self.assertEqual(api.DN_PredictBatch(h, x, n, dim, layout, y), 0)
        for value in [float('nan'), float('inf'), 1e100]:
            x[0] = value
            self.assertEqual(api.DN_LoadBatch(h, x, t, 9, 6, 2, 0), 0)
            self.assertEqual(api.DN_PredictBatch(h, x, 9, 6, 0, y), 0)
        for epochs, target, lr, wd in [(0, 0, .01, 0), (1, -1, .01, 0), (1, 0, float('nan'), 0), (1, 0, .01, -1)]:
            self.assertEqual(api.DN_TrainAsync(h, epochs, target, lr, wd), 0)
        self.assertEqual(api.DN_SetSequenceLength(h, 0), 0)
        self.assertEqual(api.DN_SetMiniBatchSize(h, -1), 0)
        self.assertEqual(api.DN_AddLayerEx(h, 5, 3, 0, 1, 0), 0)
        self.assertEqual(api.DN_GetState(h, None, len(state)+1), 0)
        self.assertEqual(api.DN_GetState(h, C.create_string_buffer(1), 1), 0)
        self.assertEqual(state, self.save(h))

    def test_training_loss_matches_returned_weights(self):
        for kind in (0, 1, 2):
            with self.subTest(kind=kind):
                h = self.model(kind)
                x, target = self.data(h)
                self.ok(api.DN_Train(h, 3, 0, .005, 0))
                self.assertEqual(api.DN_GetTrainingStatus(h), 2)
                mse, epochs = D(), I()
                api.DN_GetTrainingResult(h, C.byref(mse), C.byref(epochs))
                actual = sum((a-b)**2 for a, b in zip(self.predict(h, x), target)) / len(target)
                self.assertAlmostEqual(mse.value, actual, delta=max(1e-7, actual * 1e-5))
                self.assertEqual(epochs.value, 3)

    def test_snapshot_restore(self):
        h = self.model(1)
        x, _ = self.data(h)
        before = self.predict(h, x)
        self.ok(api.DN_SnapshotWeights(h))
        self.ok(api.DN_Train(h, 2, 0, .005, 0))
        self.ok(api.DN_RestoreWeights(h))
        self.assertEqual(before, self.predict(h, x))

    def test_async_busy_stop_and_other_model_remain_responsive(self):
        h, other = self.model(), self.model()
        self.data(h)
        x, _ = self.data(other)
        self.ok(api.DN_TrainAsync(h, 100000, 0, .001, 0))
        started = time.monotonic()
        self.assertEqual(api.DN_SetOutputDim(h, 2), 0)
        self.predict(other, x)
        self.assertLess(time.monotonic() - started, 5)
        api.DN_StopTraining(h)
        deadline = time.monotonic() + 20
        while api.DN_GetTrainingStatus(h) == 1 and time.monotonic() < deadline:
            mse, epochs = D(), I()
            api.DN_GetTrainingResult(h, C.byref(mse), C.byref(epochs))
            time.sleep(.005)
        self.assertEqual(api.DN_GetTrainingStatus(h), 2, error())
        self.ok(api.DN_Train(h, 1, 0, .001, 0))

    def test_free_during_training_and_invalid_handle(self):
        h = self.model()
        self.data(h)
        self.ok(api.DN_TrainAsync(h, 100000, 0, .001, 0))
        api.DN_Free(h)
        self.assertEqual(api.DN_GetTrainingStatus(h), -1)
        self.assertEqual(api.DN_SetOutputDim(h, 2), 0)
        api.DN_Free(h)

    def test_forward_and_backward_against_cpu_equations(self):
        samples = [[.1, -.2, .3], [-.3, .2, .15]]
        targets = [.2, -.1]
        for kind in (0, 1, 2):
            with self.subTest(kind=kind):
                count = 3*(1, 2, 4)[kind] + (4 if kind == 1 else 0) + 2
                params = [.3 * math.sin(i + 1) for i in range(count)]
                h = self.model()
                self.ok(api.DN_LoadState(h, scalar_checkpoint(kind, params)))
                x = (D * 6)(*(v for row in samples for v in row))
                t, y = (D * 2)(*targets), (D * 2)()
                self.ok(api.DN_PredictBatch(h, x, 2, 3, 0, y))
                for actual, expected in zip(y, scalar_reference(kind, params, samples)):
                    self.assertAlmostEqual(actual, expected, delta=1e-6)
                def loss(p):
                    return sum((a-b)**2 for a, b in zip(scalar_reference(kind, p, samples), targets))/2
                gradients = []
                for i in range(count):
                    plus, minus = params.copy(), params.copy()
                    plus[i] += 1e-5
                    minus[i] -= 1e-5
                    gradients.append((loss(plus)-loss(minus))/2e-5)
                expected_norm = math.sqrt(sum(g*g for g in gradients))
                self.ok(api.DN_SetGradClip(h, 0))
                self.ok(api.DN_LoadBatch(h, x, t, 2, 3, 1, 0))
                self.ok(api.DN_Train(h, 1, 0, .0001, 0))
                self.assertAlmostEqual(api.DN_GetGradNorm(h), expected_norm, delta=2e-5)

    def test_stacked_gru_propagates_candidate_input_gradient(self):
        samples, targets = [[.4, -.2, .6], [-.7, .5, .15]], [.3, -.2]
        params = [.6*math.sin(i+1) for i in range(15)]
        lower = scalar_checkpoint(0, params[:3] + [1, 0]).decode().split('LAYERS 1 ')[1].split('OUTPUT')[0]
        upper = scalar_checkpoint(1, params[3:]).decode().split('LAYERS 1 ')[1]
        checkpoint = ('SEQMODEL_V3 SEQ_LEN 3 LAYERS 2 ' + lower + upper).encode()
        h = self.model()
        self.ok(api.DN_LoadState(h, checkpoint))
        def predict(p):
            features = scalar_reference(0, p[:3] + [1, 0], samples, sequence=True)
            return scalar_reference(1, p[3:], features)
        def loss(p):
            return sum((a-b)**2 for a,b in zip(predict(p), targets))/2
        x = (D * 6)(*(v for row in samples for v in row))
        t, y = (D * 2)(*targets), (D * 2)()
        self.ok(api.DN_PredictBatch(h, x, 2, 3, 0, y))
        for actual, expected in zip(y, predict(params)):
            self.assertAlmostEqual(actual, expected, delta=1e-6)
        gradients = []
        for i in range(len(params)):
            plus, minus = params.copy(), params.copy()
            plus[i] += 1e-5
            minus[i] -= 1e-5
            gradients.append((loss(plus)-loss(minus))/2e-5)
        self.ok(api.DN_SetGradClip(h, 0))
        self.ok(api.DN_LoadBatch(h, x, t, 2, 3, 1, 0))
        self.ok(api.DN_Train(h, 1, 0, .0001, 0))
        self.assertAlmostEqual(api.DN_GetGradNorm(h), math.sqrt(sum(g*g for g in gradients)), delta=2e-5)

    def test_dropout_and_repeated_training(self):
        for kind in (0, 1, 2):
            with self.subTest(kind=kind):
                h = self.model(kind)
                self.ok(api.DN_AddLayerEx(h, 5, 3, kind, 0, .3))
                self.ok(api.DN_SetOutputDim(h, 2))
                x, _ = self.data(h)
                self.ok(api.DN_Train(h, 2, 0, .001, 0))
                self.assertEqual(self.predict(h, x), self.predict(h, x))
                self.data(h)
                self.ok(api.DN_Train(h, 1, 0, .001, 0))
                self.assertTrue(math.isfinite(api.DN_GetGradNorm(h)))

if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]], verbosity=2)
