"""Run a Neural Amp Modeler capture (.nam) over a guitar DI signal, offline (no audio device).

NAM needs pydantic 2, which clashes with other tools (gdsfactory wants pydantic<2), so it lives in its
own environment nn/venv_nam (once):

    python3 -m venv nn/venv_nam
    nn/venv_nam/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
    nn/venv_nam/bin/pip install neural-amp-modeler

render_metal.py then calls this file with that environment's python (worker mode) automatically.

Used by render_metal.py (--nam). Both the current .nam format and the older one (v0.5, WaveNet with
head_size/head_bias) load. The signal is resampled to the capture's sample rate and back, and run in
chunks so a whole song fits in memory.
"""
import json
import sys
import types
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly


def _import_nam():
    try:
        import tkinter  # noqa: F401  (nam imports it for its training GUI)
    except Exception:
        sys.modules['tkinter'] = types.ModuleType('tkinter')
    try:
        import torch
        from nam.models import init_from_nam
    except ImportError:
        sys.exit('NAM is not installed:  pip install --break-system-packages neural-amp-modeler')
    return torch, init_from_nam


def _legacy(cfg):
    """Old WaveNet layer configs (head_size / head_bias) -> the 'head' object newer nam versions want."""
    cfg = json.loads(json.dumps(cfg))
    for layer in cfg.get('config', {}).get('layers', []):
        if 'head' not in layer and 'head_size' in layer:
            layer['head'] = {'out_channels': layer.pop('head_size'), 'kernel_size': 1,
                             'bias': layer.pop('head_bias', False)}
    return cfg


@lru_cache(maxsize=8)
def load(path):
    torch, init_from_nam = _import_nam()
    cfg = json.load(open(Path(path).expanduser()))
    if cfg.get('architecture') == 'SlimmableContainer':
        # NAM Architecture 2 files hold several sizes of the same model ("A2-Full", "A2-Lite"):
        # use the largest (most accurate) one
        subs = cfg['config']['submodels']
        inner = max(subs, key=lambda s: s.get('max_value', 0))['model']
        inner.setdefault('sample_rate', cfg.get('sample_rate'))
        cfg = inner
    try:
        model = init_from_nam(cfg)
    except (KeyError, TypeError, RuntimeError):
        model = init_from_nam(_legacy(cfg))
    model.eval()
    sr = int(cfg.get('sample_rate') or 48000)
    model._is_lstm = cfg.get('architecture') == 'LSTM'          # keeps state: no chunking
    return model, sr


VENV_PY = Path(__file__).resolve().parent / 'venv_nam' / 'bin' / 'python'


def _have_nam():
    try:
        import pydantic
        if int(pydantic.VERSION.split('.')[0]) < 2:
            return False
        try:
            import tkinter  # noqa: F401
        except Exception:
            sys.modules['tkinter'] = types.ModuleType('tkinter')
        import nam  # noqa: F401
        return True
    except Exception:
        return False


def run(x, sr, path, in_db=-10.0, chunk_s=8.0):
    """Same as _run, done in nn/venv_nam when NAM cannot be imported here."""
    if _have_nam():
        return _run(x, sr, path, in_db, chunk_s)
    if not VENV_PY.exists():
        sys.exit('NAM environment missing. Set it up once (see the top of nn/nam_amp.py):\n'
                 '  python3 -m venv nn/venv_nam\n'
                 '  nn/venv_nam/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu\n'
                 '  nn/venv_nam/bin/pip install neural-amp-modeler')
    import subprocess
    import tempfile
    d = Path(tempfile.mkdtemp())
    np.save(d / 'in.npy', np.asarray(x, dtype=np.float64))
    subprocess.run([str(VENV_PY), __file__, '--worker', str(d / 'in.npy'), str(d / 'out.npy'), str(path),
                    str(in_db), str(sr), str(chunk_s)], check=True)
    return np.load(d / 'out.npy')


def _run(x, sr, path, in_db=-10.0, chunk_s=8.0):
    """x: mono float signal at sr. The DI is set to peak at in_db dBFS before the capture
    (the level a guitar DI usually sits at; higher = more gain into the amp)."""
    torch, _ = _import_nam()
    model, msr = load(str(path))
    x = np.asarray(x, dtype=np.float64)
    x = x / (np.abs(x).max() + 1e-12) * 10 ** (in_db / 20)
    if msr != sr:
        g = gcd(msr, sr)
        x = resample_poly(x, msr // g, sr // g)
    rf = int(getattr(model, 'receptive_field', 1))
    n = len(x) + 1 if getattr(model, '_is_lstm', False) else int(chunk_s * msr)
    out = []
    with torch.no_grad():
        for s in range(0, len(x), n):
            if s == 0:
                seg = torch.tensor(x[:n], dtype=torch.float32)
                y = model(seg, pad_start=True)
            else:
                seg = torch.tensor(x[max(0, s - (rf - 1)):s + n], dtype=torch.float32)
                y = model(seg, pad_start=(s - (rf - 1) < 0))
                y = y[-min(n, len(x) - s):]
            out.append(y.numpy().astype(np.float64))
    y = np.concatenate(out)[:len(x)]
    if msr != sr:
        g = gcd(msr, sr)
        y = resample_poly(y, sr // g, msr // g)
    return y


if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--worker':
    _, _, src, dst, model_path, in_db, sr, chunk_s = sys.argv
    np.save(dst, _run(np.load(src), int(sr), model_path, float(in_db), float(chunk_s)))
elif __name__ == '__main__':
    # quick check: python3 nn/nam_amp.py some.nam
    for p in sys.argv[1:]:
        m, msr = load(p)
        y = _run(np.sin(np.arange(48000) / 48000 * 2 * np.pi * 82.4) * 0.5, 48000, p)
        print(f'{Path(p).name}: ok, {msr} Hz, receptive field {getattr(m, "receptive_field", "?")}, '
              f'out peak {np.abs(y).max():.2f}')
