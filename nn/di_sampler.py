"""Render the guitar tracks of a .gp5 as a real guitar DI from the free METAL-GTX sample library
(Unreal Instruments, DI recordings of a metal guitar), instead of the GM SoundFont guitar.

Used by render_metal.py --di DIR  (DIR = the extracted UI_METAL-GTX folder) and --bass-di DIR (Standard Bass of
the same author: folders with 'sus' / 'mute' in their names are used). Standalone check:
    python3 nn/di_sampler.py works/song.gp5 /tmp/mgtx/UI_METAL-GTX --out /tmp/di_check.wav

What it does per note
  - palm-muted notes use Samples/Mute_Down, other notes Samples/Sus_Down (dead notes: a short mute)
  - alternate picking: notes on an off 16th use the up-stroke samples (Sus_Up / Mute_Up) when they are there
  - the take (1..3) is chosen round-robin with a seeded shuffle, so repeated notes are not the same sample
    and guitar 2 (another seed) really is a second take, not a copy
  - missing pitches are taken from the nearest sample and resampled
  - one voice per string: a new note on a string cuts the old one (short fade), like a real guitar
  - when a held (not muted) note stops and the string stays silent, a string-stop sample (Release1) is added
  - chords are strummed down low -> high over a few ms
  - ties extend the note on that string
The .sfz files are not needed (only the samples, file names give pitch and take).
"""
import argparse
import random
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

SR = 44100
NAMES = {'c': 0, 'c#': 1, 'd': 2, 'd#': 3, 'e': 4, 'f': 5, 'f#': 6, 'g': 7, 'g#': 8, 'a': 9, 'a#': 10, 'b': 11}
RX = re.compile(r'^([a-g]#?)(-?\d)_.*\.(flac|wav)$', re.I)
_CACHE = {}


def midi_of(name, octv):
    return NAMES[name.lower()] + 12 * (int(octv) + 1)          # sfz convention c4 = 60


def _folder(s, exact, words):
    """Sample folder for one articulation: an exact name, else the first folder whose name has all words."""
    if (s / exact).is_dir():
        return s / exact
    for d in sorted(p for p in s.iterdir() if p.is_dir()):
        n = d.name.lower()
        if all(w in n for w in words) and 'up' not in n.replace('pickup', ''):
            return d
    return None


def index(root, verbose=False):
    """{kind: {midi: [paths...]}} for kind in sus, sus_up, mute, mute_up, rel (up / rel may be empty).
    Folder names of METAL-GTX (Sus_Down, Mute_Down, Sus_Up, Mute_Up, Release1); for other libraries of the
    same author (Standard Bass ...) folders with 'sus' / 'mute' in the name are used."""
    root = Path(root).expanduser()
    s = root / 'Samples' if (root / 'Samples').exists() else root
    want = {'sus': ('Sus_Down', ['sus']), 'mute': ('Mute_Down', ['mute']), 'rel': ('Release1', ['release1'])}
    out, used = {}, {}
    for kind, (exact, words) in want.items():
        d = _folder(s, exact, words)
        used[kind] = d
        out[kind] = {}
        for p in sorted(d.rglob('*')) if d else []:
            m = RX.match(p.name)
            if m and p.stat().st_size > 0:
                out[kind].setdefault(midi_of(m.group(1), m.group(2)), []).append(p)
    for kind, exact in (('sus_up', 'Sus_Up'), ('mute_up', 'Mute_Up')):
        d = s / exact
        out[kind] = {}
        for p in sorted(d.glob('*')) if d.is_dir() else []:
            m = RX.match(p.name)
            if m and p.stat().st_size > 0:
                out[kind].setdefault(midi_of(m.group(1), m.group(2)), []).append(p)
    if not out['sus']:
        raise SystemExit(f'no sustain samples under {s} (folders: {[p.name for p in s.iterdir()]})')
    if not out['mute']:
        out['mute'] = out['sus']                    # library without mutes: short sustains
    if verbose:
        print('  samples: ' + ', '.join(f"{k}={sum(map(len, v.values()))}" for k, v in out.items()))
    return out


def load(p):
    p = str(p)
    if p in _CACHE:
        return _CACHE[p]
    x = None
    try:
        import soundfile as sf
        x, sr = sf.read(p, dtype='float32', always_2d=True)
        x = x.mean(axis=1)
        if sr != SR:
            x = np.interp(np.arange(0, len(x), sr / SR), np.arange(len(x)), x).astype(np.float32)
    except Exception:
        if not shutil.which('ffmpeg'):
            raise SystemExit('need ffmpeg (or python soundfile) to read the FLAC samples')
        r = subprocess.run(['ffmpeg', '-loglevel', 'error', '-i', p, '-ac', '1', '-ar', str(SR), '-f', 'f32le', '-'],
                           stdout=subprocess.PIPE, check=True)
        x = np.frombuffer(r.stdout, dtype=np.float32).copy()
    _CACHE[p] = x
    return x


def pick(table, midi, rng, last):
    """Nearest sampled pitch, a take that differs from the last one used for that pitch, resample ratio."""
    keys = np.array(sorted(table))
    k = int(keys[np.argmin(np.abs(keys - midi))])
    takes = table[k]
    choices = [t for t in takes if t != last.get(k)] or takes
    t = rng.choice(choices)
    last[k] = t
    return t, 2 ** ((midi - k) / 12)


def shifted(x, ratio):
    if abs(ratio - 1) < 1e-4:
        return x
    n = int(len(x) / ratio)
    return np.interp(np.arange(n) * ratio, np.arange(len(x)), x).astype(np.float32)


def notes_of(track, tempo):
    """[(t_sec, dur_sec, string, midi, kind, vel, up)] with ties merged; kind = sus | mute | dead;
    up = the note sits on an off 16th (alternate picking: up stroke)."""
    import guitarpro as gp
    spt = 60.0 / tempo / 960.0                                 # seconds per tick
    tun = {s.number: s.value for s in track.strings}
    out, open_ = [], {}
    for m in track.measures:
        for v in m.voices:
            t = m.header.start
            for b in v.beats:
                d = b.duration.time
                for n in b.notes:
                    if n.type == gp.NoteType.tie and n.string in open_:
                        out[open_[n.string]][1] += d * spt
                        continue
                    kind = 'dead' if n.type == gp.NoteType.dead else ('mute' if n.effect.palmMute else 'sus')
                    up = ((t - m.header.start) % 480) == 240           # off-16th = up stroke
                    out.append([(t - 960) * spt, d * spt, n.string, tun[n.string] + n.value, kind, n.velocity, up])
                    open_[n.string] = len(out) - 1
                t += d
    return sorted(out, key=lambda e: (e[0], -e[2]))


def render_track(gp5, track_name, root, seed=0, jitter_ms=0.0, strum_ms=2.0, release=True):
    """Mono DI (float32, SR) of one guitar track of the .gp5."""
    import guitarpro as gp
    song = gp.parse(str(gp5))
    tr = next(t for t in song.tracks if t.name == track_name)
    idx = index(root, verbose=seed == 0)
    nstr = len(tr.strings)
    rng = random.Random(seed)
    notes = notes_of(tr, song.tempo)
    total = max((e[0] + e[1] for e in notes), default=0) + 3.0
    y = np.zeros(int(total * SR) + 1, dtype=np.float32)
    last = {k: {} for k in ('sus', 'mute', 'sus_up', 'mute_up', 'rel')}
    by_string = {}
    for e in notes:
        by_string.setdefault(e[2], []).append(e)
    fade = int(0.008 * SR)
    for s, lst in by_string.items():
        for i, (t, d, _, midi, kind, vel, up) in enumerate(lst):
            nxt = lst[i + 1][0] if i + 1 < len(lst) else None
            # chord strum: lower strings (higher number) first
            t0 = t + (nstr - s) * strum_ms / 1000.0 + (rng.gauss(0, jitter_ms) / 1000.0 if jitter_ms else 0.0)
            t0 = max(t0, 0.0)
            base = 'mute' if kind in ('mute', 'dead') else 'sus'
            key = base + '_up' if up and idx[base + '_up'] else base
            path, ratio = pick(idx[key], midi, rng, last[key])
            x = shifted(load(path), ratio)
            if kind == 'dead':
                end = t0 + 0.05
            elif kind == 'mute':
                end = t0 + max(d, 0.09) + 0.02
            else:
                end = t0 + d
            if nxt is not None:
                end = min(end, nxt + (nstr - s) * strum_ms / 1000.0)
            n = max(int((end - t0) * SR), fade + 1)
            x = x[:n].copy()
            if len(x) > fade:
                x[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            g = (max(vel, 30) / 127.0) ** 1.2 * 10 ** (rng.gauss(0, 0.8) / 20)
            a = int(t0 * SR)
            y[a:a + len(x)] += g * x[:len(y) - a]
            # string stop noise after a held note when the string then rests
            if release and kind == 'sus' and idx['rel'] and (nxt is None or nxt - (t + d) > 0.06):
                rp, rr = pick(idx['rel'], midi, rng, last['rel'])
                r = shifted(load(rp), rr)[:int(0.4 * SR)]
                b = int(end * SR) - fade
                if 0 <= b < len(y):
                    y[b:b + len(r)] += 0.6 * g * r[:len(y) - b]
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gp5')
    ap.add_argument('root', help='extracted UI_METAL-GTX folder')
    ap.add_argument('--track', default='Guitar')
    ap.add_argument('--out', default='/tmp/di_check.wav')
    a = ap.parse_args()
    from scipy.io import wavfile
    y = render_track(a.gp5, a.track, a.root)
    y = y / (np.abs(y).max() + 1e-9) * 0.5
    wavfile.write(a.out, SR, (y * 32767).astype(np.int16))
    print(f'wrote {a.out} ({len(y) / SR:.1f} s)')


if __name__ == '__main__':
    main()
