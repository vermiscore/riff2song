"""Render a generated song to a metal-sounding mp3 (better than plain General MIDI).

  python3 nn/render_metal.py outputs/<run>/cand_01.gp5          -> cand_01_metal.mp3 next to it
  python3 nn/render_metal.py song.mid --out song.mp3

How: each part is rendered separately with fluidsynth, the guitars as a *clean* electric guitar
(a "DI" signal), then run through a simple amp simulation written here (tight low cut, mid boost,
high-gain clipping, 4x12 cabinet-like EQ). The two guitars are panned hard left/right with a few
milliseconds of offset (double tracking); bass gets a little drive; drums use the GM kit.
Needs: fluidsynth, a GM SoundFont, ffmpeg, numpy, scipy, mido.

With lv2file installed (sudo apt install lv2file) the guitars go through the real Guitarix
plugins instead (GxTubeScreamer -> GxMetalHead), processed offline from files, so your audio
setup (PipeWire/JACK/ALSA) is not touched. Knobs: nn/amp_settings.json (created on first run).
"""
import argparse
import warnings
warnings.filterwarnings("ignore", message="Chunk")
import subprocess
import sys
import tempfile
from pathlib import Path

import mido
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfilt

HERE = Path(__file__).resolve().parent
SR = 44100
DI = [None]          # METAL-GTX folder: guitars from real DI samples (di_sampler.py)
BASS_DI = [None]     # Standard Bass folder (di_sampler.py)
DRUMKIT = [None]     # DrumGizmo kit folder (drum_sampler.py)
SF2S = ['/usr/share/sounds/sf2/FluidR3_GM.sf2', '/usr/share/sounds/sf2/default-GM.sf2']
CLEAN_GTR, BASS = 27, 33          # GM: electric guitar (clean), electric bass (finger)


def to_midi_from_gp5(gp5, mid_path):
    import guitarpro as gp
    sys.path.insert(0, str(HERE))
    import postprocess as P
    P.to_midi(gp.parse(str(gp5)), str(mid_path))


def split_tracks(mid_path, tmp):
    mid = mido.MidiFile(str(mid_path))
    meta = mid.tracks[0]
    parts = {}
    for tr in mid.tracks[1:]:
        name = next((m.name for m in tr if m.type == 'track_name'), 'x')
        out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
        out.tracks.append(meta.copy())
        t2 = mido.MidiTrack()
        for m in tr:
            if m.type == 'program_change':
                prog = CLEAN_GTR if name.startswith('Guitar') else (BASS if name == 'Bass' else m.program)
                m = m.copy(program=prog)
            if name.startswith('Guitar') and m.type in ('note_on', 'note_off'):
                m = m.copy(channel=0)
            t2.append(m)
        out.tracks.append(t2)
        p = Path(tmp) / f'{name.replace(" ", "_")}.mid'
        out.save(str(p))
        parts[name] = p
    return parts


def synth(mid, sf2, tmp):
    wav = Path(tmp) / (mid.stem + '.wav')
    subprocess.run(['fluidsynth', '-ni', '-g', '1.0', '-R', '0', '-C', '0', '-F', str(wav), '-r', str(SR), sf2, str(mid)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    sr, x = wavfile.read(str(wav))
    x = x.astype(np.float32) / 32768.0
    return x.mean(axis=1) if x.ndim == 2 else x, x


def peak(x, f, gain_db, q=1.0):
    """RBJ peaking EQ as one biquad section."""
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f / SR; al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]; a = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    sos = np.array([[b[0] / a[0], b[1] / a[0], b[2] / a[0], 1, a[1] / a[0], a[2] / a[0]]])
    return sosfilt(sos, x)


def hp(x, f, o=2):
    return sosfilt(butter(o, f, 'highpass', fs=SR, output='sos'), x)


def lp(x, f, o=2):
    return sosfilt(butter(o, f, 'lowpass', fs=SR, output='sos'), x)


def amp(x, gain=60.0, seed=0):
    """High-gain amp sim: tight low cut + tube-screamer-like mid push, two clipping stages, cab EQ."""
    x = x / (np.abs(x).max() + 1e-9)
    x = hp(x, 120, 2)                         # tighten the low end before the clipping
    x = peak(x, 750, 9, 0.8)                  # mid hump (the "tubescreamer" in front)
    x = np.tanh(gain * x + 0.15) - np.tanh(0.15)        # asymmetric first stage
    x = hp(x, 60, 1)
    x = np.tanh(3.0 * x)                      # second stage
    # 4x12 cabinet-ish: low resonance, scooped low mids, presence, steep fizz cut
    x = peak(x, 110, 4, 0.9)
    x = peak(x, 400, -5, 1.0)
    x = peak(x, 2400, 4, 1.2)
    x = lp(x, 5200, 4)
    x = hp(x, 70, 2)
    return x / (np.abs(x).max() + 1e-9)


GX_TS9 = 'http://guitarix.sourceforge.net/plugins/gxts9#ts9sim'
GX_AMP = 'http://guitarix.sourceforge.net/plugins/gxmetal_head#metal_head'
AMP_SETTINGS = HERE / 'amp_settings.json'
DEFAULT_SETTINGS = {
    "_help": "Guitarix knobs used by render_metal.py --amp guitarix. Edit and re-run.",
    "ts9": {"use": 1, "fslider2_": 1.0, "fslider0_": 4.0, "fslider1_": 700.0,
            "_names": "fslider2_ = Drive (0-1), fslider0_ = Level (-20..4 dB), fslider1_ = Tone (100-1000)"},
    "metal_head": {"DRIVE": 0.8, "PREGAIN": 15.0, "GAIN1": 0.0, "TONE": 0.6, "HIGHGAIN": 1.0,
                  "_names": "DRIVE 0-1, PREGAIN/GAIN1 -20..20 dB, TONE 0-1, HIGHGAIN 0/1"},
    "guitar2_drive_offset": 0.0
}


def settings():
    import json
    if not AMP_SETTINGS.exists():
        AMP_SETTINGS.write_text(json.dumps(DEFAULT_SETTINGS, indent=1))
    return json.loads(AMP_SETTINGS.read_text())


CAB_IR = [None]


def cab(y, ir_path=None):
    """Speaker cabinet after the amp. With an IR (.wav of a miked cab) it is convolved; without one a rough
    4x12 curve is used (lows below 80 Hz and highs above ~5.5 kHz rolled off, a bump around 2.5 kHz)."""
    from scipy.signal import fftconvolve
    ir_path = ir_path or CAB_IR[0] or settings().get('cab_ir')
    if ir_path and Path(ir_path).expanduser().exists():
        sr, ir = wavfile.read(str(Path(ir_path).expanduser()))
        ir = ir.astype(np.float64)
        if ir.ndim == 2:
            ir = ir.mean(axis=1)
        if sr != SR:
            from scipy.signal import resample_poly
            ir = resample_poly(ir, SR, sr)
        ir = ir[:int(0.5 * SR)] / (np.abs(ir).max() + 1e-12)
        y = fftconvolve(y, ir)[:len(y)]
    else:
        y = hp(y, 80, 2)
        y = lp(lp(y, 5500, 2), 7000, 2)
        y = peak(y, 2500, 3.0, 1.0)
        y = peak(y, 120, 2.0, 0.8)
    return y / (np.abs(y).max() + 1e-9)


HUMANIZE = [8.0]
NAM = {'amp': None, 'amp2': None, 'pre': None, 'in_db': -10.0, 'gate': None, 'tight': 0.0}


def gate(x, thr_db=-45.0, hold_ms=15.0, release_ms=35.0, attack_ms=0.5):
    """Noise gate on the DI before the amp (like an NS-2 / ISP Decimator in front of the head): the signal is
    muted where its envelope stays thr_db below the DI's peak, so chugs stop dead between hits."""
    x = np.asarray(x, dtype=np.float64)
    env = np.abs(x)
    win = max(1, int(0.002 * SR))
    env = np.convolve(env, np.ones(win) / win, mode='same')            # 2 ms average
    thr = np.abs(x).max() * 10 ** (thr_db / 20)
    open_ = env > thr
    hold = int(hold_ms / 1000 * SR)
    if hold:                                                            # keep open hold_ms after the last hit
        idx = np.flatnonzero(open_)
        mark = np.zeros(len(x) + hold + 1, dtype=np.int32)
        np.add.at(mark, idx, 1)
        np.add.at(mark, idx + hold, -1)
        open_ = np.cumsum(mark)[:len(x)] > 0
    g = open_.astype(np.float64)
    a = np.exp(-1 / (attack_ms / 1000 * SR)); r = np.exp(-1 / (release_ms / 1000 * SR))
    out = np.empty_like(g)
    from scipy.signal import lfilter
    # one-pole smoothing: fast when opening, slow when closing (done in two passes)
    up = lfilter([1 - a], [1, -a], g)
    down = lfilter([1 - r], [1, -r], g)
    out = np.where(g >= up, np.maximum(up, down), down)
    out = np.clip(np.maximum(out, np.minimum(up, g)), 0, 1)
    return x * out



def post_eq(y):
    """Rhythm guitar EQ the usual way for metal (no compression): high-pass, a wide low-mid (mud) dip,
    low-pass for fizz. Knobs: "post_eq" in nn/amp_settings.json."""
    c = settings().get('post_eq', {})
    y = hp(hp(y, c.get('hpf', 100), 2), c.get('hpf', 100), 2)
    y = peak(y, c.get('mud_hz', 400), c.get('mud_db', -3.0), c.get('mud_q', 1.0))
    if c.get('harsh_db', 0):
        y = peak(y, c.get('harsh_hz', 3500), c['harsh_db'], 1.5)
    y = lp(lp(y, c.get('lpf', 10000), 2), c.get('lpf', 10000), 2)
    return y / (np.abs(y).max() + 1e-9)


def nam_amp(x, second=False):
    """DI -> (pedal capture) -> amp capture (.nam) -> cabinet IR (skipped for full-rig captures, whose
    file names start with FR) -> post EQ."""
    import nam_amp as N
    y = np.asarray(x, dtype=np.float64)
    if NAM['tight']:                       # low cut into the amp (the "tight" knob of metal boosts)
        y = hp(y, NAM['tight'], 2)
    if NAM['gate'] is not None:
        y = gate(y, NAM['gate'])
    if NAM['pre']:
        y = N.run(y, SR, NAM['pre'], NAM['in_db'])
        y = y / (np.abs(y).max() + 1e-9) * 0.5
        y = N.run(y, SR, (NAM['amp2'] if second and NAM['amp2'] else NAM['amp']), in_db=-6.0)
    else:
        y = N.run(y, SR, (NAM['amp2'] if second and NAM['amp2'] else NAM['amp']), NAM['in_db'])
    model = Path(NAM['amp2'] if second and NAM['amp2'] else NAM['amp']).name
    if not model.upper().startswith('FR'):
        y = cab(y)
    return post_eq(y)


def humanize(mid_path, ms=8.0, vel=12, seed=1):
    """Make a second guitar part sound like another take: note starts moved by up to +-ms, velocities
    varied, so hard-panned L/R guitars are not identical copies."""
    import random
    rng = random.Random(seed)
    mid = mido.MidiFile(str(mid_path))
    tpb = mid.ticks_per_beat
    tempo = next((m.tempo for t in mid.tracks for m in t if m.type == 'set_tempo'), 500000)
    tick_ms = tempo / 1000 / tpb
    for ti, tr in enumerate(mid.tracks):
        absmsgs, now = [], 0
        for m in tr:
            now += m.time
            absmsgs.append([now, m])
        shift = {}
        for e in absmsgs:
            m = e[1]
            if m.type == 'note_on' and m.velocity > 0:
                d = int(round(rng.uniform(-ms, ms) / tick_ms))
                shift[(m.channel, m.note)] = d
                e[0] = max(0, e[0] + d)
                e[1] = m.copy(velocity=max(1, min(127, m.velocity + rng.randint(-vel, vel))))
            elif m.type in ('note_off', 'note_on'):
                e[0] = max(0, e[0] + shift.get((m.channel, m.note), 0))
        absmsgs.sort(key=lambda e: e[0])
        new, prev = mido.MidiTrack(), 0
        for t, m in absmsgs:
            new.append(m.copy(time=t - prev)); prev = t
        mid.tracks[ti] = new
    mid.save(str(mid_path))


def gx_amp(x, tmp, tag, drive_offset=0.0):
    """Run the guitar through the real Guitarix plugins (GxTubeScreamer -> GxMetalHead) with lv2file.
    Offline file processing: no audio device, nothing in the sound settings is touched."""
    import shutil
    if not shutil.which('lv2file'):
        sys.exit('lv2file not found: sudo apt install lv2file   (or use --amp builtin)')
    cfg = settings()
    x = x / (np.abs(x).max() + 1e-9) * 0.5
    src = Path(tmp) / f'{tag}_di.wav'
    wavfile.write(str(src), SR, x.astype(np.float32))
    chain = []
    if cfg['ts9'].get('use', 1):
        chain.append((GX_TS9, {k: v for k, v in cfg['ts9'].items() if not k.startswith('_') and k != 'use'}))
    amp_p = {k: v for k, v in cfg.get('metal_head', cfg.get('metal_amp', {})).items() if not k.startswith('_')}
    amp_p['DRIVE'] = min(1.0, amp_p.get('DRIVE', 0.8) + drive_offset)
    chain.append((GX_AMP, amp_p))
    for i, (uri, params) in enumerate(chain):
        dst = Path(tmp) / f'{tag}_{i}.wav'
        cmd = ['lv2file', '-i', str(src), '-o', str(dst), '--ignore-clipping']
        for k, v in params.items():
            cmd += ['-p', f'{k}:{v}']
        subprocess.run(cmd + [uri], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        src = dst
    sr, y = wavfile.read(str(src))
    y = y.astype(np.float64)
    if y.ndim == 2:
        y = y.mean(axis=1)
    y = hp(y, 70, 2)
    return cab(y)


def render(src, out, sf2, amp_kind='builtin'):
    tmp = Path(tempfile.mkdtemp())
    src = Path(src)
    mid = src
    if src.suffix == '.gp5':
        mid = tmp / 'song.mid'
        to_midi_from_gp5(src, mid)
    parts = split_tracks(mid, tmp)
    if HUMANIZE[0]:
        for name, p in parts.items():
            if name.startswith('Guitar') and name != 'Guitar':
                humanize(p, HUMANIZE[0])
    sig = {}
    for name, p in parts.items():
        if DI[0] and name.startswith('Guitar') and src.suffix == '.gp5':
            import di_sampler as D
            i = sorted(k for k in parts if k.startswith('Guitar')).index(name)
            x = D.render_track(src, name, DI[0], seed=i, jitter_ms=HUMANIZE[0] / 2 if i else 0.0)
            sig[name] = (x, x)
            continue
        if BASS_DI[0] and name == 'Bass' and src.suffix == '.gp5':
            import di_sampler as D
            x = D.render_track(src, name, BASS_DI[0], seed=7, strum_ms=0.0, release=False)
            sig[name] = (x, x)
            continue
        if DRUMKIT[0] and name == 'Drums':
            import drum_sampler as DS
            st = DS.render(p, DRUMKIT[0])
            sig[name] = (st.mean(axis=1), st)
            continue
        mono, stereo = synth(p, sf2, tmp)
        sig[name] = (mono, stereo)
    n = max(len(m) for m, _ in sig.values())
    L, R = np.zeros(n), np.zeros(n)

    def add(x, gl, gr, delay=0):
        x = np.pad(x, (delay, 0))[:n]
        x = np.pad(x, (0, n - len(x)))
        L[:] += gl * x; R[:] += gr * x
    gtrs = [k for k in sig if k.startswith('Guitar')]
    for i, k in enumerate(gtrs):
        if amp_kind == 'nam':
            g = nam_amp(sig[k][0], second=i % 2 == 1)
        elif amp_kind == 'guitarix':
            off = settings().get('guitar2_drive_offset', 0.05) * i
            g = gx_amp(sig[k][0], tmp, f'g{i}', off)
        else:
            g = amp(sig[k][0], gain=55 + 10 * i)
        side = (1.0, 0.25) if i % 2 == 0 else (0.25, 1.0)          # hard-ish left / right
        add(g * 0.42, *side, delay=int(0.011 * SR) * (i % 2))       # double tracking offset
    if 'Bass' in sig:
        b = sig['Bass'][0]
        b = b / (np.abs(b).max() + 1e-9)
        low = lp(b, 250, 2)
        grit = lp(np.tanh(4 * hp(b, 300, 2)), 3000, 2)              # a bit of drive on top
        add((0.9 * low + 0.25 * grit) * 0.55, 1, 1)
    if 'Drums' in sig:
        st = sig['Drums'][1]
        if st.ndim == 2:
            d = st / (np.abs(st).max() + 1e-9)
            d = np.tanh(1.8 * d) / np.tanh(1.8)                     # punch / glue
            dl, dr = np.pad(d[:, 0], (0, n - len(d))), np.pad(d[:, 1], (0, n - len(d)))
            L[:] += 0.55 * dl[:n]; R[:] += 0.55 * dr[:n]
        else:
            add(st * 0.55, 1, 1)
    mix = np.stack([L, R], 1)
    mix = mix / (np.abs(mix).max() + 1e-9)
    mix = np.tanh(1.5 * mix) / np.tanh(1.5) * 0.95                  # gentle limiter
    wav = tmp / 'mix.wav'
    wavfile.write(str(wav), SR, (mix * 32767).astype(np.int16))
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(wav), '-b:a', '192k', str(out)], check=True)
    return out


def export_stems(src, outdir, sf2, amp_kind='builtin'):
    """One wav per part for an online mixer (RoEx Automix etc.): guitars through the amp, bass with a little
    drive, drums as rendered. No panning, no limiter, each peak at -6 dBFS so the mixer has headroom."""
    tmp = Path(tempfile.mkdtemp())
    src = Path(src)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    mid = src
    if src.suffix == '.gp5':
        mid = tmp / 'song.mid'
        to_midi_from_gp5(src, mid)
    parts = split_tracks(mid, tmp)
    if HUMANIZE[0]:
        for name, p in parts.items():
            if name.startswith('Guitar') and name != 'Guitar':
                humanize(p, HUMANIZE[0])
    sig = {name: synth(p, sf2, tmp) for name, p in parts.items()}
    n = max(len(m) for m, _ in sig.values())
    head = 10 ** (-6 / 20)
    written = []

    def save(name, x):
        x = np.asarray(x, dtype=np.float64)
        x = np.pad(x, [(0, n - len(x))] + [(0, 0)] * (x.ndim - 1))[:n]
        x = x / (np.abs(x).max() + 1e-9) * head
        p = outdir / f'{src.stem}_{name}.wav'
        wavfile.write(str(p), SR, (x * 32767).astype(np.int16))
        written.append(p)
    gtrs = [k for k in sig if k.startswith('Guitar')]
    for i, k in enumerate(gtrs):
        if amp_kind == 'nam':
            g = nam_amp(sig[k][0], second=i % 2 == 1)
        elif amp_kind == 'guitarix':
            g = gx_amp(sig[k][0], tmp, f'g{i}', settings().get('guitar2_drive_offset', 0.05) * i)
        else:
            g = amp(sig[k][0], gain=55 + 10 * i)
        save(f'guitar{i + 1}', g)
    if 'Bass' in sig:
        b = sig['Bass'][0]
        b = b / (np.abs(b).max() + 1e-9)
        save('bass', 0.9 * lp(b, 250, 2) + 0.25 * lp(np.tanh(4 * hp(b, 300, 2)), 3000, 2))
    if 'Drums' in sig:
        save('drums', sig['Drums'][1])
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('src', help='.gp5 or .mid made by generate.py')
    ap.add_argument('--out')
    ap.add_argument('--sf2')
    ap.add_argument('--ir', help='cabinet impulse response (.wav) applied after the Guitarix amp; '
                    'or set "cab_ir" in nn/amp_settings.json. Without one a rough 4x12 curve is used')
    ap.add_argument('--stems', metavar='DIR', help='write one wav per part into DIR (for online mixers) '
                    'instead of the mixed mp3')
    ap.add_argument('--nam', help='amp capture (.nam) for the guitars, e.g. nam/.../6505+ - RED - GAIN 5.nam. '
                    'Captures named FR... (full rig) already include the cabinet, so no IR is used for them')
    ap.add_argument('--nam2', help='another capture for guitar 2 (e.g. a different mic position) - more like a '
                    'real double track')
    ap.add_argument('--nam-pre', help='pedal capture in front of the amp, e.g. nam/.../IBANEZ TS808.nam')
    ap.add_argument('--nam-in', type=float, default=-10.0, help='DI peak level into the capture in dBFS '
                    '(higher = more gain), default -10')
    ap.add_argument('--humanize', type=float, default=8.0, help='guitar 2 timing spread in ms (0 = exact copy)')
    ap.add_argument('--amp', choices=['nam', 'guitarix', 'builtin'], default=None,
                    help='guitarix = real Guitarix plugins via lv2file (default when lv2file is installed); '
                         'builtin = the simple amp written in this file')
    ap.add_argument('--di', metavar='DIR', help='extracted UI_METAL-GTX folder: the guitars are played from its real '
                    'DI samples (palm mutes = real mute samples) instead of the GM SoundFont. Needs a .gp5 source')
    ap.add_argument('--bass-di', metavar='DIR', help='extracted Standard Bass folder (Unreal Instruments): the bass '
                    'from real samples. Needs a .gp5 source')
    ap.add_argument('--drumkit', metavar='DIR', help='unzipped DrumGizmo kit folder (e.g. CrocellKit): the drums '
                    'from a real multi-mic kit')
    ap.add_argument('--gate', type=float, default=None, metavar='DB', help='noise gate on the DI before the amp: '
                    'closes DB below the peak, e.g. -45 (tighter stops between chugs). Off when not given')
    ap.add_argument('--tight', type=float, default=0.0, metavar='HZ', help='low cut on the DI before the amp, '
                    'e.g. 120-200 (less low-end flub in the distortion, tighter chugs). Off when not given')
    a = ap.parse_args()
    # defaults for anything not given on the command line: "render" in nn/amp_settings.json
    for k, v in settings().get('render', {}).items():
        k = k.replace('-', '_')
        if hasattr(a, k) and getattr(a, k) in (None, 0.0, -10.0) and v is not None:
            setattr(a, k, v)
    import shutil
    NAM['gate'] = a.gate
    NAM['tight'] = a.tight
    for flag, val, slot in (('--bass-di', a.bass_di, BASS_DI), ('--drumkit', a.drumkit, DRUMKIT)):
        if val:
            if not Path(val).expanduser().exists():
                sys.exit(f'not found: {val}')
            if flag == '--bass-di' and Path(a.src).suffix != '.gp5':
                sys.exit('--bass-di needs the .gp5')
            slot[0] = str(Path(val).expanduser())
            print(f'{flag[2:]}: {slot[0]}')
    if a.di:
        if not Path(a.di).expanduser().exists():
            sys.exit(f'not found: {a.di}')
        if Path(a.src).suffix != '.gp5':
            sys.exit('--di needs the .gp5 (palm mutes are read from it)')
        DI[0] = str(Path(a.di).expanduser())
        print(f'guitar DI: {DI[0]}')
    HUMANIZE[0] = a.humanize
    if a.nam:
        for k, v in (('amp', a.nam), ('amp2', a.nam2), ('pre', a.nam_pre)):
            if v and not Path(v).expanduser().exists():
                sys.exit(f'not found: {v}')
            NAM[k] = v
        NAM['in_db'] = a.nam_in
    amp_kind = a.amp or ('nam' if a.nam else 'guitarix' if shutil.which('lv2file') else 'builtin')
    print(f'amp: {amp_kind}' + ('  (knobs: nn/amp_settings.json)' if amp_kind == 'guitarix' else ''))
    sf2 = next((p for p in ([a.sf2] if a.sf2 else []) + SF2S if p and Path(p).exists()), None)
    if not sf2:
        sys.exit('no GM SoundFont found (sudo apt install fluid-soundfont-gm)')
    if a.ir:
        if not Path(a.ir).expanduser().exists():
            sys.exit(f'IR not found: {a.ir}')
        CAB_IR[0] = a.ir
    print('cab: ' + (CAB_IR[0] or settings().get('cab_ir') or 'built-in 4x12 curve (no IR given)'))
    src = Path(a.src)
    if a.stems:
        for p in export_stems(src, a.stems, sf2, amp_kind):
            print(f'wrote {p}')
        return
    out = Path(a.out) if a.out else src.with_name(src.stem + '_metal.mp3')
    print(f'wrote {render(src, out, sf2, amp_kind)}')


if __name__ == '__main__':
    main()
