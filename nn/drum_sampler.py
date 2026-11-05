"""Render the drum track from a real multi-mic drum kit of DrumGizmo (e.g. CrocellKit, the kit of the Danish
metal band Crocell) instead of the GM SoundFont drums. No DrumGizmo program is needed: the kit's XML files and
wav samples are read directly.

Used by render_metal.py --drumkit DIR (DIR = the unzipped kit folder, e.g. .../CrocellKit). Standalone check:
    python3 nn/drum_sampler.py works/song.mid /path/to/CrocellKit --out /tmp/drums_check.wav

  - the kit file (*_full.xml, else the largest *.xml with <instruments>) and its midimap (Midimap_full.xml,
    else the first Midimap*.xml) are found automatically
  - GM notes the midimap lacks are sent to the nearest instrument (china -> crash, bell -> ride, ...)
  - each hit picks a sample whose recorded power matches the velocity, never the same one twice in a row
  - all microphones of a hit are summed to stereo: overheads / room left-right, close mics centred
"""
import argparse
import random
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

SR = 44100
FALLBACK = {52: [49, 57, 55], 53: [51, 59], 55: [49, 57], 57: [49], 59: [51], 44: [42], 46: [42], 35: [36],
            40: [38], 37: [38], 41: [43, 45], 43: [41, 45], 45: [47, 43], 47: [45, 48], 48: [50, 47], 50: [48]}
_CACHE = {}


def find_files(root):
    root = Path(root).expanduser()
    kits = [p for p in root.rglob('*.xml') if 'midimap' not in p.name.lower()]
    kits = [p for p in kits if '<instruments' in p.read_text(errors='ignore')[:20000]]
    if not kits:
        raise SystemExit(f'no drumkit xml under {root}')
    kit = next((p for p in kits if 'full' in p.name.lower()), max(kits, key=lambda p: p.stat().st_size))
    maps = sorted(p for p in root.rglob('*.xml') if 'midimap' in p.name.lower())
    if not maps:
        raise SystemExit(f'no Midimap*.xml under {root}')
    mm = next((p for p in maps if 'full' in p.name.lower()), maps[0])
    return kit, mm


def load_kit(kit_xml):
    """{instrument: [(power, [(channel, wav_path, filechannel), ...]), ...]}"""
    kit_xml = Path(kit_xml)
    root = ET.parse(kit_xml).getroot()
    out = {}
    for ins in root.iter('instrument'):
        name, f = ins.get('name'), ins.get('file')
        if not name or not f:
            continue
        ipath = (kit_xml.parent / f).resolve()
        iroot = ET.parse(ipath).getroot()
        samples = []
        for smp in iroot.iter('sample'):
            pw = float(smp.get('power', 0) or 0)
            files = [(a.get('channel'), (ipath.parent / a.get('file')).resolve(), int(a.get('filechannel', 1)))
                     for a in smp.iter('audiofile')]
            if files:
                samples.append((pw, files))
        if samples:
            out[name] = sorted(samples, key=lambda x: x[0])
    return out


def load_midimap(mm_xml):
    m = {}
    for e in ET.parse(mm_xml).getroot().iter('map'):
        m.setdefault(int(e.get('note')), []).append(e.get('instr'))
    return m


def wav(path, ch):
    key = (str(path), ch)
    if key in _CACHE:
        return _CACHE[key]
    try:
        import soundfile as sf
        x, sr = sf.read(str(path), dtype='float32', always_2d=True)
    except Exception:
        from scipy.io import wavfile
        sr, raw = wavfile.read(str(path))
        scale = {'int16': 32768.0, 'int32': 2147483648.0, 'uint8': 128.0}.get(raw.dtype.name, 1.0)
        x = raw.astype(np.float32) / scale
        x = x[:, None] if x.ndim == 1 else x
    y = x[:, min(ch, x.shape[1]) - 1]
    if sr != SR:
        y = np.interp(np.arange(0, len(y), sr / SR), np.arange(len(y)), y).astype(np.float32)
    _CACHE[key] = y
    return y


def pan_gain(channel):
    c = channel.lower()
    g = 1.0
    if 'amb' in c or 'room' in c:
        g = 0.55
    elif c.startswith('oh') or 'overhead' in c:
        g = 0.8
    elif 'bottom' in c:
        g = 0.35
    if 'left' in c or re.search(r'(^|[^a-z])l$', c):
        return g * 1.0, g * 0.25
    if 'right' in c or re.search(r'(^|[^a-z])r$', c):
        return g * 0.25, g * 1.0
    return g * 0.8, g * 0.8


def hits_of(mid_path):
    import mido
    mf = mido.MidiFile(str(mid_path))
    out = []
    for msg_t, msg in _abs_messages(mf):
        if msg.type == 'note_on' and msg.velocity > 0:
            out.append((msg_t, msg.note, msg.velocity))
    return out


def _abs_messages(mf):
    t = 0.0
    for msg in mf:                         # iterating a MidiFile yields seconds and follows tempo changes
        t += msg.time
        if not msg.is_meta:
            yield t, msg


def render(mid_path, root, seed=0):
    """Stereo float32 (n, 2) of the drum hits in mid_path."""
    kit_xml, mm_xml = find_files(root)
    kit, mm = load_kit(kit_xml), load_midimap(mm_xml)
    print(f'  drumkit: {kit_xml.name} ({len(kit)} instruments), map: {mm_xml.name}')
    rng = random.Random(seed)
    hits = hits_of(mid_path)
    n = int((max((h[0] for h in hits), default=0) + 4) * SR)
    y = np.zeros((n, 2), dtype=np.float32)
    last, missing = {}, set()
    for t, note, vel in hits:
        names = mm.get(note) or next((mm[f] for f in FALLBACK.get(note, []) if f in mm), None)
        if not names:
            missing.add(note)
            continue
        for name in names:
            smps = kit.get(name)
            if not smps:
                continue
            q = (vel / 127.0) ** 1.3
            i = int(round(q * (len(smps) - 1)))
            cand = [j for j in range(max(0, i - 2), min(len(smps), i + 3)) if j != last.get(name)] or [i]
            j = rng.choice(cand)
            last[name] = j
            a = int(t * SR)
            for ch, path, fch in smps[j][1]:
                x = wav(path, fch)[:n - a]
                gl, gr = pan_gain(ch)
                y[a:a + len(x), 0] += gl * x
                y[a:a + len(x), 1] += gr * x
    if missing:
        print(f'  drum notes without an instrument: {sorted(missing)}')
    return y


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('mid')
    ap.add_argument('root')
    ap.add_argument('--out', default='/tmp/drums_check.wav')
    a = ap.parse_args()
    from scipy.io import wavfile
    y = render(a.mid, a.root)
    y = y / (np.abs(y).max() + 1e-9) * 0.7
    wavfile.write(a.out, SR, (y * 32767).astype(np.int16))
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
