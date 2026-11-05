"""Note-choice palettes: how the riffs of a song pick their notes, measured from the references.

  python3 nn/palette.py mine        (nn/data/tokens + section names of ../tabs_ref/*.gp -> nn/data/palettes.json)
  python3 nn/palette.py list

For every song, each section is one riff. Inside a riff the most common low note is its pedal/root;
every guitar note is counted as an interval above it (1 b2 2 b3 3 4 b5 5 b6 6 b7 7), and moves between
non-pedal notes are counted by size (semitone, whole tone, minor/major third, fourth, tritone, fifth,
octave, other). A palette = those two distributions for one song. In the references, riffs of the same
song share their palette (similarity 0.51 vs 0.37 between songs), so using one palette for all riffs of
a generated song keeps it coherent.

generate.py --palette NAME (part of a song name; default: the song of the --form template) biases
  * NN sampling: each guitar note token gets  strength * (log P(interval) + log P(move from previous note))
  * prefab choice: pitch prefabs are weighted by how well their notes fit the palette
"""
import argparse
import collections
import json
import math
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUT = DATA / 'palettes.json'
BASE = {'s1': 64, 's2': 59, 's3': 55, 's4': 50, 's5': 45, 's6': 40, 's7': 35}
IV = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', 'b6', '6', 'b7', '7']
V9 = {'intervals': [0.664, 0.0448, 0.0102, 0.0346, 0.001, 0.058, 0.1314, 0.0244, 0.0102, 0.0, 0.0132, 0.0081],
      'moves': {'semi': 0.425, 'whole': 0.1893, 'm3': 0.0607, 'M3': 0.0321, '4th': 0.0679, 'tritone': 0.1071,
                '5th': 0.0036, 'octave': 0.0, 'other': 0.1143}}   # the hand-arranged song of this project
MOVES = ['semi', 'whole', 'm3', 'M3', '4th', 'tritone', '5th', 'octave', 'other']


def move_class(d):
    d = abs(d)
    return {1: 'semi', 2: 'whole', 3: 'm3', 4: 'M3', 5: '4th', 6: 'tritone', 7: '5th', 12: 'octave'}.get(d, 'other')


def bars_of(tokens):
    out, cur, t, step = [], None, 0, []
    for x in tokens:
        if x == 'new_measure':
            if cur is not None:
                out.append(cur)
            cur, t, step = [], 0, []
        elif cur is None:
            continue
        elif x.startswith('distorted0:note'):
            _, _, s, f = x.split(':')
            step.append(BASE[s] + int(f[1:]))
        elif x.startswith('wait:'):
            if step:
                cur.append(min(step))
            t += int(x[5:]); step = []
    if cur is not None:
        out.append(cur)
    return out


def section_starts(gp):
    try:
        x = zipfile.ZipFile(gp).read('Content/score.gpif').decode('utf-8', 'replace')
    except Exception:
        return []
    mbs = re.findall(r'<MasterBar>(.*?)</MasterBar>', x, re.S)
    return [i for i, m in enumerate(mbs) if re.search(r'<Section>.*?<Text>\s*<!\[CDATA\[\s*\S', m, re.S)]


def profile(bar_lists):
    pcs, mv = collections.Counter(), collections.Counter()
    for bars in bar_lists:
        lows = [p for b in bars for p in b]
        if len(lows) < 8:
            continue
        root = collections.Counter(lows).most_common(1)[0][0]
        pcs.update((p - root) % 12 for p in lows)
        mel = [p for p in lows if p != root]
        mv.update(move_class(b - a) for a, b in zip(mel, mel[1:]) if a != b)
    if not pcs:
        return None
    tp, tm = sum(pcs.values()), sum(mv.values()) or 1
    return dict(intervals=[pcs.get(i, 0) / tp for i in range(12)], moves={k: mv.get(k, 0) / tm for k in MOVES})


def mine(token_dir, ref_dir, extra=None):
    pals = {}
    for p in sorted(Path(token_dir).glob('*.txt')):
        B = bars_of(p.read_text().split())
        marks = section_starts(Path(ref_dir) / (p.stem + '.gp'))
        if len(marks) < 4:
            marks = list(range(0, len(B), 8))
        marks = sorted(set(marks)) + [len(B)]
        prof = profile([B[a:b] for a, b in zip(marks, marks[1:])])
        if prof:
            pals[p.stem] = prof
    for name, path in (extra or {}).items():
        if Path(path).exists():
            B = bars_of(Path(path).read_text().split())
            prof = profile([B[a:a + 8] for a in range(0, len(B), 8)])
            if prof:
                pals[name] = prof
    pals['v9 (this project)'] = V9
    OUT.write_text(json.dumps(pals, indent=1))
    return pals


def load():
    return json.loads(OUT.read_text()) if OUT.exists() else None


def base(pals):
    return [sum(p['intervals'][i] for p in pals.values()) / len(pals) for i in range(12)]


def find(pals, name):
    hits = [k for k in pals if name.lower() in k.lower()]
    return hits[0] if hits else None


class Palette:
    """Logit bias for the NN and weights for prefabs from one song's palette."""
    def __init__(self, prof, vocab, tonic, strength=1.0, base=None):
        import torch
        base = base or [1 / 12] * 12                     # average palette of all songs (for prefab weights)
        self.ratio = [math.log((prof['intervals'][i] + 0.01) / (base[i] + 0.01)) for i in range(12)]
        self.strength = strength
        self.tonic = tonic
        self.iv = [math.log(x + 0.01) for x in prof['intervals']]
        self.mv = {k: math.log(v + 0.02) for k, v in prof['moves'].items()}
        pitch = []
        for t in vocab:
            if t.startswith('distorted0:note'):
                _, _, s, f = t.split(':')
                pitch.append(BASE.get(s, 40) + int(f[1:]))
            else:
                pitch.append(-1)
        self.pitch = torch.tensor(pitch)
        is_note = self.pitch >= 0
        self.is_note = is_note
        self.static = torch.zeros(len(vocab))
        pcs = ((self.pitch - tonic) % 12).clamp(min=0)
        ivt = torch.tensor(self.iv)
        self.static[is_note] = strength * ivt[pcs[is_note]]
        self.static[is_note] -= self.static[is_note].max()
        # move bias per distance 0..48
        self.mvt = torch.tensor([self.mv.get(move_class(d), self.mv['other']) if d else 0.0 for d in range(49)])
        self.low = None

    def bias(self, prev_mel):
        b = self.static.clone()
        if prev_mel is not None:
            d = (self.pitch - prev_mel).abs().clamp(max=48)
            mv = self.strength * 0.5 * self.mvt[d]
            nonped = self.is_note & (((self.pitch - self.tonic) % 12) != 0)
            b[nonped] += mv[nonped] - mv[nonped].max()
        return b

    def prefab_weight(self, rel, root_rel):
        """How well a pitch prefab (semitones above the bar root) fits: exp(strength * mean log P)."""
        pcs = [(root_rel + r) % 12 for r in rel]
        s = sum(self.ratio[p] for p in pcs) / max(1, len(pcs))   # how much more typical of THIS song than of all
        mel = [r for r in rel if r % 12]
        s += 0.5 * sum(self.mv.get(move_class(b - a), self.mv['other']) for a, b in zip(mel, mel[1:]) if a != b) / max(1, len(mel))
        return math.exp(self.strength * s)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['mine', 'list'])
    ap.add_argument('--tokens', default=str(DATA / 'tokens'))
    ap.add_argument('--refs', default=str(HERE.parent.parent / 'tabs_ref'))
    a = ap.parse_args()
    if a.cmd == 'mine':
        pals = mine(a.tokens, a.refs)
        print(f'{len(pals)} palettes saved to {OUT}')
    pals = load()
    if not pals:
        sys.exit('run: python3 nn/palette.py mine')
    for k, p in pals.items():
        top = sorted(range(12), key=lambda i: -p['intervals'][i])[:6]
        mv = sorted(p['moves'].items(), key=lambda x: -x[1])[:3]
        print(f'{k[:40]:40s} pedal {100 * p["intervals"][0]:3.0f}%  ' + ' '.join(f'{IV[i]}:{100 * p["intervals"][i]:.0f}' for i in top[1:])
              + '   ' + ' '.join(f'{m}:{100 * v:.0f}' for m, v in mv))


if __name__ == '__main__':
    main()
