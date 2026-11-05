"""Root progressions of the reference songs (the paper's chord-pattern mining, for metal riffs).

  python3 nn/roots.py          (reads nn/data/tokens, adds "roots" to nn/data/structure.json)

Each bar of the main guitar gets a root: the pitch class of its lowest note, weighted by how
long it sounds. Roots are taken relative to the song's tonic (its most common root), so
0 = the pedal / home note, 1 = minor 2nd above (the dark semitone), 6 = tritone, 11 = semitone below...
Frequent root patterns (2-4 bars) are mined Apriori-style: a pattern is kept if enough songs
contain it, and longer patterns are only built from frequent shorter ones.
"""
import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
BASE = {'s1': 64, 's2': 59, 's3': 55, 's4': 50, 's5': 45, 's6': 40, 's7': 35}
NAMES = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', 'b6', '6', 'b7', '7']


def pitch(tok, downtune=0):
    _, _, s, f = tok.split(':')
    return BASE[s] + int(f[1:]) + downtune


def bar_roots(tokens):
    """Root pitch class per bar (None for bars without guitar)."""
    dt = next((int(t.split(':')[1]) for t in tokens if t.startswith('downtune:')), 0)
    roots, weight, step = [], None, []
    for t in tokens:
        if t == 'new_measure':
            if weight is not None:
                roots.append(max(weight, key=weight.get) if weight else None)
            weight, step = collections.Counter(), []
        elif weight is None:
            continue
        elif t.startswith('distorted0:note'):
            step.append(pitch(t, dt))
        elif t.startswith('wait:'):
            if step:
                weight[min(step) % 12] += int(t[5:])
            step = []
    if weight is not None:
        roots.append(max(weight, key=weight.get) if weight else None)
    return roots


def relative(roots):
    known = [r for r in roots if r is not None]
    if not known:
        return [], None
    tonic = collections.Counter(known).most_common(1)[0][0]
    return [None if r is None else (r - tonic) % 12 for r in roots], tonic


def collapse(seq):
    """Drop repeated neighbours: 0 0 0 1 1 0 -> 0 1 0 (patterns describe root *changes*)."""
    out = []
    for x in seq:
        if x is not None and (not out or out[-1] != x):
            out.append(x)
    return out


def mine(songs, min_support=0.3, max_len=4):
    """Apriori over contiguous root patterns: support = share of songs containing the pattern."""
    n = len(songs)
    def support(p):
        L = len(p)
        return sum(any(tuple(s[i:i + L]) == p for i in range(len(s) - L + 1)) for s in songs) / n
    freq = {}
    level = {(x,): support((x,)) for x in range(12)}
    level = {p: v for p, v in level.items() if v >= min_support}
    for L in range(1, max_len + 1):
        freq.update(level)
        if L == max_len:
            break
        cands = {p + (q[-1],) for p in level for q in level if p[1:] == q[:-1] and p[-1] != q[-1]}
        level = {c: support(c) for c in cands}
        level = {p: v for p, v in level.items() if v >= min_support}
    return freq


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tokens', default=str(DATA / 'tokens'))
    ap.add_argument('--structure', default=str(DATA / 'structure.json'))
    ap.add_argument('--min-support', type=float, default=0.3)
    a = ap.parse_args()

    rel, bigram, unigram, barbi = [], collections.Counter(), collections.Counter(), collections.Counter()
    for p in sorted(Path(a.tokens).glob('*.txt')):
        toks = p.read_text().split()
        if not any(t.startswith('distorted0:note') for t in toks):
            continue
        r, _ = relative(bar_roots(toks))
        c = collapse(r)
        rel.append(c)
        unigram.update(x for x in r if x is not None)
        bigram.update(zip(c, c[1:]))
        rr = [x for x in r if x is not None]
        barbi.update(zip(rr, rr[1:]))
        print(f'{p.stem[:45]:45s} ' + ' '.join(NAMES[x] for x in c[:28]))
    freq = mine(rel, a.min_support)
    tot = sum(unigram.values())
    print('\nbar roots (share of bars):', ', '.join(f'{NAMES[k]} {v / tot:.0%}' for k, v in unigram.most_common(8)))
    bt = sum(bigram.values())
    print('root moves:', ', '.join(f'{NAMES[x]}->{NAMES[y]} {v / bt:.0%}' for (x, y), v in bigram.most_common(10)))
    longest = sorted((p for p in freq if len(p) >= 3), key=lambda p: -freq[p])[:12]
    print(f'frequent patterns (in >= {a.min_support:.0%} of songs):',
          ', '.join(f"{'-'.join(NAMES[x] for x in p)} {freq[p]:.0%}" for p in longest))

    sp = Path(a.structure)
    model = json.loads(sp.read_text()) if sp.exists() else {}
    model['roots'] = dict(
        unigram={str(k): v / tot for k, v in unigram.items()},
        bigram={f'{x},{y}': v / bt for (x, y), v in bigram.items()},
        bar_bigram={f'{x},{y}': v for (x, y), v in barbi.items()},
        patterns=[[list(p), v] for p, v in freq.items() if len(p) >= 2])
    sp.write_text(json.dumps(model, indent=1))
    print(f'saved roots to {sp}')


if __name__ == '__main__':
    main()
