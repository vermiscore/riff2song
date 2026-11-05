"""Learn song form from the reference songs (after Shan & Chiu 2009, "Algorithmic
compositions based on discovered musical patterns"), adapted to metal riffs.

  python3 nn/structure.py            (reads nn/data/tokens, writes nn/data/structure.json)

1. Each bar of the main guitar becomes a symbol (same rhythm + same notes = same symbol).
2. Repeating patterns of 2/4/8/16 bars are mined with a correlative matrix.
3. Non-overlapping pattern instances covering the most bars are chosen by dynamic
   programming (the paper's exon-chaining step); the chosen instances become sections,
   the gaps between them become one-off sections.
4. Sections are labelled A, B, C... (same pattern = same label) and summarised:
   song forms, section lengths, and riff forms inside sections (A A A B, A B A C, ...).
"""
import argparse
import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
LENGTHS = (16, 8, 4, 2)


def bars_of(tokens, inst='distorted0'):
    """Split a token list into bars; return (symbol tuple per bar) for one instrument."""
    bars, cur, t, step = [], None, 0, []
    for x in tokens:
        if x == 'new_measure':
            if cur is not None:
                bars.append(tuple(cur))
            cur, t, step = [], 0, []
        elif cur is None:
            continue
        elif x.startswith(inst + ':note'):
            step.append(x.split(':', 1)[1])
        elif x.startswith('wait:'):
            if step:
                cur.append((t, tuple(sorted(step))))
            t += int(x[5:]); step = []
    if cur is not None:
        bars.append(tuple(cur))
    return bars


def skeleton(bar):
    """Looser bar identity: rhythm (onsets) + lowest note of each hit, ignoring ties/extra chord tones,
    so small riff-end variations still count as the same bar."""
    out = []
    for t, notes in bar:
        low = max(notes, key=lambda n: (int(n.split(':')[1][1:]), -int(n.split(':f')[1])))
        out.append((t // 120, low))
    return tuple(out)


def symbolize(bars, loose=True):
    ids, table = [], {}
    for b in bars:
        k = skeleton(b) if loose else b
        ids.append(table.setdefault(k, len(table)) if b else -1)   # -1 = empty bar
    return ids


def repeating_instances(s):
    """Correlative matrix over bar symbols -> {pattern tuple: set(start positions)}."""
    n = len(s)
    occ = collections.defaultdict(set)
    prev = [0] * (n + 1)
    for i in range(n):
        cur = [0] * (n + 1)
        for j in range(i + 1, n):
            if s[i] == s[j] and s[i] != -1:
                cur[j + 1] = prev[j] + 1
                L = cur[j + 1]
                for P in LENGTHS:
                    if L >= P and j - i >= P:              # instances must not overlap
                        occ[tuple(s[i - P + 1:i + 1])].update((i - P + 1, j - P + 1))
        prev = cur
    return occ


def choose_sections(n, occ):
    """Exon chaining: non-overlapping instances maximizing covered bars (longer patterns win ties)."""
    inst = [(st, st + len(p), p) for p, sts in occ.items() for st in sts]
    inst.sort(key=lambda x: x[1])
    best = [0.0] * (n + 1)        # best[k]: best score using bars < k
    pick = [None] * (n + 1)
    by_end = collections.defaultdict(list)
    for st, en, p in inst:
        by_end[en].append((st, p))
    for k in range(1, n + 1):
        best[k], pick[k] = best[k - 1], None
        for st, p in by_end.get(k, []):
            sc = best[st] + len(p) ** 1.3       # favour long patterns (sections, not 2-bar bits)
            if sc > best[k]:
                best[k], pick[k] = sc, (st, p)
    out, k = [], n
    while k > 0:
        if pick[k] is None:
            k -= 1
        else:
            st, p = pick[k]
            out.append((st, k, p)); k = st
    return out[::-1]


def riff_form(syms):
    """Bar symbols -> letters by first appearance, e.g. (5,5,5,9) -> 'AAAB'."""
    m = {}
    return ''.join(m.setdefault(x, 'ABCDEFGHIJKLMNOP'[min(len(m), 15)]) for x in syms)


def analyse(tokens):
    s = symbolize(bars_of(tokens))
    n = len(s)
    chosen = choose_sections(n, repeating_instances(s))
    secs, pos = [], 0
    for st, en, p in chosen:
        if st > pos:
            secs.append((pos, st, tuple(s[pos:st]), False))
        secs.append((st, en, p, True)); pos = en
    if pos < n:
        secs.append((pos, n, tuple(s[pos:n]), False))
    labels, out = {}, []
    for st, en, p, rep in secs:
        key = p if rep else ('once', st)
        lab = labels.setdefault(key, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[len(labels) % 26] +
                                ('' if len(labels) < 26 else str(len(labels) // 26)))
        out.append(dict(label=lab, start=st, bars=en - st, repeated=rep, form=riff_form(p)))
    return n, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--tokens', default=str(DATA / 'tokens'))
    ap.add_argument('--out', default=str(DATA / 'structure.json'))
    a = ap.parse_args()

    songs, riff4, riff2, sec_len, counts = [], collections.Counter(), collections.Counter(), \
        collections.Counter(), []
    for p in sorted(Path(a.tokens).glob('*.txt')):
        toks = p.read_text().split()
        if not any(t.startswith('distorted0:note') for t in toks):
            print(f'skip (no main guitar): {p.stem}'); continue
        n, secs = analyse(toks)
        songs.append(dict(name=p.stem, bars=n, sections=secs))
        counts.append(len(secs))
        for sc in secs:
            sec_len[sc['bars']] += 1
            f = sc['form']
            for i in range(0, len(f) - 3, 4):           # riff forms in 4-bar groups
                riff4[riff_form(f[i:i + 4])] += 1
            for i in range(0, len(f) - 1, 2):
                riff2[riff_form(f[i:i + 2])] += 1
        form = ' '.join(f"{sc['label']}{sc['bars']}" for sc in secs)
        cover = sum(sc['bars'] for sc in secs if sc['repeated']) / max(1, n)
        print(f'{p.stem[:45]:45s} {n:4d} bars  {cover:4.0%} repeated   {form[:110]}')

    def dist(c, top=8):
        tot = sum(c.values())
        return [(k, round(v / tot, 3)) for k, v in c.most_common(top)]
    model = dict(songs=songs, riff4=dist(riff4), riff2=dist(riff2), section_bars=dist(sec_len, 12))
    Path(a.out).write_text(json.dumps(model, indent=1))
    print('\nriff forms in 4-bar groups:', ', '.join(f'{k} {v:.0%}' for k, v in model['riff4']))
    print('riff forms in 2-bar groups:', ', '.join(f'{k} {v:.0%}' for k, v in model['riff2']))
    print('section lengths (bars):', ', '.join(f'{k} {v:.0%}' for k, v in model['section_bars']))
    print(f'saved {a.out}')


if __name__ == '__main__':
    main()
