"""Prefab riff composer (after Malcolm Sailor's dumb_composer, adapted to metal riffs).

dumb_composer realizes a chord progression with hand-written "prefabs" (rhythm and melodic
patterns) under strict rules. Here the prefabs are mined from your reference songs instead:

  rhythm prefabs   one bar of the main guitar: onsets, lengths and how each hit is played
                   (p = palm-muted, n = open single note, c = chord, x = dead note)
  pitch prefabs    the notes of those hits as semitones above the bar's root (0 = root)

  python3 nn/prefab.py mine      (reads nn/data/tokens, writes nn/data/prefabs.json, shows the top ones)

Generation (used by generate.py --phrases N --prefab 0.5): pick a riff form (A A A A, A B A C...)
and a root progression from structure.json, then fill each new bar with a rhythm prefab and a
pitch prefab with the same number of hits, on that bar's root. Bars that break the rules
(off the neck, jumps over an octave, too many chords) are thrown away and redrawn.
"""
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
BAR = 3840
Q = 60                                   # tick grid for rhythm cells (1/64 note)
GBASE = {'s1': 64, 's2': 59, 's3': 55, 's4': 50, 's5': 45, 's6': 40, 's7': 35}
BBASE = {'s1': 43, 's2': 38, 's3': 33, 's4': 28, 's5': 23, 's6': 18}


def _pitch(tok, base):
    _, _, s, f = tok.split(':')
    return base[s] + int(f[1:])


def bars_with_hits(tokens):
    """Per bar: list of hits (onset, length, kind, [pitches]) of the main guitar."""
    out, cur, t, notes, effects = [], None, 0, [], set()
    events = []

    def close_step():
        if notes:
            kind = 'x' if 'dead' in effects else ('c' if len(set(p % 12 for p in notes)) >= 2 or len(notes) >= 2
                                                   else ('p' if 'pm' in effects else 'n'))
            events.append([t, None, kind, sorted(notes)])
        elif rest:
            events.append([t, None, 'r', []])

    def close_bar():
        for i, e in enumerate(events):
            e[1] = (events[i + 1][0] if i + 1 < len(events) else BAR) - e[0]
        out.append([tuple(e[:3]) + (tuple(e[3]),) for e in events if e[2] != 'r'])

    rest, last_inst = False, ''
    for x in tokens:
        if x == 'new_measure':
            if cur is not None:
                close_step(); close_bar()
            cur, t, notes, effects, rest, events = True, 0, [], set(), False, []
        elif cur is None:
            continue
        elif x.startswith('distorted0:note'):
            notes.append(_pitch(x, GBASE)); last_inst = 'g'
        elif x == 'distorted0:rest':
            rest = True; last_inst = 'g'
        elif x.startswith(('bass:', 'distorted', 'clean', 'drums', 'leads', 'pads')):
            last_inst = 'o'
        elif x.startswith('nfx:') and last_inst == 'g':
            if 'palm_mute' in x:
                effects.add('pm')
            if 'dead' in x:
                effects.add('dead')
        elif x.startswith('wait:'):
            close_step()
            t += int(x[5:]); notes, effects, rest = [], set(), False
    if cur is not None:
        close_step(); close_bar()
    return out


def root_of_hits(hits):
    w = collections.Counter()
    for on, ln, kind, ps in hits:
        w[min(ps) % 12] += ln
    return max(w, key=w.get) if w else None


HALF = BAR // 2


def mine(token_dir):
    """Prefabs are half bars (2 beats): enough variety, and halves combine into bars the way they
    follow each other in the references (first half -> second half statistics)."""
    rhythm, pitch, chord_shapes = collections.Counter(), collections.defaultdict(collections.Counter), \
        collections.Counter()
    rsongs, follow = collections.defaultdict(set), collections.defaultdict(collections.Counter)
    files = sorted(Path(token_dir).glob('*.txt'))
    for p in files:
        for hits in bars_with_hits(p.read_text().split()):
            if len(hits) < 2 or sum(h[1] for h in hits) < BAR // 2:
                continue
            r = root_of_hits(hits)
            low = min(min(h[3]) for h in hits)
            base = low - ((low - r) % 12)           # the root at or below the lowest note
            keys = []
            for h0 in (0, HALF):
                hh = [h for h in hits if h0 <= h[0] < h0 + HALF]
                if not hh or hh[0][0] != h0:        # a half must start with a hit on its downbeat
                    keys.append(None); continue
                key = tuple(((on - h0) // Q, max(1, min(ln, h0 + HALF - on) // Q), kind) for on, ln, kind, _ in hh)
                keys.append(key)
                rhythm[key] += 1
                rsongs[key].add(p.stem)
                rel = tuple(min(h[3]) - base for h in hh)
                if max(rel) <= 24:
                    pitch[len(hh)][rel] += 1
            if keys[0] and keys[1]:
                follow[keys[0]][keys[1]] += 1
            for h in hits:
                if h[2] == 'c':
                    chord_shapes[tuple(x - min(h[3]) for x in h[3])] += 1
    # keep rhythms that appear in at least 2 songs (idioms, not one song's quirk)
    keep = {k for k in rhythm if len(rsongs[k]) >= 2}
    order = [k for k, _ in rhythm.most_common() if k in keep]
    idx = {k: i for i, k in enumerate(order)}
    rh = [[list(map(list, k)), rhythm[k], len(rsongs[k])] for k in order]
    fol = {str(idx[a]): [[idx[b], n] for b, n in c.items() if b in idx] for a, c in follow.items() if a in idx}
    pt = {str(n): [[list(k), v] for k, v in c.most_common(300) if v >= 2] for n, c in pitch.items()}
    shapes = [[list(k), v] for k, v in chord_shapes.most_common(8)]
    return dict(rhythm=rh, follow=fol, pitch=pt, chord_shapes=shapes, songs=len(files))


def show(model):
    names = {'p': 'p', 'n': 'n', 'c': 'C', 'x': 'x'}

    def grid(cells):                               # 16th-note picture of half a bar (2 beats)
        s = ['.'] * 8
        for on, ln, kind in cells:
            i = on * Q // 240
            if i < 8:
                s[i] = names[kind] if (on * Q) % 240 == 0 else names[kind].lower() + "'"[:0]
                for j in range(i + 1, min(8, (on + ln) * Q // 240)):
                    s[j] = '-'
        return ''.join(s[:4]) + ' ' + ''.join(s[4:])
    print(f"{len(model['rhythm'])} half-bar rhythm prefabs (used in >= 2 of {model['songs']} songs). Top 15:")
    print('   16ths: 1   2      (p palm-muted, n open note, C chord, x dead, - held)')
    for cells, n, songs in model['rhythm'][:15]:
        print(f'   {grid(cells)}   x{n} in {songs} songs')
    print('\nchord shapes (semitones above the low note):', ', '.join(f'{s} x{n}' for s, n in model['chord_shapes'][:5]))
    for k in ('4', '8'):
        if k in model['pitch']:
            print(f'pitch prefabs for {k} hits (semitones above the root):',
                  ', '.join(str(tuple(p)) for p, _ in model['pitch'][k][:6]))


# ---------------------------------------------------------------- generation

class PrefabComposer:
    def __init__(self, model, rng, home=2):
        self.rng = rng
        self.rhythm = [([tuple(c) for c in cells], n) for cells, n, _ in model['rhythm']]
        self.follow = {int(k): v for k, v in model.get('follow', {}).items()}
        self.pitch = {int(k): v for k, v in model['pitch'].items()}
        self.shape = [tuple(s) for s, _ in model['chord_shapes'][:3]] or [(0, 7)]
        self.home = home                    # pitch class of the NN's home note (fret pitch, no downtune)
        self.palette = None                 # optional palette.Palette: weights pitch prefabs

    def _place(self, pitch, strings, base, lowest):
        """Lowest-position string/fret for a pitch: (string, fret) or None."""
        for s in strings:
            f = pitch - base[s]
            if lowest.get(s, 0) <= f <= 15:
                return s, f
        return None

    def bar(self, root_rel, tries=60):
        """One bar as DadaGP tokens (without 'new_measure') on a root (semitones above home)."""
        r = self.rng
        root = 38 + ((self.home - 2 + root_rel) % 12)        # D2 = open dropped low string area
        for _ in range(tries):
            i1 = r.choices(range(len(self.rhythm)), weights=[n for _, n in self.rhythm])[0]
            nxt = self.follow.get(i1)
            if nxt and r.random() < 0.85:
                i2 = r.choices([b for b, _ in nxt], weights=[w for _, w in nxt])[0]
            else:
                i2 = r.choices(range(len(self.rhythm)), weights=[n for _, n in self.rhythm])[0]
            halves = [self.rhythm[i1][0], self.rhythm[i2][0]]
            cells, rel = [], []
            for h, half in enumerate(halves):
                cand = self.pitch.get(len(half))
                if not cand:
                    break
                ws = [w * (self.palette.prefab_weight(pp, root_rel) if self.palette else 1) for pp, w in cand]
                rel += list(r.choices([pp for pp, _ in cand], weights=ws)[0])
                cells += [(on + h * HALF // Q, ln, kind) for on, ln, kind in half]
            if len(rel) != len(cells) or not cells:
                continue
            # rules (the "dumb" part): stay on the neck, no jump over an octave, not too many chords
            if any(abs(a - b) > 12 for a, b in zip(rel, rel[1:])):
                continue
            if sum(1 for c in cells if c[2] == 'c') > 8:
                continue
            toks, ok, t = [], True, 0
            for (on, ln, kind), d in zip(cells, rel):
                on, ln = on * Q, ln * Q
                if on > t:
                    toks.append(f'wait:{on - t}'); t = on
                p = root + d
                pos = self._place(p, ['s6', 's5', 's4', 's3'], GBASE, {'s6': -2})
                if pos is None:
                    ok = False; break
                toks.append(f'distorted0:note:{pos[0]}:f{pos[1]}')
                if kind == 'p':
                    toks.append('nfx:palm_mute')
                elif kind == 'x':
                    toks.append('nfx:dead')
                elif kind == 'c':
                    for iv in self.rng.choice(self.shape)[1:]:
                        up = self._place(p + iv, ['s5', 's4', 's3', 's2'], GBASE, {})
                        if up and up[0] != pos[0]:
                            toks.append(f'distorted0:note:{up[0]}:f{up[1]}')
                bp = self._place(p - 12, ['s5', 's4', 's3'], BBASE, {'s5': -2})   # bass doubles the low note
                if bp:
                    toks.append(f'bass:note:{bp[0]}:f{bp[1]}')
                    if kind in 'px':
                        toks.append('nfx:palm_mute')
            if not ok:
                continue
            if t < BAR:
                toks.append(f'wait:{BAR - t}')
            return toks
        return ['distorted0:note:s6:f-2', 'nfx:palm_mute', 'bass:note:s5:f-2', f'wait:{BAR}']

    def phrase(self, n_bars, forms, root_plan):
        """Bars following a riff form; each new letter takes the next planned root."""
        r = self.rng
        form = None
        if forms:
            fs = forms.get(n_bars)
            if fs:
                form = r.choices([f for f, _ in fs], weights=[p for _, p in fs])[0]
        form = form or 'ABCD'[:n_bars]
        made, out, i = {}, [], 0
        for ch in form:
            if ch not in made:
                made[ch] = self.bar(root_plan[i] if i < len(root_plan) else 0)
                i += 1
            out += ['new_measure'] + made[ch]
        return out, form


def load(path=None):
    p = Path(path) if path else DATA / 'prefabs.json'
    return json.loads(p.read_text()) if p.exists() else None


if __name__ == '__main__':
    if len(sys.argv) >= 2 and sys.argv[1] == 'mine':
        src = sys.argv[2] if len(sys.argv) > 2 else str(DATA / 'tokens')
        out = Path(sys.argv[3]) if len(sys.argv) > 3 else DATA / 'prefabs.json'
        m = mine(src)
        out.write_text(json.dumps(m))
        show(m)
        print(f'\nsaved {out}')
    else:
        print(__doc__)
