"""Riffs built from riffs extracted from the reference songs.

    python3 nn/riffs.py mine        # nn/data/tokens -> nn/data/riffs.json (2-bar guitar riffs)
    python3 nn/riffs.py list

A riff = list of events [onset, dur, rel_pitches, palm_mute] over 2 bars (7680 ticks), pitches relative
to the riff pedal (0 = the low open string). Three generators:

A  rhythm_first : a reference riff's rhythm, pedal hits and chord shapes; the other notes are chosen
                  from one song's note-choice palette (Markov chain over intervals)
B  motif        : a half-bar core from a reference riff, built as  X X' X Y  (X' = core moved by a
                  typical move of that song, Y = an ending half-bar from another riff of that song)
C  mutate       : a reference riff with its intervals remapped to another song's palette, and
                  sometimes its 2nd bar replaced by the 2nd bar of a riff with a similar note count
"""
import json, math, random, collections, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
OUT = DATA / 'riffs.json'
BASE = {'s1': 64, 's2': 59, 's3': 55, 's4': 50, 's5': 45, 's6': 40, 's7': 35}
BAR, HALF = 3840, 1920
INSTS = ('distorted0', 'distorted1', 'distorted2', 'clean0')


# ---------------------------------------------------------------- extraction
def parse_bars(tokens, inst):
    """bars = [(length, [[onset, pitches, pm, tie]])]; rests have pitches=[]."""
    bars, cur, t, last = [], None, 0, None
    for x in tokens:
        if x == 'new_measure':
            if cur is not None:
                bars.append((t, cur))
            cur, t, last = [], 0, None
            continue
        if cur is None:
            continue
        if x.startswith(inst + ':note'):
            _, _, s, f = x.split(':')
            if s not in BASE:
                continue
            p = BASE[s] + int(f[1:])
            if cur and cur[-1][0] == t and last == inst:
                cur[-1][1].append(p)
            else:
                cur.append([t, [p], False, False])
            last = inst
        elif x == inst + ':rest':
            cur.append([t, [], False, False]); last = inst
        elif x.startswith('nfx:') and last == inst and cur:
            if x == 'nfx:palm_mute':
                cur[-1][2] = True
            elif x == 'nfx:tie':
                cur[-1][3] = True
        elif x.startswith('wait:'):
            t += int(x[5:]); last = None
        elif ':' in x:
            last = None
    if cur is not None:
        bars.append((t, cur))
    return bars


def to_events(bar):
    length, ev = bar
    kept = []
    for on, ps, pm, tie in ev:
        if tie and kept and kept[-1][2]:
            continue                                  # a tied note = sustain of the previous one
        kept.append([on, 0, sorted(set(ps)), pm])
    for i in range(len(kept)):
        kept[i][1] = (kept[i + 1][0] if i + 1 < len(kept) else length) - kept[i][0]
    return kept


def _riffs(song, bars, off):
    out = []
    for i in range(off, len(bars) - 1, 2):
        b1, b2 = bars[i], bars[i + 1]
        if b1[0] != BAR or b2[0] != BAR:
            continue
        ev = to_events(b1) + [[on + BAR, d, ps, pm] for on, d, ps, pm in to_events(b2)]
        notes = [e for e in ev if e[2]]
        if len(notes) < 6:
            continue
        lows = [e[2][0] for e in notes]
        ped = collections.Counter(lows).most_common(1)[0][0]
        if lows.count(ped) / len(lows) < 0.2:
            continue                                  # not a pedal riff (lead line etc.)
        rel = [[on, d, [p - ped for p in ps], pm] for on, d, ps, pm in ev]
        if any(p < -1 or p > 24 for e in rel for p in e[2]):
            continue
        out.append(dict(song=song, bar=i + 1, ev=rel))
    seen, uniq = {}, []
    for r in out:
        k = json.dumps(r['ev'])
        if k in seen:
            seen[k]['count'] += 1
        else:
            r['count'] = 1; seen[k] = r; uniq.append(r)
    return uniq


def riffs_from_song(path):
    toks = Path(path).read_text().split()
    song = Path(path).stem
    best = []
    for inst in INSTS:
        bars = parse_bars(toks, inst)
        for off in (0, 1):
            r = _riffs(song, bars, off)
            if len(r) > len(best):
                best = r
    return best


def mine(tok_dir=DATA / 'tokens', out=OUT):
    allr = []
    for f in sorted(Path(tok_dir).glob('*.txt')):
        r = riffs_from_song(f)
        allr += r
        print(f'{len(r):4d}  {f.stem[:60]}')
    json.dump(allr, open(out, 'w'))
    print(f'{len(allr)} riffs from {len(set(r["song"] for r in allr))} songs -> {out}')


# ---------------------------------------------------------------- generators
def low(e):
    return e[2][0] if e[2] else None


def pick(counter, rng):
    items = list(counter.items())
    x = rng.random() * sum(w for _, w in items)
    for k, w in items:
        x -= w
        if x <= 0:
            return k
    return items[-1][0]


DARK, BRIGHT = {1, 3, 6, 8, 10}, {4, 9, 11}          # b2 b3 b5 b6 b7  vs  3 6 7


def darkness(ev):
    """+1 per dark non-pedal note (b2 b3 b5 b6 b7), -1 per bright one (3 6 7), averaged over non-pedal notes."""
    pcs = [e[2][0] % 12 for e in ev if e[2] and e[2][0] % 12]
    if not pcs:
        return 0.0
    return sum((p in DARK) - (p in BRIGHT) for p in pcs) / len(pcs)


def is_lead(ev):
    """A mostly high single-note line (lead / melody) rather than a riff."""
    notes = [e for e in ev if e[2]]
    if not notes:
        return True
    ped = sum(1 for e in notes if e[2][0] == 0) / len(notes)
    high = sum(1 for e in notes if e[2][0] >= 12) / len(notes)
    return ped < 0.25 and high > 0.4


def rock_cadence(ev):
    """Power chords whose roots only move among 1, 4, 5 (and use both 4 and 5) = I-IV-V."""
    roots = {e[2][0] % 12 for e in ev if len(e[2]) > 1}
    return len(roots) >= 2 and roots <= {0, 5, 7} and {5, 7} <= roots


def holes(ev):
    return sum(d for on, d, ps, pm in ev if not ps)


def straight(ev):
    """No triplets / odd tuplets (everything on the 32nd grid)."""
    return not any(on % 120 or d % 120 for on, d, ps, pm in ev)


def clean(ev, max_fill=480, top=24):
    """Tidy a riff: a rest of an 8th or less right after a note becomes part of that note (let it ring);
    notes more than two octaves above the pedal come down an octave; repeated 32nds of the same notes
    become 16ths."""
    out = []
    for on, d, ps, pm in ev:
        ps = sorted({p - 12 * max(0, (p - top + 11) // 12) for p in ps})
        if not ps and out and out[-1][2] and d <= max_fill:
            out[-1][1] += d; continue
        if (out and ps and out[-1][2] == ps and out[-1][1] == 120 and d == 120 and out[-1][3] == pm
                and out[-1][0] % 240 == 0):
            out[-1][1] = 240; continue
        out.append([on, d, ps, pm])
    return out


def chord_like(src, root):
    return [root + (p - src[0]) for p in src] if src else []


def cut(ev, a, b):
    """Events inside [a, b) shifted to start at 0 (a note sounding at a is clipped in)."""
    out = []
    for on, d, ps, pm in ev:
        end = on + d
        if end <= a or on >= b:
            continue
        s, e = max(on, a), min(end, b)
        out.append([s - a, e - s, list(ps), pm])
    if not out or out[0][0] > 0:
        out.insert(0, [0, out[0][0] if out else b - a, [], False])
    return out


def shift(ev, k):
    return [[on + k, d, list(ps), pm] for on, d, ps, pm in ev]


class Riffs:
    def __init__(self, path=OUT, song=None, rng=None, dark=2.0, exclude=(), balance=True, learn=True):
        self.dark = dark
        self.balance = balance
        self.song_w = {}
        self.all = [r for r in json.load(open(path))
                    if not any(x.lower().replace(' ', '') in r['song'].lower().replace(' ', '') for x in exclude)]
        self.by = collections.defaultdict(list)
        for r in self.all:
            self.by[r['song']].append(r)
        self.rng = rng or random.Random()
        if learn:
            self.song_w = rated_song_weights(self.by)
        self.song = self.find(song) if song else None     # the song palette all riffs follow
        if self.song is None:
            self.song = self.rng.choice(self.big_songs())
        self._pal = {}

    def big_songs(self):
        return [s for s in self.by if len(self.by[s]) >= 8] or list(self.by)

    def find(self, name):
        if not name:
            return None
        n = name.lower().replace(' ', '')
        for s in self.by:
            if n in s.lower().replace(' ', ''):
                return s
        return None

    def palette(self, song):
        if song not in self._pal:
            first, trans = collections.Counter(), collections.defaultdict(collections.Counter)
            for r in self.by[song]:
                prev = None
                for e in r['ev']:
                    p = low(e)
                    if p is None or p % 12 == 0:
                        continue
                    (first if prev is None else trans[prev])[p] += r['count']
                    prev = p
            self._pal[song] = (first, trans)
        return self._pal[song]

    def chord_palette(self, song):
        """Root moves of the power chords actually used in the song: prev root -> next root."""
        key = ('chord', song)
        if key not in self._pal:
            trans = collections.defaultdict(collections.Counter)
            for r in self.by[song]:
                prev = None
                for e in r['ev']:
                    if len(e[2]) > 1:
                        p = e[2][0]
                        if prev is not None:
                            trans[prev][p] += r['count']
                        prev = p
            self._pal[key] = trans
        return self._pal[key]

    def choose(self, song=None, min_notes=8, max_notes=99):
        pool = [r for r in (self.by[song] if song else self.all)
                if min_notes <= sum(1 for e in r['ev'] if e[2]) <= max_notes
                and all(sum(1 for e in r['ev'] if e[2] and h * BAR <= e[0] < (h + 1) * BAR) >= 3 for h in (0, 1))
                and any(low(e) and low(e) % 12 for e in r['ev'])]
        pool = [r for r in pool if not is_lead(r['ev']) and straight(r['ev']) and holes(r['ev']) <= 960] or pool
        if not pool:
            pool = self.all
        if self.balance and not song:
            # pick a song first (all songs equal, times its weight from your ratings), then a riff of it
            songs = sorted({r['song'] for r in pool})
            sw = [self.song_w.get(s, 1.0) for s in songs]
            s = self.rng.choices(songs, weights=sw)[0]
            pool = [r for r in pool if r['song'] == s]
            w = [math.exp(self.dark * darkness(r['ev'])) for r in pool]
        else:
            w = [r['count'] * math.exp(self.dark * darkness(r['ev'])) for r in pool]
        return self.rng.choices(pool, weights=w)[0]

    def refill(self, ev, song):
        first, trans = self.palette(song)
        if not first:
            return [list(e) for e in ev]
        ctrans = self.chord_palette(song)
        out, prev, cprev = [], None, None
        for on, d, ps, pm in ev:
            if ps and len(ps) > 1:                         # power chord: follow the song's chord-root moves
                if ps[0] % 12 == 0 and cprev is None:
                    out.append([on, d, list(ps), pm]); cprev = 0; continue
                nxt = ctrans.get(cprev if cprev is not None else 0)
                if nxt:
                    p = pick(nxt, self.rng)
                    out.append([on, d, chord_like(ps, p), pm]); cprev = p; continue
            if not ps or ps[0] % 12 == 0:
                out.append([on, d, list(ps), pm]); continue
            nxt = trans.get(prev) if prev is not None else None
            p = pick(nxt if nxt else first, self.rng)
            out.append([on, d, chord_like(ps, p), pm]); prev = p
            if len(ps) > 1:
                cprev = p
        return out

    def moves(self, song):
        _, trans = self.palette(song)
        m = collections.Counter()
        for a, c in trans.items():
            for b, w in c.items():
                if a != b and abs(b - a) <= 7:
                    m[b - a] += w
        return m

    def move(self, ev, song):
        """Move the non-pedal notes by one of the song's typical moves."""
        mv = self.moves(song)
        for _ in range(10):
            k = pick(mv, self.rng) if mv else self.rng.choice([1, -1, 2, 3])
            new = [[on, d, [p + k for p in ps] if ps and ps[0] % 12 else list(ps), pm] for on, d, ps, pm in ev]
            if all(-1 <= ps[0] <= 19 and (ps[0] % 12 or not o[2] or o[2][0] % 12 == 0)
                   for (on, d, ps, pm), o in zip(new, ev) if ps):
                return new, k
        return [list(e) for e in ev], 0

    def remap(self, ev, from_song, to_song):
        def ranks(song):
            c = collections.Counter()
            for r in self.by[song]:
                for e in r['ev']:
                    p = low(e)
                    if p is not None and p % 12:
                        c[p % 12] += r['count']
            order = [p for p, _ in c.most_common()]
            return order
        a, b = ranks(from_song), ranks(to_song)
        b = b + [p for p in (1, 3, 8, 6, 10, 2, 5, 7) if p not in b]   # unused -> dark intervals first, never 3 6 7
        a = a + [p for p in range(1, 12) if p not in a]
        m = dict(zip(a, b))
        out = []
        for on, d, ps, pm in ev:
            if ps and ps[0] % 12:
                pc = ps[0] % 12
                out.append([on, d, chord_like(ps, ps[0] - pc + m.get(pc, pc)), pm])
            else:
                out.append([on, d, list(ps), pm])
        return out

    # --- the three generators; each returns (events, description)
    def A(self, **kw):
        src = self.choose(**kw)
        ev = self.refill(src['ev'], self.song)
        return ev, f"A rhythm {short(src['song'])} b{src['bar']}"

    def B(self, **kw):
        for _ in range(50):
            src = self.choose(**kw)
            half = self.rng.choice([0, 1, 2])
            core = cut(src['ev'], half * HALF, (half + 1) * HALF)
            if sum(1 for e in core if e[2]) >= 3 and any(low(e) and low(e) % 12 for e in core):
                break
        core = self.refill(core, self.song)
        var, k = self.move(core, self.song)
        end_src = self.choose(song=self.song)
        end = self.refill(cut(end_src['ev'], 3 * HALF, 4 * HALF), self.song)
        ev = core + shift(var, HALF) + shift(core, 2 * HALF) + shift(end, 3 * HALF)
        return ev, f"B core {short(src['song'])} b{src['bar']} {k:+d}"

    def C(self, **kw):
        src = self.choose(**kw)
        ev = [list(e) for e in src['ev']]
        desc = f"C {short(src['song'])} b{src['bar']}"
        if self.rng.random() < 0.5:
            d = sum(1 for e in cut(ev, BAR, 2 * BAR) if e[2])
            cands = [r for r in self.all if r is not src
                     and abs(sum(1 for e in cut(r['ev'], BAR, 2 * BAR) if e[2]) - d) <= 3]
            if cands:
                o = self.rng.choice(cands)
                ev = cut(ev, 0, BAR) + shift(cut(o['ev'], BAR, 2 * BAR), BAR)
                desc += f"+bar2 {short(o['song'])}"
        to = self.song if self.song != src['song'] else self.rng.choice(self.big_songs())
        return self.remap(ev, src['song'], to), desc

    def make(self, method, role=None):
        kw = {}
        if role == 'BD':
            kw = dict(min_notes=4, max_notes=14)      # breakdowns: fewer, heavier hits
        best = None
        for _ in range(6):
            ev, d = getattr(self, method)(**kw)
            ev = clean(ev)
            sc = darkness(ev) - holes(ev) / 1920 - (5 if rock_cadence(ev) else 0) - (1 if is_lead(ev) else 0)
            if best is None or sc > best[0]:
                best = (sc, ev, d)
        return best[1], best[2]

    def vary_end(self, ev):
        """Second time around: change only the last half-bar (move it, or a new ending)."""
        head = cut(ev, 0, 3 * HALF)
        tail = cut(ev, 3 * HALF, 4 * HALF)
        if self.rng.random() < 0.5:
            tail, _ = self.move(tail, self.song)
        else:
            tail = self.refill(cut(self.choose(song=self.song)['ev'], 3 * HALF, 4 * HALF), self.song)
        return head + shift(tail, 3 * HALF)


def rated_song_weights(by, strength=1.0, prior=5):
    """Weight per reference song from your ratings of riffs built from it (riff_sample rows of
    taste/rate.py): exp(strength * (song mean - overall mean)), the mean pulled toward the overall
    mean when a song has few ratings."""
    import csv, re
    path = HERE.parent / 'outputs' / 'ratings.csv'
    if not path.exists():
        return {}
    by_short = {}
    for s in by:
        by_short.setdefault(short(s), s)
    got = collections.defaultdict(list)
    for r in csv.DictReader(open(path)):
        if r.get('type') != 'riff_sample' or not r.get('rating', '').strip():
            continue
        m = re.match(r'(?:A rhythm|B core|C) (.+?) b\d+', r.get('note', ''))
        if m and m.group(1) in by_short:
            got[by_short[m.group(1)]].append(float(r['rating']))
    allv = [v for vs in got.values() for v in vs]
    if not allv:
        return {}
    mu = sum(allv) / len(allv)
    return {s: math.exp(strength * ((sum(vs) + prior * mu) / (len(vs) + prior) - mu)) for s, vs in got.items()}


def short(song):
    p = song.split('-')
    return (p[1] if len(p) > 1 else song)[:18]


# ---------------------------------------------------------------- to tokens
LENGTHS = [3840, 2880, 1920, 1440, 960, 720, 480, 360, 240, 120]


def place(rel):
    """Rel pitches (0 = low open string) -> [(string, token fret)] (token space: s6 f-2 = open drop string,
    the other strings are 7, 12, 17, 21, 26 semitones above it with plain fret numbers)."""
    strings = [('s6', 0), ('s5', 7), ('s4', 12), ('s3', 17), ('s2', 21), ('s1', 26)]
    out, si = [], 0
    for p in sorted(set(rel)):
        while si < len(strings):
            s, o = strings[si]; si += 1
            f = p - 2 if s == 's6' else p - o
            if (s == 's6' and -2 <= f <= 13) or (s != 's6' and 0 <= f <= 15):
                out.append((s, f)); break
    return out


def bass_place(p):
    p = p % 12 if p > 14 else p
    return ('s5', p - 2) if p <= 12 else ('s4', p - 7)


def ev_to_bars(ev, n_bars=2):
    """Events -> list of per-bar token lists (without 'new_measure')."""
    bars = []
    for b in range(n_bars):
        part = cut(ev, b * BAR, (b + 1) * BAR)
        toks = []
        for on, d, ps, pm in part:
            if ps:
                for s, f in place(ps):
                    toks.append(f'distorted0:note:{s}:f{f}')
                    if pm:
                        toks.append('nfx:palm_mute')
                s, f = bass_place(ps[0])
                toks.append(f'bass:note:{s}:f{f}')
                if pm:
                    toks.append('nfx:palm_mute')
            else:
                toks += ['distorted0:rest', 'bass:rest']
            toks.append(f'wait:{d}')
        bars.append(toks)
    return bars


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'mine':
        mine()
    elif len(sys.argv) > 1 and sys.argv[1] == 'list':
        R = Riffs()
        for s, rs in sorted(R.by.items()):
            print(f'{len(rs):4d}  {s}')
    else:
        print(__doc__)


if __name__ == '__main__':
    main()
