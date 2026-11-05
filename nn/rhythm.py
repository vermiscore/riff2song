"""Change the rhythm of some bars of a .gp5, keeping the notes.

    python3 nn/rhythm.py song.gp5 1 332            # bar 1 -> dotted 4th, dotted 4th, 4th
    python3 nn/rhythm.py song.gp5 1-4 3,3,2        # bars 1 to 4
    python3 nn/rhythm.py song.gp5 12 2222 --unit 16     # lengths counted in 16ths
    python3 nn/rhythm.py song.gp5 5 3,r1,2,2       # r = a rest of that length

The pattern lists note lengths in 8th notes (or --unit 16 / 4); they must fill the bar (8 eighths in 4/4).
Use commas when a length is 10 or more. Each guitar/bass track keeps its own notes of that bar, in order
(tied notes count once); if the pattern has more notes than the bar, the last note is repeated.
Drums are not touched. Writes song_rhythm.gp5 next to the input (use --out to choose, --in-place to overwrite).
"""
import argparse
import sys
from pathlib import Path

import guitarpro as gp

QUARTER = 960
LEN = [(3840, 1, False), (2880, 2, True), (1920, 2, False), (1440, 4, True), (960, 4, False),
       (720, 8, True), (480, 8, False), (360, 16, True), (240, 16, False), (120, 32, False)]


def parse_pattern(text, unit):
    parts = text.split(',') if ',' in text else list(text.replace('r', ' r').split()) if ' ' in text else None
    if parts is None:                                     # plain digits like 332 or 3r12
        parts, cur = [], ''
        for ch in text:
            if ch == 'r':
                cur = 'r'
            else:
                parts.append(cur + ch); cur = ''
    tick = QUARTER * 4 // unit
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        rest = p.startswith('r')
        out.append((int(p.lstrip('r')) * tick, rest))
    return out


def split(d):
    out = []
    while d > 0:
        for t, v, dot in LEN:
            if t <= d:
                out.append((v, dot)); d -= t; break
        else:
            sys.exit(f'length {d} ticks cannot be written')
    return out


def notes_of(measure):
    """The sounding note groups of a bar in order: [[(string, fret, velocity, palm_mute, dead), ...], ...]."""
    groups = []
    for b in measure.voices[0].beats:
        real = [n for n in b.notes if n.type != gp.NoteType.tie]
        if real:
            groups.append([(n.string, n.value, n.velocity, n.effect.palmMute, n.type == gp.NoteType.dead)
                           for n in real])
    return groups


def rewrite(measure, pattern):
    groups = notes_of(measure)
    if not groups:
        return False
    v = gp.Voice(measure)
    gi = 0
    for d, rest in pattern:
        g = None if rest else groups[min(gi, len(groups) - 1)]
        if not rest:
            gi += 1
        for k, (val, dot) in enumerate(split(d)):
            b = gp.Beat(v, duration=gp.Duration(value=val, isDotted=dot))
            if g:
                for s, f, vel, pm, dead in g:
                    t = gp.NoteType.tie if k else (gp.NoteType.dead if dead else gp.NoteType.normal)
                    n = gp.Note(b, value=f, string=s, velocity=vel, type=t)
                    n.effect.palmMute = pm
                    b.notes.append(n)
                b.status = gp.BeatStatus.normal
            else:
                b.status = gp.BeatStatus.rest
            v.beats.append(b)
    measure.voices = [v, gp.Voice(measure)]
    return True


def bar_range(text, n):
    a, _, b = text.partition('-')
    a, b = int(a), int(b or a)
    if not (1 <= a <= b <= n):
        sys.exit(f'bars must be within 1-{n}')
    return range(a - 1, b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gp5')
    ap.add_argument('bars', help='bar number or range, e.g. 1 or 1-4')
    ap.add_argument('pattern', help='note lengths, e.g. 332 or 3,3,2 or 3,r1,2,2')
    ap.add_argument('--unit', type=int, default=8, choices=[4, 8, 16], help='lengths counted in 4ths, 8ths or 16ths')
    ap.add_argument('--tracks', default='', help='only these track numbers (1 = first), e.g. 1,3; default all but drums')
    ap.add_argument('--out')
    ap.add_argument('--in-place', action='store_true')
    a = ap.parse_args()

    src = Path(a.gp5)
    song = gp.parse(str(src))
    pat = parse_pattern(a.pattern, a.unit)
    want = {int(x) for x in a.tracks.split(',') if x}
    changed = []
    for bi in bar_range(a.bars, len(song.measureHeaders)):
        h = song.measureHeaders[bi]
        bar_len = h.timeSignature.numerator * QUARTER * 4 // h.timeSignature.denominator.value
        total = sum(d for d, _ in pat)
        if total != bar_len:
            sys.exit(f'pattern is {total // (QUARTER * 4 // a.unit)} {a.unit}ths but bar {bi + 1} needs '
                     f'{bar_len // (QUARTER * 4 // a.unit)}')
        for ti, t in enumerate(song.tracks, 1):
            if t.isPercussionTrack or (want and ti not in want):
                continue
            if rewrite(t.measures[bi], pat):
                changed.append(f'{t.name} bar {bi + 1}')
    out = src if a.in_place else Path(a.out) if a.out else src.with_name(src.stem + '_rhythm.gp5')
    gp.write(song, str(out))
    print('changed: ' + (', '.join(changed) or 'nothing (no notes in those bars)'))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
