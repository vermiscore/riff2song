"""Move the notes of some bars of a .gp5 up or down by semitones (same strings, frets +/- n).

    python3 shift.py picks.gp5 3 +3            # bar 3 up 3 frets (e.g. 12 -> 15)
    python3 shift.py picks.gp5 3-4 -2          # bars 3 to 4 down 2
    python3 shift.py picks.gp5 3 +3 --tracks 1 # only track 1
    python3 shift.py picks.gp5 3 +3 --beat 1   # only the 1st note/chord of the bar

Drums are not touched. Writes picks_shift.gp5 (use --out to choose, --in-place to overwrite).
Also prints the fret limit of each track: if it says 14, the track only allows frets up to 14 in the editor.
"""
import argparse
import sys
from pathlib import Path

import guitarpro as gp


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gp5')
    ap.add_argument('bars', help='bar number or range, e.g. 3 or 3-4')
    ap.add_argument('semitones', type=int)
    ap.add_argument('--tracks', default='', help='only these track numbers (1 = first), e.g. 1,3')
    ap.add_argument('--beat', type=int, default=0, help='only this note/chord of the bar (1 = first)')
    ap.add_argument('--out')
    ap.add_argument('--in-place', action='store_true')
    a = ap.parse_args()

    src = Path(a.gp5)
    song = gp.parse(str(src))
    n = len(song.measureHeaders)
    lo, _, hi = a.bars.partition('-')
    lo, hi = int(lo), int(hi or lo)
    if not 1 <= lo <= hi <= n:
        sys.exit(f'bars must be within 1-{n}')
    want = {int(x) for x in a.tracks.split(',') if x}
    for ti, t in enumerate(song.tracks, 1):
        print(f'track {ti} {t.name}: fret limit {t.fretCount}')
        if t.isPercussionTrack or (want and ti not in want):
            continue
        if t.fretCount < 24:
            t.fretCount = 24                                   # allow high frets in the editor too
        for bi in range(lo - 1, hi):
            k = 0
            for b in t.measures[bi].voices[0].beats:
                if not b.notes:
                    continue
                k += 1
                if a.beat and k != a.beat:
                    continue
                for nt in b.notes:
                    v = nt.value + a.semitones
                    if not 0 <= v <= 24:
                        sys.exit(f'{t.name} bar {bi + 1}: fret {nt.value} would become {v}')
                    nt.value = v
    out = src if a.in_place else Path(a.out) if a.out else src.with_name(src.stem + '_shift.gp5')
    gp.write(song, str(out))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
