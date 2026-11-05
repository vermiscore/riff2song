"""Make many single 2-bar riffs to listen to, and copy the ones you like into your picks file.

    python3 nn/riff_sampler.py --n 20 --src mix              # 20 riffs: half rebuilt from reference riffs, half NN
    python3 nn/riff_sampler.py --n 12 --src riff --palette picks --picks ../tabs_picks/picks.gp5
    python3 nn/riff_sampler.py pick 3 7 12 --to ../tabs_picks/picks.gp5     # newest run; --run to choose

Output (outputs/<time>_riffs/): riff_01.mp3 ... (each riff played 4 times, plain MIDI sound with bass unison and
simple drums), riff_01.mid ... (rate them with taste/rate.py --run <run>; the list is not shown while rating),
samples.gp5 (all riffs, one marker each, an empty bar between them), samples.json, list.txt.
--src riff = rebuilt from the reference riffs (nn/riffs.py: dark, no triplets, no long holes)
--src nn   = written by the NN alone;  mix = both (shuffled, list.txt says which is which)
songs are chosen evenly, then weighted by how you rated riffs built from them (--no-balance / --no-learn)
--palette  = note choice the riffs follow: a reference song name, 'picks' (your picks.gp5), or random per riff
"""
import argparse
import collections
import json
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V1))

import guitarpro as gp                                          # noqa: E402
import mido                                                     # noqa: E402
import riffs as RF                                              # noqa: E402

BAR = 3840
GTR = [62, 57, 53, 48, 43, 36]                                  # drop C, as the generated songs
BASS = [41, 36, 31, 24]
LOW = 36
SF2S = ['/usr/share/sounds/sf2/FluidR3_GM.sf2', '/usr/share/sounds/sf2/default-GM.sf2']
LEN = [(3840, 1, False), (2880, 2, True), (1920, 2, False), (1440, 4, True), (960, 4, False),
       (720, 8, True), (480, 8, False), (360, 16, True), (240, 16, False), (120, 32, False)]
GSTR = {'s6': 2, 's5': 7, 's4': 12, 's3': 17, 's2': 21, 's1': 26}


# ------------------------------------------------------------------ riff makers
def ok_rhythm(ev):
    if not RF.straight(ev):
        return False                                            # triplets / odd tuplets
    if RF.holes(ev) > 960:
        return False                                            # no long holes (more than a beat of silence)
    if len({p for e in ev for p in e[2]}) < 3:
        return False                                            # one or two notes only: too flat
    return all(len({p for e in ev if h * BAR <= e[0] < (h + 1) * BAR for p in e[2]}) >= 2
               for h in (0, 1))                                 # each bar: not empty, not one note only


def rhythm_score(ev):
    notes = [e for e in ev if e[2]]
    if not notes:
        return -9
    six = sum(1 for e in notes if e[1] <= 240) / len(notes)
    mel = [e for e in notes if e[2][0] % 12]
    sync = sum(1 for e in mel if e[0] % 480) / max(1, len(mel))
    return 0.6 * six + 0.8 * sync


def gp_events(path, track=0):
    """Guitar events per bar of a gp5: [[onset, dur, rel_pitches, pm]] (rel to the lowest open string)."""
    s = gp.parse(str(path))
    t = s.tracks[track]
    tun = [x.value for x in t.strings]
    low = min(tun)
    bars = []
    for m in t.measures:
        ev = []
        for b in m.voices[0].beats:
            on, d = b.start - m.start, b.duration.time
            real = [n for n in b.notes if n.type != gp.NoteType.tie]
            if b.notes and not real and ev:
                ev[-1][1] += d; continue
            ev.append([on, d, sorted(tun[n.string - 1] + n.value - low for n in real),
                       any(n.effect.palmMute for n in real)])
        bars.append(ev)
    return bars


def picks_material(path):
    bars = gp_events(path)
    out = []
    for i in range(len(bars) - 1):
        a, b = bars[i], bars[i + 1]
        if any(e[2] for e in a) and any(e[2] for e in b):
            out.append(dict(song='picks', bar=i + 1, count=1,
                            ev=a + [[on + BAR, d, ps, pm] for on, d, ps, pm in b]))
    return out


class Maker:
    def __init__(self, rng, palette=None, picks=None, balance=True, learn=True):
        self.rng = rng
        self.R = RF.Riffs(rng=rng, dark=2.0, balance=balance, learn=learn)
        if self.R.song_w:
            w = sorted(self.R.song_w.items(), key=lambda x: -x[1])
            print('song weights from your ratings:  top ' + ', '.join(f'{RF.short(s)} {v:.2f}' for s, v in w[:3])
                  + '  |  bottom ' + ', '.join(f'{RF.short(s)} {v:.2f}' for s, v in w[-3:]))
        self.R.all = [r for r in self.R.all if ok_rhythm(r['ev'])]
        self.R.by = collections.defaultdict(list)
        for r in self.R.all:
            self.R.by[r['song']].append(r)
        self.fixed = None
        if palette == 'picks':
            if not picks or not Path(picks).exists():
                sys.exit('--palette picks needs --picks path/to/picks.gp5')
            self.R.by['picks'] = picks_material(picks)
            self.fixed = 'picks'
        elif palette:
            self.fixed = self.R.find(palette)
            if self.fixed is None:
                sys.exit(f'no reference song matches {palette!r}')
        self.nn = None
        self.clean = True

    def riff(self):
        R = self.R
        R.song = self.fixed or self.rng.choice(R.big_songs())
        best = None
        for i in range(4):                       # few tries: more tries = the same 'best' songs every time
            m = self.rng.choice('ABC')
            ev, d = R.make(m)                    # make() already tidies (riffs.clean)
            if not ok_rhythm(ev) or RF.rock_cadence(ev):
                continue
            sc = RF.darkness(ev) + rhythm_score(ev)
            if best is None or sc > best[0]:
                best = (sc, ev, f'{d} / palette {RF.short(R.song)}')
        return best[1], best[2]

    def nn_riff(self, a):
        import generate as G
        import torch
        if self.nn is None:
            model, vocab = G.load_model(a.model)
            stoi = {t: i for i, t in enumerate(vocab)}
            prompt = [t for t in ['unknown', 'downtune:-2', 'start'] if t in stoi] or ['start']
            pal = None
            if self.fixed and self.fixed != 'picks':
                import palette as PL
                pals = PL.load()
                key = PL.find(pals, self.fixed) if pals else None
                if key:
                    pal = PL.Palette(pals[key], vocab, G.home_root(G.DATA_TOKENS), 1.0, base=PL.base(pals))
            self.nn = (G, model, vocab, stoi, prompt, G.build_masks(vocab, {'distorted0', 'bass'}), pal)
        G, model, vocab, stoi, prompt, masks, pal = self.nn
        best = None
        for _ in range(10):
            ids = [stoi[t] for t in prompt]
            toks = G.sample_riff(model, ids, 2, vocab, stoi, masks, a.temperature, 40, self.rng, 0.8, palette=pal)
            ev = tokens_to_events(toks)
            if self.clean:
                ev = RF.clean(ev)
            if not ok_rhythm(ev):
                continue
            sc = RF.darkness(ev) + rhythm_score(ev)
            if best is None or sc > best[0]:
                best = (sc, ev)
        if best is None:                                  # the NN kept writing holes: use a rebuilt riff instead
            ev, d = self.riff()
            return ev, d + ' (NN failed)'
        return best[1], f'NN (temperature {a.temperature:g})'


def tokens_to_events(toks):
    """NN tokens (2 bars) -> events; waits that do not fit the 120-tick grid are snapped."""
    ev, cur, pm, t = [], [], False, 0
    bar_start = -BAR
    for x in toks:
        if x == 'new_measure':
            bar_start += BAR; t = bar_start
        elif x.startswith('distorted0:note:'):
            _, _, s, f = x.split(':')
            if s in GSTR:
                cur.append(int(f[1:]) + GSTR[s])
        elif x == 'nfx:palm_mute' and cur:
            pm = True
        elif x.startswith('wait:'):
            d = max(240, round(int(x[5:]) / 240) * 240)        # 16th grid: NN timing wobble is not rhythm
            if cur or not ev or ev[-1][2]:
                ev.append([t, d, sorted(set(cur)), pm])
            else:
                ev[-1][1] += d
            t += d; cur, pm = [], False
    # durations = gap to the next onset (snapping can leave gaps or overlaps), inside 2 bars
    ev = [e for e in sorted(ev, key=lambda e: e[0]) if e[0] < 2 * BAR]
    seen, out = set(), []
    for e in ev:
        if e[0] not in seen:
            seen.add(e[0]); out.append(e)
    if not out or out[0][0] > 0:
        out.insert(0, [0, 0, [], False])
    for i, e in enumerate(out):
        e[1] = (out[i + 1][0] if i + 1 < len(out) else 2 * BAR) - e[0]
    # a note may not cross the bar line in the gp5 writer: split it
    res = []
    for on, d, ps, p in out:
        if on < BAR < on + d:
            res += [[on, BAR - on, ps, p], [BAR, on + d - BAR, ps, p]]
        else:
            res.append([on, d, ps, p])
    return res


# ------------------------------------------------------------------ audio
def write_midi(ev, path, bpm, reps=4):
    mid = mido.MidiFile(ticks_per_beat=960)
    out = {0: [], 1: [], 9: []}
    for r in range(reps):
        o = r * 2 * BAR
        for on, d, ps, pm in ev:
            if not ps:
                continue
            s, L = o + on, max(60, int(d * (0.6 if pm else 0.95)))
            for p in ps:
                out[0].append((s, s + L, LOW + p, 80 if pm else 105))
            out[1].append((s, s + L, LOW - 12 + ps[0], 100))
            if ps[0] % 12 == 0 or d >= 480:
                out[9].append((s, s + 60, 36, 110))
        for b in range(2):
            bb = o + b * BAR
            out[9].append((bb + 1920, bb + 1980, 38, 120))
            for q in range(4):
                out[9].append((bb + q * 960, bb + q * 960 + 60, 51, 75))
        if r == 0:
            out[9].append((o, o + 60, 49, 120))
    for ch, prog, nm in ((0, 30, 'Guitar'), (1, 33, 'Bass'), (9, None, 'Drums')):
        t = mido.MidiTrack(); mid.tracks.append(t)
        t.append(mido.MetaMessage('track_name', name=nm))          # taste/features.py finds tracks by name
        if ch == 0:
            t.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(bpm)))
        if prog is not None:
            t.append(mido.Message('program_change', channel=ch, program=prog))
        msgs = []
        for s, e, n, v in out[ch]:
            msgs += [(s, 1, mido.Message('note_on', channel=ch, note=n, velocity=v)),
                     (e, 0, mido.Message('note_off', channel=ch, note=n, velocity=0))]
        now = 0
        for s, _, m in sorted(msgs, key=lambda x: (x[0], x[1])):
            m.time = s - now; now = s; t.append(m)
    mid.save(str(path))


def to_mp3(mid, mp3, sf2):
    wav = mp3.with_suffix('.wav')
    subprocess.run(['fluidsynth', '-ni', '-g', '0.8', '-F', str(wav), '-r', '44100', sf2, str(mid)],
                   check=True, capture_output=True)
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(wav), '-b:a', '192k', str(mp3)], check=True)
    wav.unlink()


# ------------------------------------------------------------------ gp5
def split(d):
    out = []
    while d > 0:
        for t, v, dot in LEN:
            if t <= d:
                out.append((v, dot)); d -= t; break
        else:
            break
    return out


def place(pitches, tuning, max_fret=20):
    out, si = [], len(tuning) - 1
    for p in sorted(set(pitches)):
        while si >= 0:
            f = p - tuning[si]; si -= 1
            if 0 <= f <= max_fret:
                out.append((si + 2, f)); break
    return out


def fill_measure(m, ev, tuning, mono=False, octave=0):
    v = gp.Voice(m)
    for on, d, ps, pm in ev:
        pitches = [ps[0]] if (mono and ps) else ps
        for k, (val, dot) in enumerate(split(d)):
            b = gp.Beat(v, duration=gp.Duration(value=val, isDotted=dot))
            if pitches:
                for s, f in place([LOW + p + octave for p in pitches], tuning):
                    n = gp.Note(b, value=f, string=s, velocity=95, type=gp.NoteType.tie if k else gp.NoteType.normal)
                    n.effect.palmMute = pm
                    b.notes.append(n)
                b.status = gp.BeatStatus.normal
            else:
                b.status = gp.BeatStatus.rest
            v.beats.append(b)
    m.voices = [v, gp.Voice(m)]


def rest_measure(m):
    v = gp.Voice(m)
    b = gp.Beat(v, duration=gp.Duration(value=1)); b.status = gp.BeatStatus.rest
    v.beats.append(b)
    m.voices = [v, gp.Voice(m)]


def add_bar(song):
    prev = song.measureHeaders[-1] if song.measureHeaders else None
    song.newMeasure()
    h = song.measureHeaders[-1]
    h.number = len(song.measureHeaders)
    if prev is not None:
        h.start = prev.start + prev.length
        h.timeSignature.numerator = prev.timeSignature.numerator
        h.timeSignature.denominator.value = prev.timeSignature.denominator.value
    return len(song.measureHeaders) - 1


def append_riffs(song, riffs, gap_first=True):
    """riffs = [(name, events)]: each = (empty bar) + 2 bars with a marker, guitar + bass (bass = lowest note)."""
    g = song.tracks[0]
    bass = next((t for t in song.tracks if 'bass' in t.name.lower()), None)
    for name, ev in riffs:
        if gap_first and song.measureHeaders:
            bi = add_bar(song)
            for t in song.tracks:
                rest_measure(t.measures[bi])
        gap_first = True
        for half in range(2):
            bi = add_bar(song)
            part = RF.cut(ev, half * BAR, (half + 1) * BAR)
            for t in song.tracks:
                if t is g:
                    fill_measure(t.measures[bi], part, [s.value for s in g.strings])
                elif t is bass:
                    fill_measure(t.measures[bi], part, [s.value for s in t.strings], mono=True, octave=-12)
                else:
                    rest_measure(t.measures[bi])
            if half == 0:
                song.measureHeaders[bi].marker = gp.Marker(title=name, color=gp.Color(200, 0, 0))


def new_song(bpm):
    song = gp.Song()
    song.tempo = bpm
    song.measureHeaders = []
    g = song.tracks[0]
    g.name = 'Guitar'
    g.strings = [gp.GuitarString(number=i + 1, value=v) for i, v in enumerate(GTR)]
    g.channel.instrument = 30
    g.measures = []
    b = gp.Track(song, number=2, name='Bass')
    b.strings = [gp.GuitarString(number=i + 1, value=v) for i, v in enumerate(BASS)]
    b.channel = gp.MidiChannel(channel=2, effectChannel=3, instrument=33)
    b.measures = []
    song.tracks.append(b)
    return song


# ------------------------------------------------------------------ commands
def make(a):
    rng = random.Random(a.seed)
    maker = Maker(rng, a.palette, a.picks, balance=not a.no_balance, learn=not a.no_learn)
    maker.clean = not a.no_clean
    run_id = time.strftime('%Y%m%d-%H%M%S') + '_riffs'
    out = V1 / 'outputs' / run_id
    out.mkdir(parents=True, exist_ok=True)
    sf2 = next((p for p in SF2S if Path(p).exists()), None)
    kinds = (['riff'] * a.n if a.src == 'riff' else ['nn'] * a.n if a.src == 'nn'
             else ['riff'] * (a.n - a.n // 2) + ['nn'] * (a.n // 2))
    rng.shuffle(kinds)
    items, lines = [], []
    for k, kind in enumerate(kinds, 1):
        ev, desc = maker.riff() if kind == 'riff' else maker.nn_riff(a)
        name = f'riff_{k:02d}'
        write_midi(ev, out / f'{name}.mid', a.tempo)
        bars = RF.ev_to_bars(ev)
        toks = ['start'] + [t for _ in range(2) for b in bars for t in ['new_measure'] + b] + ['end']
        (out / f'{name}.tokens.txt').write_text('\n'.join(toks) + '\n')     # for nn/finetune.py
        if sf2:
            to_mp3(out / f'{name}.mid', out / f'{name}.mp3', sf2)
        items.append(dict(name=name, kind=kind, desc=desc, ev=ev))
        lines.append(f'{name}  {kind:4s}  {desc}')
        print(lines[-1])
    song = new_song(a.tempo)
    append_riffs(song, [(it['name'], it['ev']) for it in items])
    gp.write(song, str(out / 'samples.gp5'))
    json.dump(items, open(out / 'samples.json', 'w'))
    (out / 'list.txt').write_text('\n'.join(lines) + '\n')
    from taste.common import read_ratings, write_ratings
    write_ratings(read_ratings() + [dict(run_id=run_id, candidate=it['name'],
                                         type='riff_sample' if it['kind'] == 'riff' else 'nn_riff_sample',
                                         seed='', rating='', note=it['desc']) for it in items])
    print(f'\n{len(items)} riffs in {out}\n'
          f'rate them:  python3 {V1 / "taste" / "rate.py"} --run {run_id}\n'
          f'keep some: python3 {HERE / "riff_sampler.py"} pick 3 7 --to ../tabs_picks/picks.gp5')


def pick(a):
    runs = sorted(p for p in (V1 / 'outputs').glob('*_riffs') if (p / 'samples.json').exists())
    run = (V1 / 'outputs' / a.run) if a.run else (runs[-1] if runs else None)
    if run is None or not (run / 'samples.json').exists():
        sys.exit('no riff_sampler run found')
    items = {i + 1: it for i, it in enumerate(json.load(open(run / 'samples.json')))}
    to = Path(a.to)
    song = gp.parse(str(to)) if to.exists() else new_song(150)
    # the last bar that has notes in track 1: new riffs go after it (empty bars after it are reused)
    last = max((i for i, m in enumerate(song.tracks[0].measures)
                if any(b.notes for b in m.voices[0].beats)), default=-1)
    del song.measureHeaders[last + 1:]
    for t in song.tracks:
        del t.measures[last + 1:]
    stamp = run.name.split('_')[0][4:]                           # MMDD-HHMMSS
    chosen = []
    for n in a.numbers:
        if n not in items:
            sys.exit(f'no riff {n} in {run.name} (1-{len(items)})')
        chosen.append((f'{a.prefix}{stamp}_{n:02d}', items[n]['ev']))
    append_riffs(song, chosen, gap_first=last >= 0)
    if to.exists() and not a.no_backup:
        bu = to.with_name(to.stem + '_before_pick.gp5')
        bu.write_bytes(to.read_bytes())
        print(f'backup: {bu}')
    gp.write(song, str(to))
    print(f'added {", ".join(c[0] for c in chosen)} to {to}')


def fix_names(a):
    """Add track names to the .mid files of older runs (needed by taste/train.py)."""
    n = 0
    for p in (V1 / 'outputs').glob('*_riffs/riff_*.mid'):
        m = mido.MidiFile(str(p))
        if any(msg.type == 'track_name' for tr in m.tracks for msg in tr):
            continue
        for tr, nm in zip(m.tracks, ('Guitar', 'Bass', 'Drums')):
            tr.insert(0, mido.MetaMessage('track_name', name=nm))
        m.save(str(p)); n += 1
    print(f'fixed {n} files')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    p = sub.add_parser('pick', help='copy riffs of a run into a gp5 (appended at the end)')
    p.add_argument('numbers', type=int, nargs='+')
    p.add_argument('--to', required=True)
    p.add_argument('--run', help='outputs/<run> folder name (default: newest *_riffs run)')
    p.add_argument('--prefix', default='r', help='marker name prefix')
    p.add_argument('--no-backup', action='store_true')
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--src', choices=['riff', 'nn', 'mix'], default='mix')
    ap.add_argument('--palette', help="reference song name, 'picks', or none = random per riff")
    ap.add_argument('--picks', help='path to your picks.gp5 (for --palette picks)')
    ap.add_argument('--tempo', type=int, default=150)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--model', default=str(HERE / 'data' / 'model.pt'))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--no-balance', action='store_true', help='pick riffs from all songs at once (songs with many riffs win)')
    ap.add_argument('--no-clean', action='store_true',
                    help='NN riffs as written (no 16th grid fill-in of short rests, no octave fold)')
    ap.add_argument('--no-learn', action='store_true', help='do not weight songs by your past ratings')
    sub.add_parser('fix-names', help='add track names to .mid files of older runs')
    a = ap.parse_args()
    if a.cmd == 'fix-names':
        fix_names(a)
    elif a.cmd == 'pick':
        pick(a)
    else:
        make(a)


if __name__ == '__main__':
    main()
