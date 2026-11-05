"""Rule-based metal drums that follow the guitar (idea from mix.py's drumify / Magenta Drumify).

Per bar, the guitar's 16th-step rhythm picks a groove:
  dense 16ths      -> blast      (kick + snare alternating 8ths, ride on 8ths)
  medium           -> double     (kick on every guitar hit, snare on 2 & 4, hat/china on quarters)
  sparse / chords  -> halftime   (kick on guitar hits, snare on 3, china on quarters)
  guitar silent    -> ride       (light time on the ride, kick on 1)
Crash on every 4-bar phrase start and on groove changes; tom fill on the last beat before a
change (and every 8 bars). Velocities: bar head > beats > 8ths > 16ths, like mix.py.
"""
import guitarpro as gp

KICK, SNARE, HAT, RIDE, CHINA, CRASH = 36, 38, 42, 51, 52, 49
TOMS = [50, 48, 47, 45, 43, 41]


def guitar_grid(song, bar_index, steps):
    """step -> (hits, palm muted?, is chord) for all guitar tracks in one bar."""
    grid = {}
    for t in song.tracks:
        if t.isPercussionTrack or 'bass' in t.name.lower():
            continue
        m = t.measures[bar_index]
        for v in m.voices:
            pos = 0
            for b in v.beats:
                st = pos * steps // m.header.length if m.header.length else 0
                notes = [n for n in b.notes if n.type not in (gp.NoteType.tie, gp.NoteType.rest)]
                if notes and st < steps:
                    h, pm, ch = grid.get(st, (0, True, False))
                    grid[st] = (h + 1, pm and all(n.effect.palmMute for n in notes), ch or len(notes) > 1)
                pos += b.duration.time
    return grid


def groove_of(grid, steps):
    n = len(grid)
    if n == 0:
        return 'ride'
    if n >= steps * 0.7:
        return 'blast'
    if n >= steps * 0.35:
        return 'double'
    return 'halftime'


def accent(st, sub_per_beat):
    if st == 0:
        return 1.0
    sub = st % sub_per_beat
    return 0.85 if sub == 0 else 0.7 if sub == sub_per_beat // 2 else 0.55


def drum_events(song, overrides=None):
    """[(bar_index, step, steps, [(pitch, velocity)])]"""
    bars = len(song.measureHeaders)
    info = []
    for bi, h in enumerate(song.measureHeaders):
        num = h.timeSignature.numerator
        den = h.timeSignature.denominator.value
        steps = max(1, num * 16 // den)
        grid = guitar_grid(song, bi, steps)
        g = (overrides or {}).get(bi) or groove_of(grid, steps)
        info.append((steps, grid, g, num * 4 // den or 1))
    out = []
    for bi, (steps, grid, groove, beats) in enumerate(info):
        sub = max(1, steps // max(1, beats))           # 16th steps per beat
        nxt = info[bi + 1][2] if bi + 1 < bars else None
        prev = info[bi - 1][2] if bi > 0 else None
        fill = ((nxt is not None and nxt != groove and groove != 'ride') or (bi % 8 == 7 and groove != 'ride')) and nxt != 'stop'
        fill_from = steps - sub if fill else steps
        if groove == 'rest':                           # silent bar
            continue
        if groove == 'stop':                           # band hit on beat 1, then silence
            out.append((bi, 0, steps, [(CRASH, 120), (KICK, 120), (CHINA, 100)]))
            continue
        for st in range(steps):
            hits = []
            vel = lambda f: max(30, min(127, round(127 * accent(st, sub) * f)))
            g = grid.get(st)
            beat, s = divmod(st, sub)
            if st >= fill_from:
                j = st - fill_from
                hits.append((TOMS[min(j, len(TOMS) - 1)], vel(0.9)))
                if j % 2 == 0:
                    hits.append((KICK, vel(0.8)))
                out.append((bi, st, steps, hits))
                continue
            if groove == 'blast':
                if s % 2 == 0:
                    hits.append((KICK, vel(0.95)))
                    hits.append((RIDE, vel(0.6)))
                else:
                    hits.append((SNARE, vel(0.85)))
            elif groove == 'double':
                if g or s == 0:
                    hits.append((KICK, vel(0.95)))
                if s == 0 and beat % 2 == 1:
                    hits.append((SNARE, vel(1.0)))
                if s == 0:
                    hits.append((CHINA if beat % 2 == 0 else HAT, vel(0.7)))
            elif groove == 'halftime':
                if g and (not g[1] or g[2] or s % 2 == 0):  # kick doubles the accents / chugs
                    hits.append((KICK, vel(1.0)))
                if st == 0 and not g:
                    hits.append((KICK, vel(1.0)))
                if s == 0 and beat == beats // 2:
                    hits.append((SNARE, vel(1.0)))
                if s == 0:
                    hits.append((CHINA, vel(0.75)))
            else:                                        # ride: guitar silent
                if s == 0:
                    hits.append((RIDE, vel(0.6)))
                if st == 0:
                    hits.append((KICK, vel(0.8)))
            if st == 0 and (bi % 4 == 0 or prev != groove) and groove != 'ride':
                hits.append((CRASH, 127))
            if hits:
                out.append((bi, st, steps, hits))
    return out


def add_drum_track(song, overrides=None):
    """Replace any drum track with rule-based drums; returns groove per bar."""
    song.tracks = [t for t in song.tracks if not t.isPercussionTrack]
    t = gp.Track(song, number=len(song.tracks) + 1)
    t.name = 'Drums'
    t.isPercussionTrack = True
    t.strings = [gp.GuitarString(number=i + 1, value=0) for i in range(6)]
    t.channel = gp.MidiChannel(channel=9, effectChannel=9, instrument=0)
    by_bar = {}
    for bi, st, steps, hits in drum_events(song, overrides):
        by_bar.setdefault(bi, (steps, {}))[1][st] = hits
    t.measures = []
    for bi, h in enumerate(song.measureHeaders):
        m = gp.Measure(t, h)
        m.voices = [gp.Voice(m), gp.Voice(m)]
        steps, hits = by_bar.get(bi, (max(1, h.timeSignature.numerator * 16 // h.timeSignature.denominator.value), {}))
        for st in range(steps):
            b = gp.Beat(m.voices[0], duration=gp.Duration(value=16))
            for k, (p, v) in enumerate(hits.get(st, [])[:6]):
                b.notes.append(gp.Note(b, value=p, string=k + 1, velocity=v, type=gp.NoteType.normal))
            b.status = gp.BeatStatus.normal if b.notes else gp.BeatStatus.rest
            m.voices[0].beats.append(b)
        t.measures.append(m)
    song.tracks.append(t)
    for i, tr in enumerate(song.tracks):
        tr.number = i + 1
