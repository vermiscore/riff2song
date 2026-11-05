"""After DadaGP decoding: put drop tunings back the guitarist's way and write a MIDI file.

DadaGP rebuilds 7-string drop tunings with BOTH strings 6 and 7 lowered; the pitches
are right but string 6 ends up two frets off. For drop-G-style tracks we raise string 6
back up and move its notes two frets down (same pitches, normal fingering). Same for
5-string drop basses (string 4).
"""
import guitarpro as gp
import mido

import drumify

PROGRAMS = {'guitar': 30, 'bass': 33}


def _diffs(track):
    v = [s.value for s in track.strings]          # string 1 (high) .. lowest
    return [b - a for a, b in zip(v, v[1:])]


def _raise_string(track, string_no):
    frets = [n.value for m in track.measures for vo in m.voices for b in vo.beats
             for n in b.notes if n.string == string_no and n.type != gp.NoteType.dead]
    if frets and min(frets) < 2:
        return False                            # genuinely needs the low tuning
    track.strings[string_no - 1].value += 2
    for m in track.measures:
        for vo in m.voices:
            for b in vo.beats:
                for n in b.notes:
                    if n.string == string_no:
                        n.value = max(0, n.value - 2)
    return True


def fix_tunings(song):
    for t in song.tracks:
        if t.isPercussionTrack:
            continue
        d = _diffs(t)
        if len(t.strings) == 7 and d == [-5, -4, -5, -5, -7, -5]:
            _raise_string(t, 6)
        elif len(t.strings) == 5 and d == [-5, -5, -7, -5]:
            _raise_string(t, 4)


def to_midi(song, path, tpq=480):
    mid = mido.MidiFile(ticks_per_beat=tpq)
    meta = mido.MidiTrack(); mid.tracks.append(meta)
    meta.append(mido.MetaMessage('set_tempo', tempo=mido.bpm2tempo(song.tempo), time=0))
    names, gtr_n = [], 0
    for ti, t in enumerate(song.tracks):
        drum = t.isPercussionTrack
        is_bass = (not drum) and len(t.strings) <= 6 and t.strings[-1].value < 30 and 'bass' in t.name.lower()
        if drum:
            name, ch, prog = 'Drums', 9, None
        elif is_bass:
            name, ch, prog = 'Bass', 1, PROGRAMS['bass']
        else:
            gtr_n += 1
            name = 'Guitar' if gtr_n == 1 else f'Guitar {gtr_n}'
            ch, prog = (0 if gtr_n == 1 else min(1 + gtr_n, 8)), PROGRAMS['guitar']
        names.append(name)
        tr = mido.MidiTrack(); mid.tracks.append(tr)
        tr.append(mido.MetaMessage('track_name', name=name, time=0))
        if prog is not None:
            tr.append(mido.Message('program_change', program=prog, channel=ch, time=0))
        ev = []
        for m in t.measures:
            pos = m.header.start
            for b in m.voices[0].beats:
                d = b.duration.time
                for n in b.notes:
                    if n.type in (gp.NoteType.tie, gp.NoteType.rest):
                        continue
                    p = n.value if drum else t.strings[n.string - 1].value + n.value
                    on = (pos - 960) * tpq // 960
                    L = (d * tpq // 960) // (3 if n.type == gp.NoteType.dead else 1)   # palm mutes ring for their full length
                    vel = 60 if n.type == gp.NoteType.dead else min(127, max(1, n.velocity))
                    ev.append((on, 1, mido.Message('note_on', note=p, velocity=vel, channel=ch)))
                    ev.append((on + max(30, L - 10), 0, mido.Message('note_off', note=p, velocity=0, channel=ch)))
                pos += d
        ev.sort(key=lambda e: (e[0], e[1])); last = 0
        for t0, _, msg in ev:
            msg.time = t0 - last; last = t0; tr.append(msg)
    mid.save(str(path))
    return names


def double_guitar(song):
    """If there is a single guitar, add a second one playing the same part (double tracking)."""
    import copy
    gtr = [t for t in song.tracks if not t.isPercussionTrack and 'bass' not in t.name.lower()]
    if len(gtr) != 1:
        return
    g2 = copy.deepcopy(gtr[0])
    g2.name = 'Guitar 2'
    g2.channel = gp.MidiChannel(channel=2, effectChannel=3, instrument=gtr[0].channel.instrument)
    for m in g2.measures:
        m.track = g2
    song.tracks.insert(song.tracks.index(gtr[0]) + 1, g2)
    for i, tr in enumerate(song.tracks):
        tr.number = i + 1


def finish(gp5_path, midi_path, drum_overrides=None, octave_bars=(), markers=None):
    """Fix tunings, double the guitar, add rule drums (optionally per-bar grooves), write gp5 + MIDI.
    octave_bars: bar indices where Guitar 2 plays an octave up; markers: {bar_index: text}."""
    song = gp.parse(str(gp5_path))
    fix_tunings(song)
    double_guitar(song)
    if octave_bars:
        g2 = next((t for t in song.tracks if t.name == 'Guitar 2'), None)
        for bi in octave_bars:
            if g2 is None or bi >= len(g2.measures):
                continue
            ns = [n for b in g2.measures[bi].voices[0].beats for n in b.notes]
            if ns and all(n.value + 12 <= 24 for n in ns):
                for n in ns:
                    n.value += 12
    drumify.add_drum_track(song, drum_overrides)   # drums by rules, following the generated guitar
    for bi, text in (markers or {}).items():
        if bi < len(song.measureHeaders):
            song.measureHeaders[bi].marker = gp.Marker(title=text, color=gp.Color(200, 0, 0))
    gp.write(song, str(gp5_path))
    return to_midi(song, midi_path)
