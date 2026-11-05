"""Musical features of a generated candidate, read from its MIDI file.

Works on any candidate the generator writes (tracks named Guitar / Bass / Drums,
drums on channel 10). Every feature is a plain number so a simple model can learn
which ones the listener likes.
"""
from collections import Counter
from pathlib import Path

import mido

KICKS = {35, 36}
SNARES = {37, 38, 40}

# feature name -> short explanation (shown when the model explains your taste)
DESCRIPTIONS = {
    "g_notes_per_beat": "guitar notes per beat (busy riffs)",
    "g_chord_ratio": "guitar hits that are chords",
    "g_pedal_ratio": "guitar hits on the pedal (lowest) note",
    "g_range": "guitar pitch range",
    "g_pitch_classes": "number of different notes used",
    "g_step_moves": "small melodic steps (1-2 semitones)",
    "g_semitone_moves": "chromatic half-step moves",
    "g_tritone_moves": "tritone jumps",
    "g_fifth_moves": "fifth / fourth jumps",
    "g_leaps": "large leaps (6+ semitones)",
    "g_rest_ratio": "silence in the guitar",
    "g_offbeat_ratio": "16th off-beat hits (syncopation)",
    "g_triplet_ratio": "triplet hits",
    "g_bar_repetition": "bar-to-bar repetition",
    "d_notes_per_beat": "drum hits per beat",
    "d_kick_per_beat": "kick drum density",
    "d_backbeat": "snare on beats 2 and 4",
    "kick_on_guitar": "guitar hits doubled by the kick",
    "bass_unison": "bass playing with the guitar",
    "tempo": "tempo (BPM)",
}
NAMES = list(DESCRIPTIONS)


def _abs_events(track):
    t, out = 0, []
    for msg in track:
        t += msg.time
        out.append((t, msg))
    return out


def _onsets(track, drum=False):
    """{tick: [pitches]} for note-on events; and note spans for sounding time."""
    starts, spans, open_ = {}, [], {}
    for t, msg in _abs_events(track):
        if msg.type == 'note_on' and msg.velocity > 0:
            starts.setdefault(t, []).append(msg.note)
            open_[msg.note] = t
        elif msg.type in ('note_off', 'note_on'):
            if msg.note in open_:
                spans.append((open_.pop(msg.note), t))
    return starts, spans


def _find(mid, name, channel=None):
    for tr in mid.tracks:
        chans = {m.channel for m in tr if hasattr(m, 'channel')}
        if channel is not None and channel in chans:
            return tr
        if tr.name.strip().lower() == name.lower():
            return tr
    return None


def _share(n, d):
    return n / d if d else 0.0


def extract(midi_path):
    mid = mido.MidiFile(str(midi_path))
    tpq = mid.ticks_per_beat
    gtr = _find(mid, 'Guitar')
    bass = _find(mid, 'Bass')
    drums = _find(mid, 'Drums', channel=9)
    if gtr is None:
        raise ValueError(f'{midi_path}: no Guitar track')

    tempos = [mido.tempo2bpm(m.tempo) for tr in mid.tracks for m in tr if m.type == 'set_tempo']
    g_on, g_spans = _onsets(gtr)
    total_ticks = max([e for _, e in g_spans] + [max(g_on) if g_on else 0, 1])
    beats = max(total_ticks / tpq, 1)

    ticks = sorted(g_on)
    lows = [min(g_on[t]) for t in ticks]
    tops = [max(g_on[t]) for t in ticks]
    pedal = Counter(lows).most_common(1)[0][0] if lows else 0
    ints = [abs(b - a) for a, b in zip(tops, tops[1:]) if b != a]

    sixteenth = tpq // 4
    grid = int(total_ticks // sixteenth) + 1
    sounding = [False] * grid
    for s, e in g_spans:
        for k in range(int(s // sixteenth), min(grid, int((e - 1) // sixteenth) + 1)):
            sounding[k] = True
    off16 = sum(1 for t in ticks if t % (tpq // 2) and t % sixteenth == 0)
    trip = sum(1 for t in ticks if t % sixteenth)

    bar = tpq * 4
    bars = {}
    for t in ticks:
        bars.setdefault(t // bar, set()).add((t % bar, min(g_on[t])))
    keys = sorted(bars)
    sims = []
    for a, b in zip(keys, keys[1:]):
        A, B = bars[a], bars[b]
        sims.append(_share(len(A & B), len(A | B)))

    f = {
        "g_notes_per_beat": len(ticks) / beats,
        "g_chord_ratio": _share(sum(1 for t in ticks if len(set(g_on[t])) > 1), len(ticks)),
        "g_pedal_ratio": _share(sum(1 for x in lows if x == pedal), len(lows)),
        "g_range": (max(tops) - min(lows)) if ticks else 0,
        "g_pitch_classes": len({p % 12 for t in ticks for p in g_on[t]}),
        "g_step_moves": _share(sum(1 for i in ints if i <= 2), len(ints)),
        "g_semitone_moves": _share(sum(1 for i in ints if i == 1), len(ints)),
        "g_tritone_moves": _share(sum(1 for i in ints if i % 12 == 6), len(ints)),
        "g_fifth_moves": _share(sum(1 for i in ints if i % 12 in (5, 7)), len(ints)),
        "g_leaps": _share(sum(1 for i in ints if i >= 6), len(ints)),
        "g_rest_ratio": 1 - _share(sum(sounding), grid),
        "g_offbeat_ratio": _share(off16, len(ticks)),
        "g_triplet_ratio": _share(trip, len(ticks)),
        "g_bar_repetition": sum(sims) / len(sims) if sims else 0.0,
        "tempo": sum(tempos) / len(tempos) if tempos else 120.0,
    }

    if drums is not None:
        d_on, _ = _onsets(drums)
        kicks = {t for t, ps in d_on.items() if set(ps) & KICKS}
        snare_beats = [t for t, ps in d_on.items() if set(ps) & SNARES and t % tpq == 0]
        backbeat = sum(1 for t in snare_beats if (t // tpq) % 4 in (1, 3))
        f["d_notes_per_beat"] = sum(len(v) for v in d_on.values()) / beats
        f["d_kick_per_beat"] = len(kicks) / beats
        f["d_backbeat"] = _share(backbeat, len(snare_beats))
        f["kick_on_guitar"] = _share(sum(1 for t in ticks if t in kicks), len(ticks))
    else:
        f.update(d_notes_per_beat=0.0, d_kick_per_beat=0.0, d_backbeat=0.0, kick_on_guitar=0.0)

    if bass is not None:
        b_on, _ = _onsets(bass)
        f["bass_unison"] = _share(sum(1 for t in b_on if t in g_on), len(b_on))
    else:
        f["bass_unison"] = 0.0
    return [float(f[n]) for n in NAMES]


def candidate_midi(outputs_dir, run_id, cand):
    p = Path(outputs_dir) / run_id / f'{cand}.mid'
    return p if p.exists() else None
