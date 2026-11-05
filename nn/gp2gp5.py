"""Convert Guitar Pro 7/8 (.gp) to Guitar Pro 5 (.gp5), keeping guitars, basses and drums.

  python3 gp2gp5.py input.gp output.gp5

Keeps tuning, instrument program, tempo changes, time signatures, rests, ties,
tuplets, palm mute, dead notes, hammer-on/pull-off, slides, harmonics, bends,
vibrato, accents and dynamics. Orchestral/other tracks are dropped (the NN works
on the band). Needs the current PyGuitarPro (pip install pyguitarpro).
"""
import sys
import zipfile
import xml.etree.ElementTree as ET

import guitarpro as gp

KEEP_TYPES = {'electricGuitar', 'electricBass', 'drumKit', 'steelGuitar', 'nylonGuitar',
              'acousticGuitar', 'bass', 'acousticBass'}
NOTE_VALUES = {'Whole': 1, 'Half': 2, 'Quarter': 4, 'Eighth': 8, '16th': 16, '32nd': 32,
               '64th': 64, '128th': 128}
DYN = {'PPP': 15, 'PP': 31, 'P': 47, 'MP': 63, 'MF': 79, 'F': 95, 'FF': 111, 'FFF': 127}
HARM = {'natural': gp.NaturalHarmonic, 'pinch': gp.PinchHarmonic, 'semi': gp.SemiHarmonic,
        'tap': gp.TappedHarmonic}


def text(e, path, default=None):
    x = e.find(path)
    return x.text.strip() if x is not None and x.text else default


def load_gpif(path):
    with zipfile.ZipFile(path) as z:
        return ET.fromstring(z.read('Content/score.gpif'))


def convert(src, dst):
    r = load_gpif(src)
    idx = lambda tag, child: {e.get('id'): e for e in r.findall(f'{tag}/{child}')}
    bars, voices, beats, notes, rhythms = (idx('Bars', 'Bar'), idx('Voices', 'Voice'),
                                           idx('Beats', 'Beat'), idx('Notes', 'Note'),
                                           idx('Rhythms', 'Rhythm'))
    tracks_x = r.findall('Tracks/Track')
    masters = r.findall('MasterBars/MasterBar')

    tempos = {}
    for a in r.iter('Automation'):
        if a.findtext('Type') == 'Tempo':
            tempos.setdefault(int(a.findtext('Bar')), int(float(a.findtext('Value').split()[0])))

    song = gp.Song()
    song.title = text(r, 'Score/Title', '') or ''
    song.artist = text(r, 'Score/Artist', '') or ''
    song.tempo = tempos.get(0, 120)
    song.tracks = []
    song.measureHeaders = []
    for i, mb in enumerate(masters):
        num, den = map(int, mb.findtext('Time').split('/'))
        h = gp.MeasureHeader(number=i + 1)
        h.timeSignature = gp.TimeSignature(numerator=num, denominator=gp.Duration(value=den))
        song.measureHeaders.append(h)
    # absolute start times for headers
    start = gp.Duration.quarterTime
    for h in song.measureHeaders:
        h.start = start
        start += h.length

    kept = []
    for ti, tx in enumerate(tracks_x):
        typ = text(tx, 'InstrumentSet/Type', '')
        if typ not in KEEP_TYPES:
            continue
        drum = typ == 'drumKit'
        p = tx.find('.//Property[@name="Tuning"]/Pitches')
        tuning = [int(v) for v in p.text.split()] if p is not None else [0] * 6
        if drum:
            tuning = [0] * 6
        if not drum and len(tuning) > 7:          # 8-string guitars are not supported by DadaGP
            continue
        kept.append((ti, tx, drum, tuning, typ))
    # tunings DadaGP (with our patches) can encode
    def diffs(tun):
        hi = list(reversed(tun))
        return [b - a for a, b in zip(hi, hi[1:])]
    GTR_OK = ([-5, -4, -5, -5, -5], [-5, -4, -5, -5, -7], [-5, -4, -5, -5, -5, -5],
              [-5, -4, -5, -5, -5, -7], [-5, -4, -5, -5, -7, -5])
    BASS_OK = ([-5, -5, -5], [-5, -5, -7], [-5, -5, -5, -5], [-5, -5, -5, -7], [-5, -5, -5, -5, -5])
    def ok(item):
        ti, tx, drum, tun, typ = item
        if drum:
            return True
        return diffs(tun) in (BASS_OK if ('Bass' in typ or typ == 'bass') else GTR_OK)
    kept = [k for k in kept if ok(k)]
    # DadaGP limits: max 3 guitars, 1 bass, 1 drum kit
    limits = {'guitar': 3, 'bass': 1, 'drums': 1}
    group = lambda typ: 'drums' if typ == 'drumKit' else ('bass' if 'Bass' in typ or typ == 'bass' else 'guitar')
    count, filtered = {}, []
    for item in kept:
        g = group(item[4])
        if count.get(g, 0) < limits[g]:
            count[g] = count.get(g, 0) + 1
            filtered.append(item[:4])
    kept = filtered

    for k, (ti, tx, drum, tuning) in enumerate(kept):
        t = gp.Track(song, number=k + 1)
        t.name = (text(tx, 'Name', '') or f'Track {k + 1}')[:40]
        t.isPercussionTrack = drum
        t.strings = [gp.GuitarString(number=n + 1, value=v) for n, v in enumerate(reversed(tuning))]
        t.fretCount = 24
        prog = text(tx, './/Sounds/Sound/MIDI/Program', '0')
        t.channel = gp.MidiChannel(channel=9 if drum else min(k, 8) + (1 if k >= 9 else 0),
                                   effectChannel=9 if drum else min(k, 8),
                                   instrument=0 if drum else int(prog or 0))
        t.measures = []
        nstr = len(tuning)
        for bi, mb in enumerate(masters):
            header = song.measureHeaders[bi]
            m = gp.Measure(t, header)
            m.voices = [gp.Voice(m), gp.Voice(m)]
            bar_id = mb.findtext('Bars').split()[ti]
            vids = [v for v in bars[bar_id].findtext('Voices').split() if v != '-1'][:2]
            for vi, vid in enumerate(vids):
                voice = m.voices[vi]
                bt = voices[vid].findtext('Beats')
                for b_id in (bt.split() if bt else []):
                    bx = beats[b_id]
                    if bx.find('GraceNotes') is not None:
                        continue
                    rx = rhythms[bx.find('Rhythm').get('ref')]
                    dur = gp.Duration(value=NOTE_VALUES.get(rx.findtext('NoteValue'), 4))
                    dot = rx.find('AugmentationDot')
                    if dot is not None:
                        dur.isDotted = True
                    tup = rx.find('PrimaryTuplet')
                    if tup is not None:
                        dur.tuplet = gp.Tuplet(enters=int(tup.get('num')), times=int(tup.get('den')))
                    beat = gp.Beat(voice, duration=dur)
                    vel = DYN.get(text(bx, 'Dynamic', 'MF'), 95)
                    used = set()
                    nid = text(bx, 'Notes')
                    for n_id in (nid.split() if nid else []):
                        nx = notes[n_id]
                        props = {pp.get('name'): pp for pp in nx.findall('Properties/Property')}
                        if drum:
                            midi = int(text(props['Midi'], 'Number', '35')) if 'Midi' in props else 35
                            st = next((s for s in range(1, 7) if s not in used), None)
                            if st is None:
                                continue
                            note = gp.Note(beat, value=midi, string=st, velocity=vel,
                                           type=gp.NoteType.normal)
                        else:
                            if 'String' not in props or 'Fret' not in props:
                                continue
                            sidx = int(text(props['String'], 'String'))
                            st = nstr - sidx
                            if st in used or not 1 <= st <= nstr:
                                continue
                            note = gp.Note(beat, value=int(text(props['Fret'], 'Fret')), string=st,
                                           velocity=vel, type=gp.NoteType.normal)
                            tie = nx.find('Tie')
                            if tie is not None and tie.get('destination') == 'true':
                                note.type = gp.NoteType.tie
                            e = note.effect
                            if 'PalmMuted' in props:
                                e.palmMute = True
                            if 'Muted' in props:
                                note.type = gp.NoteType.dead
                            if 'HopoOrigin' in props:
                                e.hammer = True
                            if 'Slide' in props:
                                e.slides = [gp.SlideType.shiftSlideTo]
                            if 'LetRing' in props:
                                e.letRing = True
                            if 'Harmonic' in props:
                                ht = (text(props.get('HarmonicType', props['Harmonic']), 'HType', 'natural') or 'natural').lower()
                                cls = HARM.get(ht, gp.NaturalHarmonic)
                                e.harmonic = cls() if cls is not gp.ArtificialHarmonic else cls()
                            if 'Bended' in props and 'BendDestinationValue' in props:
                                val = float(text(props['BendDestinationValue'], 'Float', '100'))
                                semis = max(1, min(4, round(val / 50)))     # 100 = full tone
                                e.bend = gp.BendEffect(type=gp.BendType.bend, value=semis * 50,
                                                       points=[gp.BendPoint(0, 0), gp.BendPoint(6, semis),
                                                               gp.BendPoint(12, semis)])
                            if nx.find('Vibrato') is not None:
                                e.vibrato = True
                            if nx.find('Accent') is not None:
                                e.accentuatedNote = True
                        used.add(st)
                        beat.notes.append(note)
                    beat.status = gp.BeatStatus.normal if beat.notes else gp.BeatStatus.rest
                    voice.beats.append(beat)
            if not m.voices[0].beats:
                rest = gp.Beat(m.voices[0], duration=gp.Duration(value=1))
                rest.status = gp.BeatStatus.rest
                m.voices[0].beats.append(rest)
            if k == 0 and bi in tempos and bi > 0:
                mix = gp.MixTableChange(tempo=gp.MixTableItem(value=tempos[bi]))
                m.voices[0].beats[0].effect.mixTableChange = mix
            t.measures.append(m)
        song.tracks.append(t)
    if not song.tracks:
        raise ValueError('no guitar/bass/drum tracks')
    gp.write(song, dst)
    return [t.name for t in song.tracks]


if __name__ == '__main__':
    print(convert(sys.argv[1], sys.argv[2]))
