"""Motif development on DadaGP tokens (after Shan & Chiu 2009, sec. 4.3), for riffs.

  transpose(part, k)    sequence: move the riff k semitones (same frets shape; refused if off the neck)
  retrograde(part)      play each bar's notes in reverse order, rhythm unchanged
  augment(part)         halftime: every duration doubled (1 bar -> 2 bars), breakdown feel
  diminish(part)        double time: every duration halved (2 bars -> 1 bar)

`part` is a token list starting with 'new_measure', bars of 3840 ticks.
"""
import random

BAR = 3840
LOWEST = {('distorted0', 's6'): -2, ('distorted0', 's7'): -2, ('bass', 's5'): -2, ('bass', 's6'): -2}


def _bars(part):
    out, cur = [], None
    for t in part:
        if t == 'new_measure':
            if cur is not None:
                out.append(cur)
            cur = []
        elif cur is not None:
            cur.append(t)
    if cur is not None:
        out.append(cur)
    return out


def _join(bars):
    out = []
    for b in bars:
        out += ['new_measure'] + b
    return out


def _steps(bar):
    """[(event tokens, wait)] - events are notes with their effect tokens."""
    steps, ev = [], []
    for t in bar:
        if t.startswith('wait:'):
            steps.append((ev, int(t[5:]))); ev = []
        else:
            ev.append(t)
    if ev:
        steps.append((ev, 0))
    return steps


def transpose(part, k):
    out = []
    for t in part:
        if ':note:' in t:
            inst, _, s, f = t.split(':')
            nf = int(f[1:]) + k
            if nf < LOWEST.get((inst, s), 0) or nf > 24:
                return None
            t = f'{inst}:note:{s}:f{nf}'
        out.append(t)
    return out


def retrograde(part):
    """Reverse the order of the note events in each bar, keeping the durations where they were."""
    bars = []
    for b in _bars(part):
        st = _steps(b)
        evs = [e for e, _ in st][::-1]
        nb = []
        for (_, w), e in zip(st, evs):
            nb += e + ([f'wait:{w}'] if w else [])
        bars.append(nb)
    return _join(bars)


def _retime(part, factor):
    """Scale every wait by factor and re-cut into 3840-tick bars (waits split at bar lines)."""
    bars, cur, room = [], [], BAR
    for b in _bars(part):
        for ev, w in _steps(b):
            cur += ev
            w = int(round(w * factor))
            while w > 0:
                d = min(w, room)
                cur.append(f'wait:{d}')
                w -= d; room -= d
                if room == 0:
                    bars.append(cur); cur, room = [], BAR
    if cur and room < BAR:                      # pad a partial last bar
        cur.append(f'wait:{room}'); bars.append(cur)
    return _join(bars)


def augment(part):
    return _retime(part, 2)


def diminish(part):
    return _retime(part, 0.5)


def develop(part, rng, vocab_ok=None):
    """Return (new part with the same number of bars, description), or (part, 'repeat')."""
    n = sum(1 for t in part if t == 'new_measure')
    choices = [('transpose', 0.45), ('halftime', 0.25), ('retrograde', 0.15), ('doubletime', 0.15)]
    r = rng.random()
    for name, p in choices:
        if r < p:
            break
        r -= p
    new, desc = None, 'repeat'
    if name == 'transpose':
        for k in rng.sample([1, -1, 2, -2, 5, -5, 3], 7):
            new = transpose(part, k)
            if new:
                desc = f'transpose {k:+d}'; break
    elif name == 'retrograde':
        new, desc = retrograde(part), 'retrograde'
    elif name == 'halftime' and n >= 2:
        half = _join(_bars(part)[:n // 2])
        new, desc = augment(half), 'halftime'
        if n % 2:
            new += _join(_bars(part)[-1:])
    elif name == 'doubletime':
        d = diminish(part)
        new, desc = d + d, 'doubletime'
        if sum(1 for t in new if t == 'new_measure') != n:
            new = None
    if new is None or (vocab_ok and any(t not in vocab_ok for t in new)):
        return part, 'repeat'
    if sum(1 for t in new if t == 'new_measure') != n:
        return part, 'repeat'
    return new, desc


def emphasize(part):
    """'Stronger version' of a riff: same rhythm, single non-pedal notes on the low strings become
    power chords (root + 5th + octave). Used for the last chorus."""
    out = []
    for ev, w in _steps_tokens(part):
        notes = [t for t in ev if t.startswith('distorted0:note')]
        add = []
        if len(notes) == 1:
            _, _, s, f = notes[0].split(':')
            f = int(f[1:])
            if s == 's6' and f > -2:                     # not the open (pedal) low string
                add = [f'distorted0:note:s5:f{f + 2}', f'distorted0:note:s4:f{f + 2}']
            elif s == 's5':
                add = [f'distorted0:note:s4:f{f + 2}', f'distorted0:note:s3:f{f + 2}']
        out += ev[:1] + add + ev[1:] if add else ev
        if w is not None:
            out.append(f'wait:{w}')
    return out


def _steps_tokens(part):
    """Like _steps but keeps new_measure tokens as separate events."""
    res, ev = [], []
    for t in part:
        if t.startswith('wait:'):
            res.append((ev, int(t[5:]))); ev = []
        elif t == 'new_measure':
            if ev:
                res.append((ev, None)); ev = []
            res.append((['new_measure'], None))
        else:
            ev.append(t)
    if ev:
        res.append((ev, None))
    return res
