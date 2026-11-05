"""Generate songs with the trained GPT and save them as candidates you can rate.

  python3 nn/generate.py --n 5 --tokens 3000 --tempo 150

Each candidate is written to v1/outputs/<time>_nn/ as cand_XX.gp5 (Guitar Pro 5,
opens in TuxGuitar), cand_XX.mid (for listening / taste/rate.py) and cand_XX.txt
(how much of it is copied from the reference songs). The candidates are added to
v1/outputs/ratings.csv so taste/rate.py picks them up.
"""
import argparse
import random
import subprocess
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
V1 = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(V1))
from gpt import GPT                                                  # noqa: E402
import postprocess as P
import develop as DV
import riffs as RF
import collections
RT_NAMES = ['1', 'b2', '2', 'b3', '3', '4', 'b5', '5', 'b6', '6', 'b7', '7']                                              # noqa: E402

DATA = HERE / 'data'
COPY_N = 32         # a run of this many tokens identical to a reference counts as copying


def dadagp_python():
    for p in (HERE / 'venv06' / 'bin' / 'python', ):
        if p.exists():
            return str(p)
    sys.exit('run nn/setup.sh first (creates nn/venv06 with PyGuitarPro 0.6 for DadaGP)')


def load_model(path):
    ck = torch.load(path, map_location='cpu')
    cfg = ck['config']
    m = GPT(cfg['vocab'], ctx=cfg['ctx'], d=cfg['d'], layers=cfg['layers'], heads=cfg['heads'],
            dropout=0.0)
    m.load_state_dict(ck['state'])
    m.eval()
    return m, ck['vocab']


HEADER = ('start', 'end', 'unknown')


def build_masks(vocab, allowed):
    """Precompute token classes for constrained sampling."""
    import torch as T
    V = len(vocab)
    wait_val = T.full((V,), -1)
    banned = T.zeros(V, dtype=T.bool)
    effect = T.zeros(V, dtype=T.bool)
    note = T.zeros(V, dtype=T.bool)
    for i, t in enumerate(vocab):
        head = t.split(':')[0]
        if t.startswith('wait:'):
            wait_val[i] = int(t.split(':')[1])
        elif t in HEADER or head in ('downtune', 'tempo', 'artist') or t == '<pad>':
            banned[i] = True
        elif head in ('nfx', 'bfx'):
            effect[i] = True
        elif head in ('param', 'new_measure') or t == 'new_measure':
            pass
        elif head not in allowed:
            banned[i] = True                       # other instruments (extra guitars, drums)
        elif ':note:' in t:
            note[i] = True
    return wait_val, banned, effect, note


def build_rules(vocab):
    """Token groups for the taste rules (A): fewer guitar rests, fewer triplets,
    more chords, bass following the guitar."""
    def m(f):
        return torch.tensor([f(t) for t in vocab])
    def wv(t):
        return int(t.split(':')[1]) if t.startswith('wait:') else 0
    return dict(
        g_rest=m(lambda t: t == 'distorted0:rest'),
        b_rest=m(lambda t: t == 'bass:rest'),
        g_note=m(lambda t: t.startswith('distorted0:note')),
        b_note=m(lambda t: t.startswith('bass:note')),
        wait=m(lambda t: wv(t) > 0),
        triplet=m(lambda t: wv(t) >= 40 and wv(t) % 120 != 0),
    )


def apply_rules(logits, R, s, g_cnt, b_cnt):
    """Shift logits by rule strength s, given guitar/bass notes already placed in this time step."""
    logits[R['g_rest']] -= 2.0 * s                  # rests in the guitar: rated much lower
    logits[R['triplet']] -= 1.5 * s                 # triplets: rated lower
    if g_cnt == 0:
        logits[R['wait']] -= 0.7 * s                # step with no new guitar hit -> mild penalty
    if g_cnt == 1:
        logits[R['g_note']] += 0.8 * s              # second note on the same step = chord
    if g_cnt >= 1 and b_cnt == 0:
        logits[R['b_note']] += 1.0 * s              # bass plays with the guitar
        logits[R['b_rest']] -= 2.0 * s
        logits[R['wait']] -= 1.0 * s


def sample_bars(model, ids, n_bars, vocab, stoi, masks, temperature, top_k, bar_ticks=3840,
                rules=None, strength=0.0, palette=None):
    """Sample exactly n_bars 4/4 measures: every bar's waits add up to bar_ticks."""
    wait_val, banned, effect, note = masks
    nm = stoi['new_measure']
    out, prev = [], None
    for _ in range(n_bars):
        ids.append(nm); out.append('new_measure'); prev = nm
        rem = bar_ticks
        guard = 0
        g_cnt = b_cnt = 0
        while rem > 0 and guard < 400:
            guard += 1
            with torch.no_grad():
                logits = model(torch.tensor([ids[-model.ctx:]]))[0][0, -1] / max(temperature, 1e-5)
            if rules is not None and strength > 0:
                apply_rules(logits, rules, strength, g_cnt, b_cnt)
            if palette is not None:
                logits += palette.bias(palette.low)
            bad = banned.clone()
            bad[nm] = True
            bad |= (wait_val > rem)
            if not (note[prev] or effect[prev]):
                bad |= effect                       # effects only right after a note
            if guard > 300:                          # stuck: only allow waits
                bad |= (wait_val < 0)
            logits[bad] = -float('inf')
            if top_k:
                v, _ = torch.topk(logits, min(top_k, int((~bad).sum())))
                logits[logits < v[-1]] = -float('inf')
            i = int(torch.multinomial(torch.softmax(logits, -1), 1))
            ids.append(i); out.append(vocab[i]); prev = i
            tok = vocab[i]
            if palette is not None and tok.startswith('distorted0:note'):
                p = int(palette.pitch[i])
                if (p - palette.tonic) % 12:
                    palette.low = p
            if tok.startswith('distorted0:'):
                g_cnt += 1
            elif tok.startswith('bass:'):
                b_cnt += 1
            if wait_val[i] > 0:
                rem -= int(wait_val[i])
                g_cnt = b_cnt = 0
    return out


RIFF_FORMS = {2: [('AA', 0.4), ('AB', 0.6)],
              4: [('AAAB', 0.45), ('ABAC', 0.25), ('AAAA', 0.1), ('ABCD', 0.2)]}


def load_structure(path=None):
    """Learned song forms / riff forms from nn/structure.py (None if not built yet)."""
    import json
    p = Path(path) if path else HERE / 'data' / 'structure.json'
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    RIFF_FORMS[4] = [tuple(x) for x in s['riff4']]
    RIFF_FORMS[2] = [tuple(x) for x in s['riff2']]
    return s


DATA_TOKENS = HERE / 'data' / 'tokens'


def make_palette(a, vocab, home, template_song=None):
    """--palette NAME / 'none'; default = the palette of the --form template song (if mined)."""
    import palette as PL
    if a.palette == 'none':
        return None
    pals = PL.load()
    if not pals:
        return None
    key = PL.find(pals, a.palette) if a.palette else (PL.find(pals, template_song) if template_song else None)
    if key is None:
        return None
    p = PL.Palette(pals[key], vocab, home_root(DATA_TOKENS), a.palette_strength, base=PL.base(pals))
    p.name = key
    return p


def home_root(token_dir):
    """Most common bar root (by fret pitch, ignoring downtune) of the training songs = the NN's home note."""
    import roots as RT
    c = collections.Counter()
    for p in Path(token_dir).glob('*.txt'):
        t = [x for x in p.read_text().split() if not x.startswith('downtune:')]
        c.update(x for x in RT.bar_roots(t) if x is not None)
    return c.most_common(1)[0][0] if c else 2


class RootGuide:
    """Target root per new bar (relative to the home note) and candidate picking."""
    def __init__(self, roots_model, n, rng, tonic, k=3):
        self.k, self.tonic, self.rng = k, tonic, rng
        bb = collections.defaultdict(dict)
        for key, v in roots_model['bar_bigram'].items():
            x, y = map(int, key.split(','))
            bb[x][y] = v
        pats = [(tuple(p), s) for p, s in roots_model['patterns']]

        def chain():
            seq = [0]
            for _ in range(n - 1):
                nxt = bb.get(seq[-1]) or {0: 1}
                seq.append(rng.choices(list(nxt), weights=list(nxt.values()))[0])
            return seq

        def score(seq):                       # paper: more frequent patterns contained = better
            c = [x for i, x in enumerate(seq) if i == 0 or seq[i - 1] != x]
            return sum(s for p, s in pats if any(tuple(c[i:i + len(p)]) == p for i in range(len(c) - len(p) + 1)))
        self.plan = max((chain() for _ in range(30)), key=score) if n > 0 else []
        self.i = 0
        self.used, self.got = [], []

    def next(self):
        t = self.plan[self.i] if self.i < len(self.plan) else 0
        self.i += 1
        self.used.append(t)
        return t

    def root_of(self, bar):
        import roots as RT
        r = RT.bar_roots(['downtune:0'] + bar)
        r = [x for x in r if x is not None]
        return None if not r else (r[0] - self.tonic) % 12


def new_bar(model, ids, vocab, stoi, masks, temperature, top_k, guide=None, **kw):
    """One new bar; with a RootGuide, best of k tries whose root matches the planned root."""
    if guide is None:
        return sample_bars(model, ids, 1, vocab, stoi, masks, temperature, top_k, **kw)
    target, best = guide.next(), None
    for _ in range(guide.k):
        trial = list(ids)
        bar = sample_bars(model, trial, 1, vocab, stoi, masks, temperature, top_k, **kw)
        r = guide.root_of(bar)
        # exact match best; otherwise prefer the home note (a wrong neighbour sounds like a mistake)
        d = 0 if r == target else (1 if r == 0 else 2 + (12 if r is None else min((r - target) % 12, (target - r) % 12)))
        if best is None or d < best[0]:
            best = (d, bar, trial)
        if d == 0:
            break
    bar, trial = best[1], best[2]
    r = guide.root_of(bar)
    if r is not None and r != target:               # still off: move the bar onto the planned root
        delta = (target - r) % 12
        for k in sorted({delta, delta - 12}, key=abs):
            moved = DV.transpose(bar, k)
            if moved:
                trial = trial[:len(trial) - len(bar)] + [stoi[t] for t in moved if t in stoi]
                bar = moved
                break
    ids[:] = trial
    guide.got.append(guide.root_of(bar))
    return bar


def sample_riff(model, ids, n_bars, vocab, stoi, masks, temperature, top_k, rng, riff=1.0, **kw):
    """Like sample_bars, but builds the phrase from repeated bars the way real riffs do
    (e.g. A A A B: one bar played three times, the last bar varied). riff=0 -> all bars new."""
    forms = RIFF_FORMS.get(n_bars)
    if not forms or rng.random() >= riff:
        if kw.get('guide') is None:
            kw.pop('guide', None)
            return sample_bars(model, ids, n_bars, vocab, stoi, masks, temperature, top_k, **kw)
        return sum((new_bar(model, ids, vocab, stoi, masks, temperature, top_k, **kw) for _ in range(n_bars)), [])
    r, form = rng.random() * sum(p for _, p in forms), forms[-1][0]
    for f, p in forms:
        if r < p:
            form = f; break
        r -= p
    made, out = {}, []
    for ch in form:
        if ch in made:
            bar = made[ch]
            ids.extend(stoi[t] for t in bar if t in stoi)
        else:
            bar = new_bar(model, ids, vocab, stoi, masks, temperature, top_k, **kw)
            made[ch] = bar
        out += bar
    return out


def sample_section(model, ids, L, vocab, stoi, masks, temperature, top_k, rng, riff, **kw):
    """L bars made of 4-bar / 2-bar riff groups (learned forms), remainder single bars."""
    out = []
    while L > 0:
        n = 4 if L >= 4 else (2 if L >= 2 else 1)
        out += sample_riff(model, ids, n, vocab, stoi, masks, temperature, top_k, rng, riff, **kw)
        L -= n
    return out


def compose_structured(model, vocab, stoi, prompt, target_bars, temperature, top_k, allowed, rng,
                       structure, scorer=None, pick=1, rules=None, strength=0.0, riff=1.0,
                       develop_p=0.0, root_k=0, tonic=0, gen='nn', composer=None, window=None):
    """Follow the section form of a reference song (Shan & Chiu style): each new label is
    generated by the NN (best of `pick` by the taste model), repeated labels reuse it."""
    masks = build_masks(vocab, allowed)
    ids = [stoi[t] for t in prompt]
    def cover(s):
        return sum(x['bars'] for x in s['sections'] if x['repeated']) / max(1, s['bars'])
    songs = [s for s in structure['songs'] if s['bars'] >= 16 and len(s['sections']) >= 3 and cover(s) >= 0.5]
    tmpl = rng.choice(songs)
    sections = tmpl['sections']
    if window:                                   # an excerpt of the form: 2+ sections, 16-24 bars
        lo, hi = window
        cands = []
        for s in songs:
            secs = s['sections']
            for i in range(len(secs)):
                tot, labs = 0, []
                for j in range(i, len(secs)):
                    tot += max(1, min(8, secs[j]['bars'])); labs.append(secs[j]['label'])
                    if tot > hi:
                        break
                    if tot >= lo and len(set(labs)) >= 2:
                        cands.append((s, secs[i:j + 1], len(labs) > len(set(labs))))
        returns = [c for c in cands if c[2]]
        pool = returns if returns and rng.random() < 0.7 else cands
        if pool:
            tmpl, sections, _ = rng.choice(pool)
            target_bars = 10 ** 6
    made, body, plan, total = {}, [], [], 0
    for sec in sections:
        if total >= target_bars:
            break
        lab, L = sec['label'], max(1, min(8, sec['bars']))     # cap long sections so returns fit in the song
        dev = ''
        if lab in made:
            part = made[lab]
            if develop_p > 0 and rng.random() < develop_p:
                part, dev = DV.develop(part, rng)
                dev = '' if dev == 'repeat' else f'({dev})'
            ids.extend(stoi[t] for t in part if t in stoi)
        else:
            mk = (lambda: RootGuide(structure['roots'], L, rng, tonic, root_k)) \
                if root_k > 0 and 'roots' in structure else (lambda: None)
            src = gen if gen != 'mix' else rng.choice(['nn', 'prefab'])
            if composer is None:
                src = 'nn'
            if src == 'prefab':
                def prefab_section():
                    plan = (RootGuide(structure['roots'], L, rng, 0, 1).plan
                            if 'roots' in structure else [0] * L)
                    out, left, i = [], L, 0
                    while left > 0:
                        n = 4 if left >= 4 else (2 if left >= 2 else 1)
                        ph, _ = composer.phrase(n, RIFF_FORMS, plan[i:])
                        out += ph; left -= n; i += n
                    return out
                cands = [prefab_section() for _ in range(max(1, pick if scorer is not None else 1))]
                part = max(cands, key=lambda c: (scorer(c) or -9)) if len(cands) > 1 else cands[0]
                ids.extend(stoi[t] for t in part if t in stoi)
                made[lab] = part
                body += part
                total += L
                plan.append(f'{lab}{L}(prefab)')
                continue
            if scorer is not None and pick > 1:
                best = None
                for _ in range(pick):
                    trial = list(ids)
                    cand = sample_section(model, trial, L, vocab, stoi, masks, temperature, top_k, rng, riff,
                                          rules=rules, strength=strength, guide=mk())
                    s = scorer(cand)
                    if best is None or (s is not None and s > best[0]):
                        best = (s if s is not None else -9, cand, trial)
                part, ids[:] = best[1], best[2]
            else:
                guide = mk()
                part = sample_section(model, ids, L, vocab, stoi, masks, temperature, top_k, rng, riff,
                                      rules=rules, strength=strength, guide=guide)
                if guide is not None:
                    dev = '[' + ' '.join(RT_NAMES[x] + ('' if x == y else '?') for x, y in zip(guide.used, guide.got)) + ']'
            made[lab] = part
        body += part
        total += L
        plan.append(f'{lab}{L}{dev}')
    plan.insert(0, f'form from "{tmpl["name"][:40]}":')
    return body, plan


FORM_DRUMS = {'BD': 'halftime', 'INT': 'halftime', 'BRK': 'stop', 'BUILD': 'blast'}


def compose_form(model, vocab, stoi, prompt, plan, temperature, top_k, allowed, rng, structure,
                 gen='mix', composer=None, scorer=None, pick=1, root_k=0, tonic=0, riff=0.8, vary=0.5,
                 palette=None, riffs=None, riff_mix=None, nn_share=0.4, chosen=None):
    """Build a song section by section from a form plan [(role, bars)].
    Each role gets a 2- or 4-bar riff unit (NN or prefab) repeated to fill its section; returning roles
    reuse their unit (last bar sometimes varied); the last chorus is the 'emphasized' version (power
    chords + guitar 2 an octave up); INT = sustained chords, BRK = held hit, BUILD = pedal 16ths."""
    masks = build_masks(vocab, allowed)
    ids = [stoi[t] for t in prompt]
    units, body = {}, []
    drums, octave, markers, desc = {}, set(), {}, []
    last_ch = max((i for i, (r, _) in enumerate(plan) if r == 'CH'), default=-1)
    roots = structure.get('roots') if structure else None

    def push(part):
        ids.extend(stoi[t] for t in part if t in stoi)

    unit_ev = {}
    slots = []                                            # bars the NN may rewrite (--nn-edit)

    def nn_riff_ev():
        """An NN riff made like riff_sampler does: up to 10 tries reading the song so far, each tidied
        (16th grid, short rests filled, octave fold); bars with holes are dropped, the darkest / most
        rhythmic one is kept."""
        import riff_sampler as RS
        best = None
        for _ in range(10):
            trial = list(ids)
            toks = sample_riff(model, trial, 2, vocab, stoi, masks, temperature, top_k, rng, 0.8, palette=palette)
            ev = RF.clean(RS.tokens_to_events(toks))
            if not RS.ok_rhythm(ev):
                continue
            sc = RF.darkness(ev) + RS.rhythm_score(ev)
            if best is None or sc > best[0]:
                best = (sc, ev)
        return best[1] if best else None

    def riff_unit(n_bars, role, key=None):
        ev = None
        if chosen and key in chosen:                       # song_builder: the riff you picked for this role
            ev, d = chosen[key]
        elif gen == 'nn' or (gen == 'riffmix' and rng.random() < nn_share):
            ev = nn_riff_ev()
            d = 'NN'
        if ev is None:
            meth = rng.choices(list(riff_mix), weights=list(riff_mix.values()))[0]
            ev, d = riffs.make(meth, role)
        bars = RF.ev_to_bars(ev)
        if n_bars >= 4:
            bars += RF.ev_to_bars(riffs.vary_end(ev))         # A A' (only the ending changes)
        part = []
        for b in bars[:max(1, n_bars)]:
            part += ['new_measure'] + b
        return part, d, ev

    def new_unit(n_bars):
        src = gen if gen != 'mix' else rng.choice(['nn', 'prefab'])
        if composer is None:
            src = 'nn'
        tries = max(1, pick if scorer is not None else 1)
        cands = []
        for _ in range(tries):
            trial = list(ids)
            if src == 'prefab':
                plan_r = RootGuide(roots, n_bars, rng, 0, 1).plan if roots else [0] * n_bars
                part, _ = composer.phrase(n_bars, RIFF_FORMS, plan_r)
            else:
                guide = RootGuide(roots, n_bars, rng, tonic, root_k) if (roots and root_k > 0) else None
                part = sample_riff(model, trial, n_bars, vocab, stoi, masks, temperature, top_k, rng, riff, guide=guide,
                                   palette=palette)
            cands.append(part)
        part = max(cands, key=lambda c: scorer(c) or -9) if len(cands) > 1 else cands[0]
        return part, src

    def chords_section(n):
        plan_r = RootGuide(roots, n, rng, 0, 1).plan if roots else [0, 8, 6, 1] * n
        out = []
        for k in plan_r[:n]:
            out += ['new_measure', f'distorted0:note:s6:f{k - 2}', f'distorted0:note:s5:f{k}',
                    f'distorted0:note:s4:f{k}', f'bass:note:s5:f{k - 2}', 'wait:3840']
        return out

    bar = 0
    for si, (r, n) in enumerate(plan):
        markers[bar] = f'{r}{n}'
        if r == 'INT':
            part, how = chords_section(n), 'chords'
        elif r == 'BRK':
            part = ['new_measure', 'distorted0:note:s6:f-2', 'distorted0:note:s5:f0', 'distorted0:note:s4:f0',
                    'bass:note:s5:f-2', 'wait:3840'] * 1
            part = part * n
            how = 'hit'
        elif r == 'BUILD':
            one = ['new_measure'] + ['distorted0:note:s6:f-2', 'nfx:palm_mute', 'bass:note:s5:f-2', 'wait:240'] * 16
            part, how = one * n, 'pedal'
        else:
            key = {'POST': 'CH', 'OUTRO': 'CH' if 'CH' in units else 'INTRO'}.get(r, r)
            if key in units and r != 'BR':
                unit, how = units[key], 'return'
            else:
                ub = 4 if n >= 8 else (2 if n >= 2 else 1)
                if riffs is not None:
                    unit, src, unit_ev[key] = riff_unit(ub, r, key)
                else:
                    unit, src = new_unit(ub)
                units[key] = unit
                how = src
            ub = sum(1 for t in unit if t == 'new_measure')
            bars_u = DV._bars(unit)
            seq = [bars_u[k % ub] for k in range(n)]
            if riffs is not None:
                slots.extend(bar + k for k in range(n) if ub >= 4 and k % ub == ub - 1)
            if how == 'return' and rng.random() < vary and n >= 2 and key in unit_ev:
                seq[-1] = RF.ev_to_bars(riffs.vary_end(unit_ev[key]))[1]
                how += '+varied'
                slots.append(bar + n - 1)
            elif how == 'return' and rng.random() < vary and n >= 2:   # vary the last bar on a return
                guide = None
                seq[-1] = new_bar(model, list(ids), vocab, stoi, masks, temperature, top_k, guide=guide,
                                  palette=palette)[1:]
            part = []
            for b in seq:
                part += ['new_measure'] + b
            if r == 'CH' and si == last_ch:
                part = DV.emphasize(part); how += '+emphasized'
                octave.update(range(bar, bar + n))
        if r in FORM_DRUMS:
            for k in range(n):
                drums[bar + k] = FORM_DRUMS[r]
        if r == 'BUILD':
            for k in range(n):
                drums[bar + k] = 'double' if k < n // 2 else 'blast'
        push(part)
        body += part
        desc.append(f'{r}{n}({how})')
        bar += n
    return body, desc, dict(drums=drums, octave=sorted(octave), markers=markers, slots=sorted(set(slots)))


GSTR = {'s6': 2, 's5': 7, 's4': 12, 's3': 17, 's2': 21, 's1': 26}     # token fret -> rel pitch offset


def bar_lows(bar):
    """Lowest guitar pitch (relative to the open low string) of each onset in a bar's tokens."""
    lows, cur = [], []
    for t in bar:
        if t.startswith('distorted0:note:'):
            _, _, s, f = t.split(':')
            if s in GSTR:
                cur.append(int(f[1:]) + GSTR[s])
        elif t.startswith('wait:') and cur:
            lows.append(min(cur)); cur = []
    if cur:
        lows.append(min(cur))
    return lows


def nn_edit(model, vocab, stoi, prompt, body, slots, p, rng, temperature, top_k, allowed, palette=None, tries=3):
    """Let the NN rewrite some bars (the endings of riff units): it reads the song so far and writes the
    next bar; of `tries` bars the one that is darkest and closest in note count to the original is kept."""
    masks = build_masks(vocab, allowed)
    starts = [i for i, t in enumerate(body) if t == 'new_measure'] + [len(body)]
    bars = [body[a:b] for a, b in zip(starts, starts[1:])]
    done = []
    for s in slots:
        if s >= len(bars) or rng.random() >= p:
            continue
        ctx = [stoi[t] for t in prompt + [t for b in bars[:s] for t in b] if t in stoi]
        n0 = len(bar_lows(bars[s]))
        best = None
        for _ in range(tries):
            trial = list(ctx)
            new = new_bar(model, trial, vocab, stoi, masks, temperature, top_k, palette=palette)
            lows = bar_lows(new)
            if not lows:
                continue
            pcs = [x % 12 for x in lows if x % 12]
            dark = sum((q in RF.DARK) - (q in RF.BRIGHT) for q in pcs) / len(pcs) if pcs else 0
            sc = dark - abs(len(lows) - n0) / max(4, n0)
            if best is None or sc > best[0]:
                best = (sc, new)
        if best is not None:
            bars[s] = best[1]
            done.append(s + 1)
    return [t for b in bars for t in b], done


def compose(model, vocab, stoi, prompt, target_bars, temperature, top_k, allowed, rng,
            scorer=None, pick=1, rules=None, strength=0.0, riff=1.0):
    """Phrases of 1-4 bars, each repeated 2-4 times (last repeat sometimes varied).
    With a scorer (taste model) and pick > 1, each phrase is the best of `pick` tries."""
    masks = build_masks(vocab, allowed)
    ids = [stoi[t] for t in prompt]
    body, plan = [], []
    while len(body) == 0 or sum(1 for t in body if t == 'new_measure') < target_bars:
        P = rng.choice([1, 2, 2, 4, 4])
        R = rng.choice([2, 2, 4]) if P <= 2 else rng.choice([2, 2, 3])
        if scorer is not None and pick > 1:
            best = None
            for _ in range(pick):
                trial_ids = list(ids)
                cand = sample_riff(model, trial_ids, P, vocab, stoi, masks, temperature, top_k, rng, riff,
                                   rules=rules, strength=strength)
                s = scorer(cand)
                if best is None or (s is not None and s > best[0]):
                    best = (s if s is not None else -9, cand, trial_ids)
            phrase, ids[:] = best[1], best[2]
        else:
            phrase = sample_riff(model, ids, P, vocab, stoi, masks, temperature, top_k, rng, riff,
                                 rules=rules, strength=strength)
        body += phrase
        for r in range(1, R):
            last = r == R - 1 and rng.random() < 0.5
            keep = phrase if not last else phrase[:_bar_starts(phrase)[-1]] if P > 1 else []
            body += keep
            ids += [stoi[t] for t in keep]
            if last:
                body += sample_bars(model, ids, 1, vocab, stoi, masks, temperature, top_k,
                                    rules=rules, strength=strength)
        plan.append(f'{P} bars x{R}')
    return body, plan


def _bar_starts(toks):
    return [i for i, t in enumerate(toks) if t == 'new_measure']


GP_LENGTHS = [3840, 2880, 1920, 1440, 960, 720, 480, 360, 240, 120]


def normalize_bar(bar):
    """Make a bar writable in GP5. Each instrument's note lasts until its own next event, so a note
    sustained over several waits can get a length GP5 has no value for (e.g. 1200 ticks) and the bar
    comes out too long. Per instrument: snap onsets to 16ths if they are tuplet-like, and split
    such lengths with tied copies of the note."""
    t, events = 0, []                                  # (onset, instrument, tokens)
    cur_inst = None
    for x in bar:
        if x.startswith('wait:'):
            t += int(x[5:]); cur_inst = None; continue
        if x.startswith(('nfx:', 'bfx:', 'param:')):
            if events and cur_inst:
                events[-1][2].append(x)
            continue
        inst = x.split(':')[0]
        if events and events[-1][0] == t and events[-1][1] == inst:
            events[-1][2].append(x)
        else:
            events.append([t, inst, [x]])
        cur_inst = inst
    total = t
    if total != 3840:
        return bar
    snap = any(e[0] % 120 for e in events)
    by = {}
    for on, inst, toks in events:
        on = int(round(on / 240)) * 240 if snap else on
        if on >= 3840:
            continue
        d = by.setdefault(inst, {})
        d.setdefault(on, [])
        d[on] += [x for x in toks if x not in d[on]]
    for inst, d in by.items():                         # leading silence: rests of writable lengths
        ons = sorted(d)
        if ons and ons[0] > 0 and inst != 'drums':
            pos, L = 0, ons[0]
            while L > 0:
                c = next(x for x in GP_LENGTHS if x <= L)
                d[pos] = [f'{inst}:rest']; pos += c; L -= c
    for inst, d in by.items():                         # split unwritable lengths with ties
        ons = sorted(d)
        for i, on in enumerate(ons):
            end = ons[i + 1] if i + 1 < len(ons) else 3840
            L, pos = end - on, on
            notes = [x for x in d[on] if ':note:' in x]
            rest = any(x.endswith(':rest') for x in d[on])
            first = True
            while L > 0:
                c = next(x for x in GP_LENGTHS if x <= L)
                if not first and notes:
                    d[pos] = sum(([n, 'nfx:tie'] for n in notes), [])
                elif not first and rest:
                    d[pos] = [f'{inst}:rest']
                pos += c; L -= c; first = False
    onsets = sorted({on for d in by.values() for on in d})
    if not onsets or onsets[0] != 0:
        onsets = [0] + onsets
    out = []
    order = ['distorted0', 'distorted1', 'distorted2', 'bass', 'drums']
    for i, on in enumerate(onsets):
        for inst in sorted(by, key=lambda k: order.index(k) if k in order else 9):
            out += by[inst].get(on, [])
        nxt = onsets[i + 1] if i + 1 < len(onsets) else 3840
        gap = nxt - on
        while gap > 0:                                  # waits themselves must be writable too
            c = next(x for x in GP_LENGTHS if x <= gap)
            out.append(f'wait:{c}'); gap -= c
    return out


def repair(tokens, tempo, complete=False):
    """Make a sampled sequence decodable: header, 'start', whole measures, 'end'."""
    if 'start' not in tokens:
        return None
    s = tokens.index('start')
    body = tokens[s + 1:]
    # effects must follow the note they belong to
    clean, prev = [], ''
    for t in body:
        if t.startswith(('nfx:', 'bfx:')) and not (':note:' in prev or prev.startswith(('nfx:', 'bfx:'))):
            continue
        clean.append(t); prev = t
    body = [t for t in clean if not t.startswith('drums:')]
    if not body or body[0] != 'new_measure':
        body = ['new_measure'] + body
    if 'end' in body:
        body = body[:body.index('end')]
    # cut after the last complete measure
    if not complete:
        last = max((i for i, t in enumerate(body) if t == 'new_measure'), default=-1)
        if last <= 0:
            return None
        body = body[:last]
    bars, cur = [], None                           # GP5-safe bar lengths
    for t in body:
        if t == 'new_measure':
            if cur is not None:
                bars.append(cur)
            cur = []
        elif cur is not None:
            cur.append(t)
    if cur is not None:
        bars.append(cur)
    body = []
    for b in bars:
        body += ['new_measure'] + normalize_bar(b)
    header = [t for t in tokens[:s] if not t.startswith('tempo:')]
    header = (header[:2] if len(header) >= 2 else ['unknown', 'downtune:0'])
    return header + [f'tempo:{tempo}', 'start'] + body + ['end']


def copy_stats(gen, refs, n=COPY_N):
    grams = set()
    for r in refs:
        grams.update(tuple(r[i:i + n]) for i in range(len(r) - n + 1))
    body = gen[gen.index('start') + 1:]
    hits = [tuple(body[i:i + n]) in grams for i in range(max(0, len(body) - n + 1))]
    longest = run = 0
    for h in hits:
        run = run + 1 if h else 0
        longest = max(longest, run)
    share = sum(hits) / len(hits) if hits else 0.0
    return share, longest + (n - 1 if longest else 0)


def render(toks, out, name, py, **finish_kw):
    """tokens -> .gp5 (DadaGP) -> fixed tunings, double guitar, drums -> .mid. Returns track names or None."""
    tok_path = out / f'{name}.tokens.txt'
    tok_path.write_text('\n'.join(toks) + '\n')
    gp5 = out / f'{name}.gp5'
    try:
        r = subprocess.run([py, 'dadagp.py', 'decode', str(tok_path), str(gp5)],
                           cwd=HERE / 'dadaGP', capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0 or not gp5.exists():
        (out / f'{name}.decode_error.txt').write_text(r.stderr[-2000:])
        return None
    return P.finish(gp5, out / f'{name}.mid', **finish_kw)


def make_scorer(prompt, tempo, py, model_path):
    """Taste-model score of a phrase (played twice), or None if there is no taste model yet."""
    import tempfile
    from taste import features as TF
    from taste import model as TM
    if not Path(model_path).exists():
        return None
    taste = TM.load(model_path)
    tmp = Path(tempfile.mkdtemp())
    counter = [0]

    def score(phrase):
        counter[0] += 1
        toks = repair(prompt + phrase + phrase, tempo, complete=True)
        if render(toks, tmp, f'p{counter[0]}', py) is None:
            return None
        return TM.predict(taste, TF.extract(tmp / f'p{counter[0]}.mid'))
    return score


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=5, help='songs to generate')
    ap.add_argument('--tokens', type=int, default=3000, help='length in tokens (~3000 = 1-2 minutes)')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-k', type=int, default=40)
    ap.add_argument('--tempo', type=int, default=150)
    ap.add_argument('--downtune', type=int, default=-2, help='-2 = drop G 7-string style refs')
    ap.add_argument('--max-copy', type=float, default=0.5,
                    help='discard songs whose share of copied 32-token runs is above this')
    ap.add_argument('--model', default=str(DATA / 'model.pt'))
    ap.add_argument('--seed', type=int)
    ap.add_argument('--bars', type=int, default=32, help='song length in bars (constrained mode)')
    ap.add_argument('--free', action='store_true', help='old unconstrained sampling (any meter, no repeats)')
    ap.add_argument('--phrases', type=int, default=0, metavar='N',
                    help='make N short phrases (2 or 4 bars, played twice) to rate quickly, instead of songs')
    ap.add_argument('--form', nargs='?', const='', default=None, metavar='TEMPLATE',
                    help='songs follow a form template from nn/data/forms.json (form.py mine); give part of a '
                         'template name or nothing for a random one')
    ap.add_argument('--palette', help="note-choice palette (part of a song name, see palette.py list); default: the "
                         "--form template's song; 'none' = off")
    ap.add_argument('--palette-strength', type=float, default=1.0)
    ap.add_argument('--freedom', type=float, default=0.2, help='with --form: share of sections the statistics may change')
    ap.add_argument('--scale', type=float, default=0.75, help='with --form: scale section lengths (1 = as long as the reference song)')
    ap.add_argument('--sections', type=int, default=0, metavar='N',
                    help='make N mid-length excerpts (16-24 bars, 2+ sections with a change or a return) to rate: '
                         'riffs + transitions in ~30-40 s. Sections are NN/prefab mixed unless --gen is given')
    ap.add_argument('--pick', type=int, default=None,
                    help='songs: try this many phrases per slot and keep the one the taste model likes best '
                         '(needs outputs/taste_model.json; 1 = off)')
    ap.add_argument('--gen', choices=['nn', 'prefab', 'mix', 'riff', 'riffmix'], default=None,
                    help='songs: who writes each new section - the NN, the prefab composer, a random mix of those, '
                         'riff = riffs rebuilt from the reference riffs (nn/riffs.py; default for --form), or '
                         'riffmix = per section NN (share --nn-share) or rebuilt. With --form, nn and riffmix make '
                         'NN riffs the riff_sampler way (10 tries, tidied, holes dropped)')
    ap.add_argument('--nn-share', type=float, default=0.4, help='--gen riffmix: share of sections written by the NN')
    ap.add_argument('--riff-mix', default='A:1,B:1,C:1',
                    help='--gen riff: weights of the three riff makers (A rhythm-first, B motif, C mutate)')
    ap.add_argument('--nn-edit', type=float, default=0.0,
                    help='--gen riff: chance that the NN rewrites each riff ending (the 4th bar of a riff, the varied '
                         'last bar of a returning section); the song before the edit is saved as cand_XX_before')
    ap.add_argument('--riff-dark', type=float, default=2.0,
                    help='--gen riff: prefer dark riffs (b2 b3 b5 b6 b7) over bright ones (3 6 7); 0 = off')
    ap.add_argument('--riff-exclude', default='',
                    help='--gen riff: comma-separated song names not used as riff material, e.g. Blooddrunk,Tartarus')
    ap.add_argument('--prefab', type=float, default=0.5,
                    help='phrases: share made by the prefab composer instead of the NN (nn/prefab.py; 0 = NN only). '
                         'They are shuffled in, so you rate them blind')
    ap.add_argument('--roots', type=int, default=4,
                    help='follow a root progression learned by nn/roots.py: try up to this many takes per bar '
                         'to match the planned root (0 = off; higher = closer but slower)')
    ap.add_argument('--develop', type=float, default=0.4,
                    help='chance that a returning section is developed (transpose / halftime / retrograde / '
                         'double time) instead of repeated exactly; 0 = exact repeats')
    ap.add_argument('--no-structure', action='store_true', help='ignore nn/data/structure.json')
    ap.add_argument('--riff', type=float, default=0.8,
                    help='chance that a phrase is built from repeated bars like real riffs (A A A B etc.); 0 = off')
    ap.add_argument('--rules', type=float, default=0.0,
                    help='strength of the taste rules (fewer rests/triplets, more chords, bass with guitar); 0 = off')
    a = ap.parse_args()

    if a.seed is not None:
        torch.manual_seed(a.seed); random.seed(a.seed)
    rng = random.Random(a.seed)
    model, vocab = load_model(a.model)
    stoi = {t: i for i, t in enumerate(vocab)}
    refs = [[t for t in p.read_text().split() if not t.startswith('drums:')]
            for p in sorted((DATA / 'tokens').glob('*.txt'))]
    tempos = [int(t.split(':')[1]) for t in vocab if t.startswith('tempo:')]
    near = min(tempos, key=lambda x: abs(x - a.tempo)) if tempos else None
    prompt = ['unknown', f'downtune:{a.downtune}'] + ([f'tempo:{near}'] if near else []) + ['start']
    prompt = [t for t in prompt if t in stoi] or ['start']

    py = dadagp_python()
    allowed = {'distorted0', 'bass'}
    rules = build_rules(vocab)
    structure = None if a.no_structure else load_structure()
    home = home_root(DATA_TOKENS) if structure and 'roots' in structure and a.roots > 0 else 0
    if a.gen is None:
        a.gen = 'riff' if (a.form is not None and RF.OUT.exists()) else ('mix' if (a.sections or a.form is not None) else 'nn')
    if a.pick is None:
        a.pick = 1 if (a.sections or a.form is not None) else 4          # sections: no picking by default (rate them as they come)
    if a.sections:
        a.n = a.sections
    song_composer = None
    if a.gen != 'nn' and not a.phrases:
        import prefab as PF
        pm = PF.load()
        song_composer = PF.PrefabComposer(pm, rng, home_root(DATA_TOKENS)) if pm else None
        print('sections made by: ' + (a.gen if song_composer else 'nn (no prefabs.json: run nn/prefab.py mine)'))
    print('song form: ' + ('learned from refs (nn/data/structure.json)' if structure else
                           'simple phrase repeats (run nn/structure.py to learn forms)'))
    print(f'riff repeats: {a.riff:g}   rules: {a.rules:g}' + (' (off)' if a.rules <= 0 else ''))
    if a.phrases:
        run_id = time.strftime('%Y%m%d-%H%M%S') + '_nnphrase'
        out = V1 / 'outputs' / run_id
        out.mkdir(parents=True, exist_ok=True)
        masks = build_masks(vocab, allowed)
        rows = []
        import prefab as PF
        pf_model = PF.load() if a.prefab > 0 else None
        if a.prefab > 0 and pf_model is None:
            print('no nn/data/prefabs.json (run: python3 nn/prefab.py mine) - NN phrases only')
        composer = PF.PrefabComposer(pf_model, rng, home_root(DATA_TOKENS)) if pf_model else None
        # decide up front which candidates are prefab ones, so the numbering gives nothing away
        n_pf = round(a.phrases * a.prefab) if composer else 0
        kinds = ['prefab'] * n_pf + ['nn'] * (a.phrases - n_pf)
        rng.shuffle(kinds)
        for k, kind in enumerate(kinds, 1):
            P = rng.choice([2, 4])
            if kind == 'prefab':
                plan = (RootGuide(structure['roots'], P, rng, 0, 1).plan
                        if structure and 'roots' in structure else [0] * P)
                phrase, form = composer.phrase(P, RIFF_FORMS, plan)
                desc = f'form {form} roots {" ".join(RT_NAMES[x] for x in plan)}'
            else:
                ids = [stoi[t] for t in prompt]
                phrase = sample_riff(model, ids, P, vocab, stoi, masks, a.temperature, a.top_k, rng, a.riff,
                                     rules=rules, strength=a.rules)
                desc = f'riff {a.riff:g} rules {a.rules:g}'
            toks = repair(prompt + phrase + phrase, a.tempo, complete=True)
            name = f'cand_{k:02d}'
            if render(toks, out, name, py) is None:
                print(f'  {name}: decode failed'); continue
            typ = 'prefab_phrase' if kind == 'prefab' else 'nn_phrase'
            (out / f'{name}.txt').write_text(f'{name}  type={typ}  {P} bars x2  tempo {a.tempo}  {desc}\n')
            (out / f'{name}.phrase.txt').write_text('\n'.join(phrase) + '\n')
            rows.append(dict(run_id=run_id, candidate=name, type=typ, seed='', rating='', note=desc))
            print(f'{name}: {P}-bar phrase')
        from taste.common import read_ratings, write_ratings
        write_ratings(read_ratings() + rows)
        print(f'saved {len(rows)} phrases to {out}\nrate them: python3 {V1 / "taste" / "rate.py"} --run {run_id}')
        return

    from taste.common import MODEL as TASTE_MODEL
    scorer = make_scorer(prompt, a.tempo, py, TASTE_MODEL) if a.pick > 1 else None
    if a.pick > 1:
        print('taste model: ' + ('picking the best of %d phrases per slot' % a.pick if scorer
                                 else 'none yet (rate some phrases, run taste/train.py) - no picking'))
    run_id = time.strftime('%Y%m%d-%H%M%S') + ('_nnsection' if a.sections else '_nn')
    out = V1 / 'outputs' / run_id
    out.mkdir(parents=True, exist_ok=True)
    rows, k, tries = [], 0, 0
    kind = 'nn_section' if a.sections else ('nn_form' if a.form is not None else 'nn')
    while k < a.n and tries < a.n * 4:
        tries += 1
        plan, before = [], None
        if a.free:
            idx = torch.tensor([[stoi[t] for t in prompt]])
            seq = model.generate(idx, a.tokens, a.temperature, a.top_k, stop=stoi.get('end'),
                                 min_len=int(a.tokens * 0.8))[0].tolist()
            toks = repair([vocab[i] for i in seq], a.tempo)
        elif a.form is not None:
            import form as FM
            fm = FM.load()
            if fm is None:
                sys.exit('run: python3 nn/form.py mine')
            fplan, fnotes = FM.sample(fm, a.form or None, a.freedom, rng, scale=a.scale)
            pal = make_palette(a, vocab, home, fnotes[0].split(': ', 1)[1] if fnotes else None)
            if pal is not None:
                fnotes.append(f'palette: {pal.name}')
                if song_composer is not None:
                    song_composer.palette = pal
            rb = None
            if a.gen in ('riff', 'riffmix', 'nn'):
                if not RF.OUT.exists():
                    sys.exit('run: python3 nn/riffs.py mine')
                want = (a.palette if a.palette and a.palette != 'none' else None) or \
                       (fnotes[0].split(': ', 1)[1] if fnotes else None)
                rb = RF.Riffs(song=want, rng=rng, dark=a.riff_dark,
                              exclude=[x for x in a.riff_exclude.split(',') if x])
                fnotes.append(f'riff palette: {rb.song}')
                mix = {k: float(v) for k, v in (x.split(':') for x in a.riff_mix.split(','))}
            body, plan, fin = compose_form(model, vocab, stoi, prompt, fplan, a.temperature, a.top_k, allowed, rng,
                                           structure, gen=a.gen, composer=song_composer, scorer=scorer, pick=a.pick,
                                           root_k=a.roots, tonic=home, riff=a.riff, palette=pal,
                                           riffs=rb, riff_mix=mix if rb else None, nn_share=a.nn_share)
            before = None
            if rb is not None and a.nn_edit > 0:
                before = repair(prompt + body, a.tempo, complete=True)
                body, edited = nn_edit(model, vocab, stoi, prompt, body, fin['slots'], a.nn_edit, rng,
                                       a.temperature, a.top_k, allowed, palette=pal)
                plan.append('nn-edit bars: ' + (' '.join(map(str, edited)) or 'none'))
                if not edited:
                    before = None
            plan = fnotes + plan
            toks = repair(prompt + body, a.tempo, complete=True)
        elif structure is not None:
            body, plan = compose_structured(model, vocab, stoi, prompt, a.bars, a.temperature, a.top_k, allowed,
                                            rng, structure, scorer=scorer, pick=a.pick, rules=rules,
                                            strength=a.rules, riff=a.riff, develop_p=a.develop, root_k=a.roots,
                                            tonic=home, gen=a.gen, composer=song_composer,
                                            window=(16, 24) if a.sections else None)
            toks = repair(prompt + body, a.tempo, complete=True)
        else:
            body, plan = compose(model, vocab, stoi, prompt, a.bars, a.temperature, a.top_k, allowed, rng,
                                 scorer=scorer, pick=a.pick, rules=rules, strength=a.rules, riff=a.riff)
            toks = repair(prompt + body, a.tempo, complete=True)
        if toks is None:
            continue
        share, longest = copy_stats(toks, refs)
        if share > a.max_copy:
            print(f'  discarded: {share:.0%} copied from the references')
            continue
        k += 1
        name = f'cand_{k:02d}'
        info = render(toks, out, name, py, **({'drum_overrides': fin['drums'], 'octave_bars': fin['octave'],
                                               'markers': fin['markers']} if a.form is not None else {}))
        if info is None:
            print(f'  {name}: decode failed, skipped'); k -= 1
            continue
        if a.form is not None and before is not None:
            render(before, out, name + '_before', py, drum_overrides=fin['drums'], octave_bars=fin['octave'],
                   markers=fin['markers'])
        measures = toks.count('new_measure')
        (out / f'{name}.txt').write_text(
            f'{name}  type={kind}  {measures} bars  tempo prompt {a.tempo}\n'
            f'copied from references: {share:.0%} of 32-token runs, longest copy {longest} tokens\n'
            f'tracks: {info}\nphrases: {", ".join(plan)}\n')
        print(f'{name}: {measures} bars, copied {share:.0%} (longest {longest} tokens), tracks {info}')
        note = f'gen={a.gen}' + (f' {plan[0]}' if a.form is not None and plan else '')
        rows.append(dict(run_id=run_id, candidate=name, type=kind, seed='', rating='', note=note))

    from taste.common import read_ratings, write_ratings
    write_ratings(read_ratings() + rows)
    print(f'saved {len(rows)} songs to {out}\nlisten and rate: python3 {V1 / "taste" / "rate.py"} --run {run_id}')


if __name__ == '__main__':
    main()
