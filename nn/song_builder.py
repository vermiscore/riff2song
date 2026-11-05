"""Build a song from riffs you choose: the form comes from a reference song, you pick each role's riff.

    python3 nn/song_builder.py new --form Memento          # 1. form + riff candidates per role (mp3)
    python3 nn/song_builder.py choose INTRO=p2 V=3 PRE=1 CH=p5 BD=4 BR=2     # 2. your picks
    python3 nn/song_builder.py suggest                      #    (or let it suggest; --apply to take it)
    python3 nn/song_builder.py build                        # 3. the song (gp5 + mp3)

new: decides the form (like generate.py --form) and writes, in outputs/<time>_song/:
  picks_01.mp3 ...     every riff of your picks.gp5 (one per marker; first 2 bars), usable for any role
  CH_01.mp3 ...        new candidates per role (--per-role, NN and rebuilt riffs mixed; BD = heavier ones)
  list.txt             what is what, and the roles you have to choose
choose: ROLE=number (a new candidate of that role) or ROLE=pN (picks_N). Works on the newest session
  (--session to pick another). You can run it again to change your mind.
build: puts the song together with the same rules as generate.py --form (repeats, varied endings on
  returns, last chorus emphasized, chords in INT, held hit in BRK, drums), writes song.gp5 / song.mid /
  song.mp3, and records the new candidates you chose as rated 5 (they feed finetune / taste).
"""
import argparse
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

import guitarpro as gp            # noqa: E402
import riffs as RF                # noqa: E402
import riff_sampler as RS         # noqa: E402

PICKS = V1.parent / 'tabs_picks' / 'picks.gp5'
NO_RIFF = {'INT', 'BRK', 'BUILD'}


def role_keys(plan):
    """The roles that need their own riff, in order (same mapping as generate.compose_form)."""
    keys, seen = [], set()
    for r, n in plan:
        if r in NO_RIFF:
            continue
        key = {'POST': 'CH', 'OUTRO': 'CH' if 'CH' in seen else 'INTRO'}.get(r, r)
        seen.add(key)
        if key not in keys:
            keys.append(key)
    return keys


def picks_riffs(path):
    """One riff per marker of picks.gp5: the first 2 bars with notes from the marker on."""
    if not Path(path).exists():
        return []
    song = gp.parse(str(path))
    bars = RS.gp_events(path)
    marks = [i for i, h in enumerate(song.measureHeaders) if h.marker]
    if not marks:
        marks = [0]
    out = []
    for k, m in enumerate(marks):
        end = marks[k + 1] if k + 1 < len(marks) else len(bars)
        part = [bars[i] for i in range(m, end) if any(e[2] for e in bars[i])][:2]
        if not part:
            continue
        if len(part) == 1:
            part = part * 2
        ev = part[0] + [[on + RS.BAR, d, ps, pm] for on, d, ps, pm in part[1]]
        title = song.measureHeaders[m].marker.title if song.measureHeaders[m].marker else f'bar {m + 1}'
        out.append(dict(name=f'picks_{len(out) + 1:02d}', desc=f'picks: {title} (bar {m + 1})', kind='picks', ev=ev))
    return out


def sessions():
    return sorted(p for p in (V1 / 'outputs').glob('*_song') if (p / 'session.json').exists())


def load_session(a):
    s = (V1 / 'outputs' / a.session) if a.session else (sessions()[-1] if sessions() else None)
    if s is None or not (s / 'session.json').exists():
        sys.exit('no song_builder session: run  python3 nn/song_builder.py new --form Memento')
    return s, json.load(open(s / 'session.json'))


def save_audio(item, out, tempo, sf2):
    mid = out / f"{item['name']}.mid"
    RS.write_midi(item['ev'], mid, tempo)
    if sf2:
        RS.to_mp3(mid, out / f"{item['name']}.mp3", sf2)
    bars = RF.ev_to_bars(item['ev'])
    toks = ['start'] + [t for _ in range(2) for b in bars for t in ['new_measure'] + b] + ['end']
    (out / f"{item['name']}.tokens.txt").write_text('\n'.join(toks) + '\n')


def cmd_new(a):
    import form as FM
    rng = random.Random(a.seed)
    fm = FM.load()
    if fm is None:
        sys.exit('run: python3 nn/form.py mine')
    plan, notes = FM.sample(fm, a.form or None, a.freedom, rng, scale=a.scale)
    template = notes[0].split(': ', 1)[1] if notes else None
    keys = role_keys(plan)
    out = V1 / 'outputs' / (time.strftime('%Y%m%d-%H%M%S') + '_song')
    out.mkdir(parents=True)
    sf2 = next((p for p in RS.SF2S if Path(p).exists()), None)

    picks = picks_riffs(a.picks)
    for it in picks:
        save_audio(it, out, a.tempo, sf2)
    maker = RS.Maker(rng)
    cands = {}
    for key in keys:
        cands[key] = []
        for i in range(a.per_role):
            nn = rng.random() < a.nn_share
            if nn:
                ev, d = maker.nn_riff(a)
                kind = 'riff' if '(NN failed)' in d else 'nn'
            elif key == 'BD':                                # breakdowns: fewer, heavier hits
                maker.R.song = rng.choice(maker.R.big_songs())
                ev, d = maker.R.make(rng.choice('ABC'), 'BD')
                kind = 'riff'
            else:
                ev, d = maker.riff()
                kind = 'riff'
            it = dict(name=f'{key}_{i + 1:02d}', desc=d, kind=kind, ev=ev)
            save_audio(it, out, a.tempo, sf2)
            cands[key].append(it)
            print(f"{it['name']:9s} {kind:4s} {d}")
    sess = dict(plan=plan, notes=notes, template=template, tempo=a.tempo, seed=a.seed, keys=keys,
                picks=picks, cands=cands, chosen={})
    json.dump(sess, open(out / 'session.json', 'w'))
    lines = ['form: ' + ' '.join(f'{r}{n}' for r, n in plan), 'roles to choose: ' + ' '.join(keys), '']
    lines += [f"{it['name']:9s} {it['desc']}" for it in picks]
    lines += [f"{it['name']:9s} {it['kind']:4s} {it['desc']}" for k in keys for it in cands[k]]
    (out / 'list.txt').write_text('\n'.join(lines) + '\n')
    example = ' '.join(f'{k}={1 + (i % a.per_role)}' for i, k in enumerate(keys)) if a.per_role else ''
    print(f"\nform: {' '.join(f'{r}{n}' for r, n in plan)}\nroles: {' '.join(keys)}   "
          f"({len(picks)} picks riffs usable for any role)\nfiles: {out}\n"
          f"next:  python3 {HERE / 'song_builder.py'} choose {example}")


def cmd_choose(a):
    s, sess = load_session(a)
    chosen = dict(sess.get('chosen', {}))
    for arg in a.picks_args:
        if '=' not in arg:
            sys.exit(f'use ROLE=number or ROLE=pN, got {arg}')
        k, v = arg.split('=', 1)
        k = k.upper()
        if k not in sess['keys']:
            sys.exit(f"{k} is not a role of this song ({' '.join(sess['keys'])})")
        v = v.lower().replace('picks_', 'p')
        if v.startswith('p'):
            n = int(v[1:])
            if not 1 <= n <= len(sess['picks']):
                sys.exit(f'there are {len(sess["picks"])} picks riffs')
            chosen[k] = f'picks_{n:02d}'
        else:
            n = int(v)
            if not 1 <= n <= len(sess['cands'][k]):
                sys.exit(f'{k} has {len(sess["cands"][k])} candidates')
            chosen[k] = f'{k}_{n:02d}'
    sess['chosen'] = chosen
    json.dump(sess, open(s / 'session.json', 'w'))
    missing = [k for k in sess['keys'] if k not in chosen]
    print('chosen: ' + ', '.join(f'{k}={v}' for k, v in chosen.items()))
    print(f"still to choose: {' '.join(missing)}" if missing else
          f"all set:  python3 {HERE / 'song_builder.py'} build")


def cmd_build(a):
    import generate as G
    s, sess = load_session(a)
    missing = [k for k in sess['keys'] if k not in sess['chosen']]
    if missing:
        sys.exit(f"choose first: {' '.join(missing)}")
    items = {it['name']: it for it in sess['picks']}
    for k in sess['keys']:
        for it in sess['cands'][k]:
            items[it['name']] = it
    chosen = {k: (items[v]['ev'], f"chosen {v}") for k, v in sess['chosen'].items()}
    rng = random.Random(sess.get('seed'))
    model, vocab = G.load_model(a.model)
    stoi = {t: i for i, t in enumerate(vocab)}
    tempos = [int(t.split(':')[1]) for t in vocab if t.startswith('tempo:')]
    near = min(tempos, key=lambda x: abs(x - sess['tempo'])) if tempos else None
    prompt = [t for t in ['unknown', 'downtune:-2'] + ([f'tempo:{near}'] if near else []) + ['start'] if t in stoi]
    rb = RF.Riffs(song=sess.get('template'), rng=rng)
    plan = [tuple(x) for x in sess['plan']]
    body, desc, fin = G.compose_form(model, vocab, stoi, prompt, plan, 1.0, 40, {'distorted0', 'bass'}, rng,
                                     G.load_structure(), gen='riff', riffs=rb, riff_mix={'A': 1, 'B': 1, 'C': 1},
                                     chosen=chosen)
    toks = G.repair(prompt + body, sess['tempo'], complete=True)
    info = G.render(toks, s, 'song', G.dadagp_python(), drum_overrides=fin['drums'], octave_bars=fin['octave'],
                    markers=fin['markers'])
    if info is None:
        sys.exit(f'decode failed (see {s})')
    sf2 = next((p for p in RS.SF2S if Path(p).exists()), None)
    if sf2:
        wav = s / 'song.wav'
        subprocess.run(['fluidsynth', '-ni', '-g', '0.7', '-F', str(wav), '-r', '44100', sf2, str(s / 'song.mid')],
                       check=True, capture_output=True)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', str(wav), '-b:a', '192k', str(s / 'song.mp3')],
                       check=True)
        wav.unlink()
    (s / 'song.txt').write_text('form: ' + ', '.join(desc) + '\nchosen: ' +
                                ', '.join(f'{k}={v}' for k, v in sess['chosen'].items()) + '\n')
    # the new candidates you chose = rated 5 (picks riffs are yours already)
    from taste.common import read_ratings, write_ratings
    rows = read_ratings()
    done = {(r['run_id'], r['candidate']) for r in rows}
    for v in set(sess['chosen'].values()):
        it = items[v]
        if it['kind'] == 'picks' or (s.name, v) in done:
            continue
        rows.append(dict(run_id=s.name, candidate=v, type='nn_riff_sample' if it['kind'] == 'nn' else 'riff_sample',
                         seed='', rating='5', note=it['desc'] + ' / chosen in song_builder'))
    write_ratings(rows)
    print(f"song: {s / 'song.gp5'}  ({toks.count('new_measure')} bars)\n      {s / 'song.mp3'}\n"
          f"metal sound:  python3 {HERE / 'render_metal.py'} {s / 'song.gp5'} --out {s / 'song_metal.wav'}")


def riff_traits(ev):
    notes = [e for e in ev if e[2]]
    lows = [e[2][0] for e in notes]
    n = max(1, len(notes))
    return dict(density=len(notes) / 32, pedal=sum(1 for p in lows if p % 12 == 0) / n,
                high=sum(1 for p in lows if p >= 12) / n, chords=sum(1 for e in notes if len(e[2]) > 1) / n,
                pitches=len(set(lows)), pm=sum(1 for e in notes if e[3]) / n, long=sum(1 for e in notes if e[1] >= 480) / n)


ROLE_FIT = {     # what each role tends to want (rough, from the reference forms)
    'CH': lambda t: 0.15 * min(t['pitches'], 8) + 0.6 * t['high'] + 0.4 * t['chords'] + 0.5 * t['long'],
    'V': lambda t: 0.8 * t['pedal'] + 0.5 * t['pm'] + 0.3 * t['density'],
    'INTRO': lambda t: 0.1 * min(t['pitches'], 8) + 0.4 * t['density'],
    'PRE': lambda t: 0.1 * min(t['pitches'], 8) + 0.4 * t['chords'] + 0.3 * t['long'],
    'BD': lambda t: 0.8 * (1 - t['density']) + 0.6 * t['pm'] + 0.5 * t['long'] + 0.4 * t['pedal'],
    'BR': lambda t: 0.15 * min(t['pitches'], 8) + 0.5 * t['high'],
}
ROLE_ORDER = ['CH', 'V', 'INTRO', 'BD', 'PRE', 'BR']


def cmd_suggest(a):
    """Suggest a riff per role: your taste model + what the role wants + a bonus for your picks;
    each riff used once. Prints the reasons and a few alternatives; --apply stores it as your choice."""
    import numpy as np
    s, sess = load_session(a)
    taste, F, TM = None, None, None
    try:
        from taste import features as F, model as TM
        from taste.common import MODEL
        taste = TM.load(MODEL) if Path(MODEL).exists() else None
    except Exception:
        taste = None
    items = list(sess['picks']) + [it for k in sess['keys'] for it in sess['cands'][k]]
    info = {}
    for it in items:
        sc = 0.0
        if taste is not None and (s / f"{it['name']}.mid").exists():
            try:
                sc = TM.predict(taste, F.extract(s / f"{it['name']}.mid")) - 3.0
            except Exception:
                sc = 0.0
        info[it['name']] = (sc, riff_traits(it['ev']), it)
    used, pick, lines = set(), {}, []
    order = [k for k in ROLE_ORDER if k in sess['keys']] + [k for k in sess['keys'] if k not in ROLE_ORDER]
    for key in order:
        fit = ROLE_FIT.get(key, ROLE_FIT['INTRO'])
        pool = [it for it in sess['picks']] + sess['cands'][key]
        ranked = []
        for it in pool:
            if it['name'] in used:
                continue
            sc, tr, _ = info[it['name']]
            total = sc + fit(tr) + (a.picks_bonus if it['kind'] == 'picks' else 0)
            ranked.append((total, sc, fit(tr), it))
        ranked.sort(key=lambda x: -x[0])
        best = ranked[0][3]
        used.add(best['name'])
        pick[key] = best['name']
        alt = ', '.join(x[3]['name'] for x in ranked[1:4])
        lines.append(f"{key:6s} {best['name']:10s} taste {ranked[0][1]:+.2f}  role fit {ranked[0][2]:.2f}  "
                     f"({best['desc'][:50]})   also try: {alt}")
    print('suggestion (listen to these first):')
    print('\n'.join(lines))
    cmd = ' '.join(f"{k}={'p' + v.split('_')[1].lstrip('0') if v.startswith('picks_') else v.split('_')[1].lstrip('0')}"
                   for k, v in pick.items())
    if a.apply:
        sess['chosen'] = pick
        json.dump(sess, open(s / 'session.json', 'w'))
        print(f"\nstored as your choice. change any:  python3 {HERE / 'song_builder.py'} choose ROLE=...\n"
              f"then:  python3 {HERE / 'song_builder.py'} build")
    else:
        print(f"\nto use it:  python3 {HERE / 'song_builder.py'} choose {cmd}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    n = sub.add_parser('new', help='form + candidates')
    n.add_argument('--form', default='Memento', help='reference song the form follows (part of its name)')
    n.add_argument('--picks', default=str(PICKS), help='your picks.gp5')
    n.add_argument('--per-role', type=int, default=5, help='new candidates per role (0 = picks only)')
    n.add_argument('--nn-share', type=float, default=0.4, help='share of new candidates written by the NN')
    n.add_argument('--freedom', type=float, default=0.2)
    n.add_argument('--scale', type=float, default=0.75)
    n.add_argument('--tempo', type=int, default=150)
    n.add_argument('--temperature', type=float, default=1.0)
    n.add_argument('--model', default=str(HERE / 'data' / 'model.pt'))
    n.add_argument('--seed', type=int)
    c = sub.add_parser('choose', help='ROLE=number or ROLE=pN ...')
    c.add_argument('picks_args', nargs='+')
    c.add_argument('--session')
    g = sub.add_parser('suggest', help='let the taste model + role rules suggest a riff per role')
    g.add_argument('--session')
    g.add_argument('--picks-bonus', type=float, default=0.5, help='extra score for riffs from your picks')
    g.add_argument('--apply', action='store_true', help='store the suggestion as your choice')
    b = sub.add_parser('build', help='make the song')
    b.add_argument('--session')
    b.add_argument('--model', default=str(HERE / 'data' / 'model.pt'))
    a = ap.parse_args()
    {'new': cmd_new, 'choose': cmd_choose, 'suggest': cmd_suggest, 'build': cmd_build}[a.cmd](a)


if __name__ == '__main__':
    main()
