"""Song forms: templates from the reference songs, bent a little by statistics.

  python3 nn/form.py mine                    reads section names in ../tabs_ref/*.gp, writes nn/data/forms.json
  python3 nn/form.py list                    shows the templates
  python3 nn/form.py sample --template "Memento" --freedom 0.2 --seed 3

A template is the section list of a real song (role + bars), e.g. INTRO18 V8 PRE8 CH9 ... The
template is used strongly; with --freedom f roughly a share f of the sections is changed, but only
in ways the statistics of all songs make likely:
  * length changed to a typical length for that role (4 / 8 / 12 / 16)
  * the 2nd verse shortened, the 2nd pre-chorus dropped
  * a short break (2-4 bars) or interlude inserted where such a section commonly follows/precedes
  * development sections (BR / SOLO / BD) reordered, or an extra breakdown added before the end
Roles: INTRO V PRE CH POST BR SOLO BD INT RIFF BRK BUILD OUTRO
"""
import argparse
import collections
import json
import random
import re
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
FORMS = DATA / 'forms.json'
RULES = [('pre-chorus', 'PRE'), ('pre chorus', 'PRE'), ('prechorus', 'PRE'), ('post-chorus', 'POST'),
         ('chorus', 'CH'), ('refrain', 'CH'), ('hook', 'CH'), ('main theme', 'CH'), ('verse', 'V'),
         ('bridge', 'BR'), ('breakdown', 'BD'), ('all kick', 'BD'), ('solo', 'SOLO'), ('lead', 'SOLO'),
         ('intro', 'INTRO'), ('full drums', 'INTRO'), ('made by', 'INTRO'), ('outro', 'OUTRO'), ('ending', 'OUTRO'),
         ('final hit', 'OUTRO'), ('interlude', 'INT'), ('orchestral', 'INT'), ('clean', 'INT'), ('piano', 'INT'),
         ('bagpipe', 'INT'), ('riff', 'RIFF'), ('build', 'BUILD'), ('break', 'BRK')]
LENGTHS = (2, 4, 8, 12, 16, 24, 32)
# v9 (arranged by hand in this project) as a template too
V9 = [('INTRO', 4), ('V', 8), ('BRK', 1), ('CH', 8), ('POST', 8), ('V', 8), ('BRK', 1), ('CH', 8), ('POST', 8),
      ('BD', 8), ('CH', 16), ('INT', 8), ('OUTRO', 9)]


def role(text):
    s = text.lower()
    for k, v in RULES:
        if k in s:
            return v
    return 'RIFF'


def read_sections(gp):
    x = zipfile.ZipFile(gp).read('Content/score.gpif').decode('utf-8', 'replace')
    mbs = re.findall(r'<MasterBar>(.*?)</MasterBar>', x, re.S)
    secs = []
    for i, m in enumerate(mbs):
        s = re.search(r'<Section>.*?<Text>\s*<!\[CDATA\[(.*?)\]\]>', m, re.S)
        if s and s.group(1).strip():
            secs.append((i, s.group(1).strip()))
    out = []
    for k, (b, t) in enumerate(secs):
        e = secs[k + 1][0] if k + 1 < len(secs) else len(mbs)
        r = role(t)
        if out and out[-1][0] == r:
            out[-1] = (r, out[-1][1] + e - b)
        else:
            out.append((r, e - b))
    return out


def mine(ref_dir):
    temps = {}
    for gp in sorted(Path(ref_dir).glob('*.gp')):
        try:
            f = read_sections(gp)
        except Exception as e:
            print(f'skip {gp.name}: {e}'); continue
        if len(f) >= 5:
            temps[gp.stem] = f
    temps['v9 (this project)'] = V9
    trans, lens = collections.Counter(), collections.defaultdict(list)
    for f in temps.values():
        seq = ['START'] + [r for r, _ in f] + ['END']
        trans.update(zip(seq, seq[1:]))
        for r, l in f:
            lens[r].append(l)
    model = dict(templates={k: [list(x) for x in v] for k, v in temps.items()},
                 transitions={f'{a}>{b}': n for (a, b), n in trans.items()},
                 lengths={k: v for k, v in lens.items()})
    FORMS.write_text(json.dumps(model, indent=1))
    return model


def load():
    return json.loads(FORMS.read_text()) if FORMS.exists() else None


def p_trans(model, a, b):
    t = model['transitions']
    tot = sum(n for k, n in t.items() if k.startswith(a + '>')) or 1
    return (t.get(f'{a}>{b}', 0) + 0.1) / (tot + 1)


def typical_len(model, r, rng):
    ls = [l for l in model['lengths'].get(r, [8]) if l > 0]
    l = rng.choice(ls)
    return min(LENGTHS, key=lambda x: abs(x - l))


def sample(model, template=None, freedom=0.2, rng=None, max_bars=None, scale=1.0):
    """Return (plan, notes): plan = [(role, bars)], notes = list of changes made."""
    rng = rng or random.Random()
    temps = model['templates']
    if template:
        hits = [k for k in temps if template.lower() in k.lower()]
        if not hits:
            raise SystemExit(f'no template matching "{template}"')
        name = rng.choice(hits)
    else:
        name = rng.choice(list(temps))
    plan = [tuple(x) for x in temps[name]]
    notes = [f'template: {name}']
    seen = collections.Counter()
    out = []
    for i, (r, l) in enumerate(plan):
        seen[r] += 1
        nxt = plan[i + 1][0] if i + 1 < len(plan) else 'END'
        prev = out[-1][0] if out else 'START'
        if rng.random() < freedom:
            kind = rng.choice(['len', 'second', 'insert', 'skip'])
            if kind == 'len':
                nl = typical_len(model, r, rng)
                nl = max(l // 2, min(l * 2, nl, 16 if r not in ('SOLO', 'BD', 'BR', 'INT') else 32))
                if nl != l:
                    notes.append(f'{r}{l} -> {r}{nl}'); l = nl
            elif kind == 'second' and seen[r] >= 2 and r in ('V', 'PRE'):
                if r == 'PRE':
                    notes.append('2nd pre-chorus dropped'); continue
                nl = max(4, l // 2)
                notes.append(f'2nd verse {l} -> {nl}'); l = nl
            elif kind == 'insert' and out:
                cands = ['BRK', 'INT', 'BUILD']
                w = [p_trans(model, prev, c) * p_trans(model, c, r) for c in cands]
                c = rng.choices(cands, weights=w)[0]
                il = 2 if c == 'BRK' else 4
                out.append((c, il)); notes.append(f'inserted {c}{il} before {r}')
            elif kind == 'skip' and r in ('INT', 'RIFF', 'BRK') and len(plan) > 6:
                notes.append(f'dropped {r}{l}'); continue
        out.append((r, l))
    # development part: maybe reorder BR/SOLO/BD, maybe add a final breakdown (deathcore ending)
    dev = [i for i, (r, _) in enumerate(out) if r in ('BR', 'SOLO', 'BD')]
    if len(dev) >= 2 and rng.random() < freedom:
        a, b = rng.sample(dev, 2)
        out[a], out[b] = out[b], out[a]; notes.append('reordered development sections')
    if rng.random() < freedom * 0.5 and not any(r == 'BD' for r, _ in out[-3:]):
        pos = next((i for i in range(len(out) - 1, -1, -1) if out[i][0] == 'OUTRO'), len(out))
        out.insert(pos, ('BD', 8)); notes.append('extra breakdown before the end')
    snap = lambda l: l if l <= 2 else min(LENGTHS, key=lambda x: (abs(x - l), -x))   # 9 -> 8, 17 -> 16 ...
    out = [(r, snap(max(1, int(round(l * scale))))) for r, l in out]
    if max_bars:
        tot, cut = 0, []
        for r, l in out:
            if tot >= max_bars:
                break
            cut.append((r, l)); tot += l
        out = cut
    return out, notes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('cmd', choices=['mine', 'list', 'sample'])
    ap.add_argument('--refs', default=str(HERE.parent.parent / 'tabs_ref'))
    ap.add_argument('--template')
    ap.add_argument('--freedom', type=float, default=0.2)
    ap.add_argument('--seed', type=int)
    a = ap.parse_args()
    if a.cmd == 'mine':
        m = mine(a.refs)
        print(f"{len(m['templates'])} templates saved to {FORMS}")
        a.cmd = 'list'
    m = load()
    if m is None:
        sys.exit('run: python3 nn/form.py mine')
    if a.cmd == 'list':
        for k, v in m['templates'].items():
            print(f'{k[:40]:40s} ' + ' '.join(f'{r}{l}' for r, l in v))
    else:
        plan, notes = sample(m, a.template, a.freedom, random.Random(a.seed))
        print(' '.join(f'{r}{l}' for r, l in plan), f'  ({sum(l for _, l in plan)} bars)')
        print('\n'.join('  ' + n for n in notes))


if __name__ == '__main__':
    main()
