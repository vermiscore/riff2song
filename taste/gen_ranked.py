"""Generate many candidates, keep the ones the taste model thinks you will like.

  python3 taste/gen_ranked.py --type section --n 40 --keep 5

Any other options are passed to gen/cli.py (e.g. --pedal A --tempo 150 --bars 16).
Kept candidates go to v1/outputs/<time>_<type>_ranked/ and into ratings.csv, so
rating them (taste/rate.py) keeps improving the model.
One extra candidate is picked at random (--explore) so the model also hears
things it would not have chosen.
"""
import argparse
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

V1 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V1))
from taste.common import OUTPUTS, RATINGS, MODEL, read_ratings, write_ratings   # noqa: E402
from taste import features as F                                                  # noqa: E402
from taste import model as M                                                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--n', type=int, default=40, help='candidates to generate')
    ap.add_argument('--keep', type=int, default=5, help='best candidates to keep')
    ap.add_argument('--explore', type=int, default=1, help='extra random candidates to keep')
    ap.add_argument('--model', default=str(MODEL))
    a, passthrough = ap.parse_known_args()

    model = M.load(a.model) if Path(a.model).exists() else None
    if model is None:
        print('no taste model yet (rate candidates, then run taste/train.py): keeping candidates unranked')

    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, str(V1 / 'gen' / 'cli.py'), '--n', str(a.n), '--out', tmp] + passthrough
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            sys.exit(res.stdout + res.stderr)
        runs = [p for p in Path(tmp).iterdir() if p.is_dir()]
        if not runs:
            sys.exit('generator produced nothing:\n' + res.stdout + res.stderr)
        run = runs[0]
        gen_rows = {r['candidate']: r for r in read_ratings(Path(tmp) / 'ratings.csv')}

        scored = []
        for midi in sorted(run.glob('cand_*.mid')):
            s = M.predict(model, F.extract(midi)) if model else 0.0
            scored.append((s, midi.stem))
        scored.sort(reverse=True)
        keep = scored[:a.keep]
        rest = scored[a.keep:]
        explore = random.sample(rest, min(a.explore, len(rest))) if model else []

        dest = OUTPUTS / f'{run.name}_ranked'
        dest.mkdir(parents=True, exist_ok=True)
        if (run / 'run.json').exists():
            shutil.copy(run / 'run.json', dest / 'run.json')
        lines, new_rows = [], []
        for tag, group in (('pick', keep), ('explore', explore)):
            for s, cand in group:
                for f in run.glob(f'{cand}.*'):
                    shutil.copy(f, dest / f.name)
                lines.append(f'{cand}  predicted {s:.2f}  ({tag})')
                g = gen_rows.get(cand, {})
                new_rows.append(dict(run_id=dest.name, candidate=cand, type=g.get('type', ''),
                                     seed=g.get('seed', ''), rating='', note=''))
        (dest / 'scores.txt').write_text(
            '\n'.join(lines) + '\n\nall candidates:\n' +
            '\n'.join(f'{c}  {s:.2f}' for s, c in scored) + '\n')

    rows = read_ratings(RATINGS)
    write_ratings(rows + new_rows, RATINGS)
    print(res.stdout.strip().splitlines()[-3] if res.stdout else '')
    print('\n'.join(lines))
    print(f'saved {len(new_rows)} candidates to {dest}\nrate them: python3 {V1 / "taste" / "rate.py"} --run {dest.name}')


if __name__ == '__main__':
    main()
