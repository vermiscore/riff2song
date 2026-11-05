"""Learn your taste from ratings.csv and show what you seem to like.

  python3 taste/train.py

Saves v1/outputs/taste_model.json, used by gen_ranked.py.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taste.common import OUTPUTS, RATINGS, MODEL, read_ratings   # noqa: E402
from taste import features as F                                   # noqa: E402
from taste import model as M                                      # noqa: E402


def dataset(rows, outputs):
    X, y, skipped = [], [], 0
    for r in rows:
        if not r['rating'].strip():
            continue
        midi = F.candidate_midi(outputs, r['run_id'], r['candidate'])
        if midi is None:
            skipped += 1
            continue
        X.append(F.extract(midi))
        y.append(float(r['rating']))
    return X, y, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ratings', default=str(RATINGS))
    ap.add_argument('--outputs', default=str(OUTPUTS))
    ap.add_argument('--model', default=str(MODEL))
    a = ap.parse_args()

    X, y, skipped = dataset(read_ratings(a.ratings), a.outputs)
    print(f'{len(y)} rated candidates with audio ({skipped} rated but their files are gone)')
    if len(y) < M.MIN_RATINGS:
        sys.exit(f'rate at least {M.MIN_RATINGS} (now {len(y)}): python3 taste/rate.py')
    model = M.train(X, y)
    M.save(model, a.model)
    gain = 1 - model['cv_rmse'] / model['baseline_rmse'] if model['baseline_rmse'] else 0
    print(f"model: {model['kind']}, cross-validated error {model['cv_rmse']:.2f} "
          f"vs {model['baseline_rmse']:.2f} for always guessing your average "
          f"({gain:+.0%} better)")
    if gain <= 0:
        print('  -> not better than guessing yet: rate more candidates (and use the whole 1-5 range)')
    print(M.explain(model))
    print(f'saved {a.model}')


if __name__ == '__main__':
    main()
