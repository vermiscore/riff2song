"""Shared paths and ratings.csv helpers."""
import csv
from pathlib import Path

V1 = Path(__file__).resolve().parent.parent
OUTPUTS = V1 / 'outputs'
RATINGS = OUTPUTS / 'ratings.csv'
MODEL = OUTPUTS / 'taste_model.json'
FIELDS = ['run_id', 'candidate', 'type', 'seed', 'rating', 'note']


def read_ratings(path=RATINGS):
    if not Path(path).exists():
        return []
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def write_ratings(rows, path=RATINGS):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in FIELDS})
