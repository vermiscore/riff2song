"""Turn the reference tabs into training tokens.

  python3 nn/prepare.py            # reads composition/tabs_ref (and v1/tabs_ref)

.gp (Guitar Pro 7/8) files are converted to .gp5 first (gp2gp5.py); .gp3/.gp4/.gp5
are used directly. Then DadaGP turns each song into a token file in nn/data/tokens/.
Add more songs to tabs_ref any time and run this again, then nn/train.py.
"""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1 = HERE.parent
sys.path.insert(0, str(HERE))
import gp2gp5                                                    # noqa: E402

DATA = HERE / 'data'
REF_DIRS = [V1.parent / 'tabs_ref', V1 / 'tabs_ref']


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('refs', nargs='*', help='tab files (default: tabs_ref folders)')
    a = ap.parse_args()
    files = [Path(p) for p in a.refs] or [p for d in REF_DIRS if d.is_dir()
                                          for p in sorted(d.iterdir())
                                          if p.suffix.lower() in ('.gp', '.gp3', '.gp4', '.gp5')]
    py = HERE / 'venv06' / 'bin' / 'python'
    if not py.exists():
        sys.exit('run nn/setup.sh first')
    (DATA / 'gp5').mkdir(parents=True, exist_ok=True)
    (DATA / 'tokens').mkdir(parents=True, exist_ok=True)
    ok = 0
    for f in files:
        stem = f.stem
        try:
            if f.suffix.lower() == '.gp':
                src = DATA / 'gp5' / f'{stem}.gp5'
                tracks = gp2gp5.convert(f, src)
            else:
                src, tracks = f, '(as is)'
        except Exception as e:                                  # noqa: BLE001
            print(f'skip {f.name}: {e}')
            continue
        out = DATA / 'tokens' / f'{stem}.txt'
        r = subprocess.run([str(py), 'dadagp.py', 'encode', str(src.resolve()), str(out.resolve()), 'unknown'],
                           cwd=HERE / 'dadaGP', capture_output=True, text=True)
        if r.returncode != 0 or not out.exists():
            err = (r.stderr.strip().splitlines() or ['?'])[-1]
            print(f'skip {f.name}: {err}')
            continue
        ok += 1
        print(f'{f.name}: {len(out.read_text().split())} tokens, tracks {tracks}')
    print(f'{ok}/{len(files)} songs ready in {DATA / "tokens"}  ->  next: python3 nn/train.py')


if __name__ == '__main__':
    main()
