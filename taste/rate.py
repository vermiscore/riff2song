"""Listen to unrated candidates and rate them 1-5.

  python3 taste/rate.py

Each candidate starts playing; type a rating and press Enter:
  1-5        rating (you can add a note: "4 nice groove")
  r          replay        s   skip        q   quit (progress is saved)
Ratings go into v1/outputs/ratings.csv, which the taste model learns from.
Audio: fluidsynth renders the MIDI with a General MIDI SoundFont, then plays it.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from taste.common import OUTPUTS, RATINGS, read_ratings, write_ratings   # noqa: E402

SOUNDFONTS = ['/usr/share/sounds/sf2/FluidR3_GM.sf2', '/usr/share/sounds/sf2/default-GM.sf2',
              '/usr/share/sounds/sf2/TimGM6mb.sf2']


def find_soundfont(arg):
    for p in ([arg] if arg else []) + SOUNDFONTS:
        if p and Path(p).exists():
            return p
    sys.exit('no SoundFont found: sudo apt install fluid-soundfont-gm  (or pass --sf2)')


def player_cmd(wav):
    for cmd in (['ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet'], ['paplay'], ['aplay', '-q']):
        if shutil.which(cmd[0]):
            return cmd + [str(wav)]
    sys.exit('no audio player found (ffplay, paplay or aplay)')


def render(midi, sf2, tmpdir):
    if not shutil.which('fluidsynth'):
        sys.exit('fluidsynth not found: sudo apt install fluidsynth')
    wav = Path(tmpdir) / (midi.stem + '.wav')
    subprocess.run(['fluidsynth', '-ni', '-g', '0.8', '-F', str(wav), '-r', '44100', sf2, str(midi)],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return wav


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ratings', default=str(RATINGS), help='ratings.csv to use')
    ap.add_argument('--outputs', default=str(OUTPUTS), help='folder with the run folders')
    ap.add_argument('--sf2', help='SoundFont file')
    ap.add_argument('--run', help='only rate candidates of this run id')
    a = ap.parse_args()

    rows = read_ratings(a.ratings)
    todo = [r for r in rows if not r['rating'].strip()
            and (not a.run or r['run_id'] == a.run)
            and (Path(a.outputs) / r['run_id'] / f"{r['candidate']}.mid").exists()]
    rated = sum(1 for r in rows if r['rating'].strip())
    print(f'{rated} rated so far, {len(todo)} waiting with audio')
    if not todo:
        return
    sf2 = find_soundfont(a.sf2)
    with tempfile.TemporaryDirectory() as tmp:
        for i, r in enumerate(todo, 1):
            midi = Path(a.outputs) / r['run_id'] / f"{r['candidate']}.mid"
            txt = midi.with_suffix('.txt')
            plan = ''
            if txt.exists():
                plan = next((l.strip() for l in txt.read_text().splitlines() if l.startswith('plan:')), '')
            print(f"\n[{i}/{len(todo)}] {r['run_id']} {r['candidate']}  {plan}")
            wav = render(midi, sf2, tmp)
            proc = subprocess.Popen(player_cmd(wav))
            while True:
                ans = input('rating 1-5 (r replay, s skip, q quit): ').strip()
                if ans == 'r':
                    proc.terminate(); proc = subprocess.Popen(player_cmd(wav)); continue
                if ans in ('s', 'q') or (ans[:1].isdigit() and 1 <= int(ans[:1]) <= 5):
                    break
                print('  type 1-5, r, s or q')
            proc.terminate()
            if ans == 'q':
                break
            if ans == 's':
                continue
            r['rating'] = ans[:1]
            extra = ans[1:].strip()
            r['note'] = (r.get('note', '') + ' ' + extra).strip() if extra else r.get('note', '')
            write_ratings(rows, a.ratings)
    print(f"\nsaved to {a.ratings}  ({sum(1 for r in rows if r['rating'].strip())} rated)")


if __name__ == '__main__':
    main()
