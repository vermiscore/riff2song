"""Train the small GPT on the tokenized reference songs (nn/data/tokens/*.txt).

  python3 nn/train.py            (defaults: 2000 steps, ~10-25 min on a laptop CPU)

CPU is fine for ~10 songs (minutes to ~half an hour). Saves nn/data/model.pt.
"""
import argparse
import json
import random
import time
from pathlib import Path

import torch

from gpt import GPT, n_params

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'


def load_songs(token_dir, drums=False):
    songs = []
    for p in sorted(Path(token_dir).glob('*.txt')):
        toks = p.read_text().split()
        if not drums:                       # drums are added by rules after generation (drumify.py)
            toks = [x for x in toks if not x.startswith('drums:')]
        if 'start' in toks and toks[-1] == 'end':
            songs.append(toks)
    return songs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--ctx', type=int, default=256)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--d', type=int, default=128)
    ap.add_argument('--layers', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--tokens', default=str(DATA / 'tokens'))
    ap.add_argument('--drums', action='store_true', help='also learn the drums (default: guitars + bass only)')
    ap.add_argument('--out', default=str(DATA / 'model.pt'))
    a = ap.parse_args()

    songs = load_songs(a.tokens, a.drums)
    if not songs:
        raise SystemExit(f'no token files in {a.tokens}: run nn/prepare.py first')
    vocab = sorted({t for s in songs for t in s} | {'<pad>'})
    stoi = {t: i for i, t in enumerate(vocab)}
    seqs = [torch.tensor([stoi[t] for t in s]) for s in songs]
    total = sum(len(s) for s in seqs)
    # hold out the last 5% of each song for validation
    train = [s[:int(len(s) * 0.95)] for s in seqs]
    val = [s[int(len(s) * 0.95):] for s in seqs]
    print(f'{len(songs)} songs, {total} tokens, vocab {len(vocab)}')

    torch.manual_seed(0)
    model = GPT(len(vocab), ctx=a.ctx, d=a.d, layers=a.layers)
    print(f'model: {n_params(model) / 1e6:.2f}M parameters')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)

    def batch(pool):
        pool = [s for s in pool if len(s) > a.ctx + 1] or pool
        xs, ys = [], []
        for _ in range(a.batch):
            s = random.choice(pool)
            L = min(a.ctx, len(s) - 1)
            i = random.randint(0, len(s) - L - 1)
            xs.append(s[i:i + L]); ys.append(s[i + 1:i + L + 1])
        L = min(len(x) for x in xs)
        return torch.stack([x[:L] for x in xs]), torch.stack([y[:L] for y in ys])

    t0 = time.time()
    best = float('inf')
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1, step / 100) * (0.1 + 0.9 * (1 - step / a.steps))
        for g in opt.param_groups:
            g['lr'] = lr
        model.train()
        x, y = batch(train)
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0 or step == a.steps:
            model.eval()
            with torch.no_grad():
                vl = sum(model(*batch(val))[1].item() for _ in range(4)) / 4
            print(flush=True, *[f'step {step:5d}  train {loss.item():.3f}  val {vl:.3f}  ({time.time() - t0:.0f}s)'])
            if vl < best:
                best = vl
                torch.save(dict(state=model.state_dict(), config=model.config, vocab=vocab), a.out)
    print(f'saved best model (val {best:.3f}) to {a.out}')


if __name__ == '__main__':
    main()
