"""Grow the NN from your ratings: continue training on the reference songs
plus the NN phrases/songs you rated highly (default: 4 or 5).

  python3 nn/finetune.py                 (~300 steps, a few minutes on CPU)
  python3 nn/finetune.py --min 5 --mix 0.4
  python3 nn/finetune.py --kto           (also learns from what you rated low: KTO)

--kto: Kahneman-Tversky Optimization (Ethayarajh et al. 2024). Short rated items (riffs, phrases) rated
>= 4 are "good", <= 2 "bad" (3 is ignored). The model is pushed to make good ones more likely and bad ones
less likely than a frozen copy of itself (the reference), with the reference songs still trained on so it
does not forget. It prints how well it separates held-out good/bad items before and after.

The reference songs always stay in the mix so the model doesn't drift into
copying only a few liked phrases. The previous model is kept as model_prev.pt.
"""
import argparse
import random
import shutil
import sys
import time
from pathlib import Path

import torch

from gpt import GPT
from train import load_songs, DATA

HERE = Path(__file__).resolve().parent
V1 = HERE.parent
sys.path.insert(0, str(V1))
from taste.common import read_ratings  # noqa: E402


def liked_sequences(min_rating, vocab_set, prefab_cap=0.5):
    """Token streams of NN and prefab outputs you rated highly. Prefab ones are capped at
    `prefab_cap` of the total so the NN learns from them without turning into the prefab composer."""
    seqs, pref = [], []
    for r in read_ratings():
        typ = r.get('type', '')
        if not typ.startswith(('nn', 'prefab', 'riff_sample')) or not r.get('rating'):
            continue
        try:
            if float(r['rating']) < min_rating:
                continue
        except ValueError:
            continue
        p = V1 / 'outputs' / r['run_id'] / f"{r['candidate']}.tokens.txt"
        if not p.exists():
            continue
        toks = p.read_text().split()
        if 'start' in toks:
            toks = toks[toks.index('start') + 1:]
        toks = [t for t in toks if t in vocab_set and t != 'end' and not t.startswith('drums:')]  # unseen tokens dropped
        if toks:
            (pref if typ.startswith(('prefab', 'riff_sample')) else seqs).append(toks)
    random.Random(0).shuffle(pref)
    limit = int(len(seqs) * prefab_cap / (1 - prefab_cap)) if prefab_cap < 1 else len(pref)
    if pref:
        print(f'liked prefab phrases: {len(pref)} (using {min(len(pref), limit)})')
    return seqs + pref[:limit]


KTO_TYPES = ('riff_sample', 'nn_riff_sample', 'nn_phrase', 'prefab_phrase')


def rated_items(vocab_set, good=4.0, bad=2.0, types=KTO_TYPES):
    """(tokens, is_good) for short rated items that still have their .tokens.txt."""
    out = []
    for r in read_ratings():
        if r.get('type') not in types or not r.get('rating', '').strip():
            continue
        v = float(r['rating'])
        if bad < v < good:
            continue
        p = V1 / 'outputs' / r['run_id'] / f"{r['candidate']}.tokens.txt"
        if not p.exists():
            continue
        toks = p.read_text().split()
        if 'start' in toks:
            toks = toks[toks.index('start') + 1:]
        toks = [t for t in toks if t in vocab_set and t != 'end' and not t.startswith('drums:')]
        if len(toks) >= 8:
            out.append((toks, v >= good))
    return out


def seq_logp(model, batch, stoi, ctx):
    """Sum of log p(item tokens | 'start') for each item of the batch."""
    import torch.nn.functional as F
    st = stoi['start']
    seqs = [[st] + [stoi[t] for t in toks][:ctx - 1] for toks in batch]
    L = max(map(len, seqs))
    x = torch.zeros(len(seqs), L - 1, dtype=torch.long)
    y = torch.zeros(len(seqs), L - 1, dtype=torch.long)
    m = torch.zeros(len(seqs), L - 1)
    for i, s in enumerate(seqs):
        n = len(s) - 1
        x[i, :n] = torch.tensor(s[:-1]); y[i, :n] = torch.tensor(s[1:]); m[i, :n] = 1
    logits, _ = model(x)
    lp = F.log_softmax(logits, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    return (lp * m).sum(1)


def kto(a, model, vocab, stoi, ctx, ref_t):
    import copy
    import torch.nn.functional as F
    items = rated_items(set(vocab), a.min, a.bad)
    if len(items) < 20:
        raise SystemExit('need at least 20 rated riffs/phrases for --kto')
    random.seed(0)
    random.shuffle(items)
    n_hold = max(10, len(items) // 10)
    hold, train = items[:n_hold], items[n_hold:]
    nD = sum(1 for _, g in train if g); nU = len(train) - nD
    # paper: lambda_D * nD / (lambda_U * nU) in [1, 4/3] -> up-weight the rarer side
    lam_D, lam_U = (max(1.0, nU / max(1, nD)), 1.0) if nD < nU else (1.0, max(1.0, nD / max(1, nU)))
    print(f'KTO: {nD} good, {nU} bad (rated >= {a.min:g} / <= {a.bad:g}), {len(hold)} held out; '
          f'weights good {lam_D:.2f} bad {lam_U:.2f}, beta {a.beta}')
    ref = copy.deepcopy(model).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    @torch.no_grad()
    def separation():
        model.eval()
        r = []
        for i in range(0, len(hold), 16):
            b = hold[i:i + 16]
            toks = [t for t, _ in b]
            d = seq_logp(model, toks, stoi, ctx) - seq_logp(ref, toks, stoi, ctx)
            lp = seq_logp(model, toks, stoi, ctx) / torch.tensor([len(t) for t in toks], dtype=torch.float)
            r += [(float(lp[k]), g) for k, (_, g) in enumerate(b)]
        model.train()
        good = [v for v, g in r if g]; badv = [v for v, g in r if not g]
        # share of (good, bad) pairs where the good one is more likely per token
        pairs = [(x > y) for x in good for y in badv]
        return sum(pairs) / max(1, len(pairs))

    @torch.no_grad()
    def song_loss():
        model.eval()
        rnd = random.Random(1)
        L = []
        for _ in range(40):
            s = rnd.choice(ref_t)
            n = min(ctx, len(s) - 1)
            i = rnd.randint(0, len(s) - n - 1)
            L.append(model(s[i:i + n][None], s[i + 1:i + n + 1][None])[1].item())
        model.train()
        return sum(L) / len(L)

    sl0 = song_loss()
    print(f'held-out: good riffs more likely than bad ones in {100 * separation():.0f}% of pairs (before)')
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
    t0 = time.time()
    model.train()
    for step in range(1, a.steps + 1):
        b = random.sample(train, min(a.batch, len(train)))
        toks = [t for t, _ in b]
        good = torch.tensor([g for _, g in b])
        with torch.no_grad():
            ref_lp = seq_logp(ref, toks, stoi, ctx)
        r = seq_logp(model, toks, stoi, ctx) - ref_lp
        z = r.detach().mean().clamp(min=0)                     # reference point (KL estimate)
        loss_k = torch.where(good, lam_D * (1 - torch.sigmoid(a.beta * (r - z))),
                             lam_U * (1 - torch.sigmoid(a.beta * (z - r)))).mean()
        # keep training on the reference songs so the model does not forget them
        xs, ys = zip(*(window(random.choice(ref_t)) for _ in range(a.batch // 2)))
        L = min(len(x) for x in xs)
        _, loss_r = model(torch.stack([x[:L] for x in xs]), torch.stack([y[:L] for y in ys]))
        loss = loss_k + a.ref_weight * loss_r
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 25 == 0 or step == a.steps:
            rg = r[good].mean().item() if good.any() else float('nan')
            rb = r[~good].mean().item() if (~good).any() else float('nan')
            print(f'step {step:4d}  kto {loss_k.item():.3f}  songs {loss_r.item():.3f}  '
                  f'good {rg:+.1f}  bad {rb:+.1f}  ({time.time() - t0:.0f}s)', flush=True)
    print(f'held-out: good riffs more likely than bad ones in {100 * separation():.0f}% of pairs (after)')
    print(f'reference-song loss {sl0:.3f} -> {song_loss():.3f}  (a big rise = the model is forgetting the songs)')


def window(s, ctx=None):
    ctx = ctx or WINDOW[0]
    L = min(ctx, len(s) - 1)
    i = random.randint(0, len(s) - L - 1)
    return s[i:i + L], s[i + 1:i + L + 1]


WINDOW = [512]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--min', type=float, default=4, help='use NN outputs rated at least this')
    ap.add_argument('--steps', type=int, default=300)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--mix', type=float, default=0.3, help='share of training batches from liked material')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--model', default=str(DATA / 'model.pt'))
    ap.add_argument('--tokens', default=str(DATA / 'tokens'))
    ap.add_argument('--kto', action='store_true', help='learn from good AND bad ratings (riffs/phrases)')
    ap.add_argument('--bad', type=float, default=2, help='--kto: rated at most this = bad')
    ap.add_argument('--beta', type=float, default=0.1, help='--kto: how far the model may move from itself')
    ap.add_argument('--ref-weight', type=float, default=1.0, help='--kto: weight of the reference-song loss')
    a = ap.parse_args()
    if a.kto and a.lr == 2e-4:
        a.lr = 2e-5                                             # gentler default for KTO
    if a.kto and a.steps == 300:
        a.steps = 150

    ck = torch.load(a.model, map_location='cpu')
    vocab = ck['vocab']
    stoi = {t: i for i, t in enumerate(vocab)}
    cfg = ck['config']
    model = GPT(len(vocab), **{k: v for k, v in cfg.items() if k != 'vocab'})
    model.load_state_dict(ck['state'])
    ctx = cfg['ctx']

    WINDOW[0] = ctx
    refs = [[t for t in s if t in stoi] for s in load_songs(a.tokens)]
    if a.kto:
        ref_t = [torch.tensor([stoi[t] for t in s]) for s in refs]
        kto(a, model, vocab, stoi, ctx, ref_t)
        shutil.copy(a.model, Path(a.model).with_name('model_prev.pt'))
        torch.save(dict(state=model.state_dict(), config=model.config, vocab=vocab), a.model)
        print(f'updated {a.model} (previous kept as model_prev.pt)')
        return
    liked = liked_sequences(a.min, set(vocab))
    if not liked:
        raise SystemExit(f'no NN outputs rated >= {a.min} yet: rate some (taste/rate.py) first')
    # liked phrases are short: chain them into one long stream so windows fill the context
    random.seed(0)
    stream = []
    for _ in range(max(1, (ctx * 4) // max(1, sum(map(len, liked))) + 1)):
        random.shuffle(liked)
        for s in liked:
            stream += s
    ref_t = [torch.tensor([stoi[t] for t in s]) for s in refs]
    liked_t = torch.tensor([stoi[t] for t in stream])
    print(f'{len(refs)} reference songs + {len(liked)} liked NN outputs (rating >= {a.min:g})')

    def window(s):
        L = min(ctx, len(s) - 1)
        i = random.randint(0, len(s) - L - 1)
        return s[i:i + L], s[i + 1:i + L + 1]

    def batch():
        xs, ys = [], []
        for _ in range(a.batch):
            x, y = window(liked_t if random.random() < a.mix else random.choice(ref_t))
            xs.append(x); ys.append(y)
        L = min(len(x) for x in xs)
        return torch.stack([x[:L] for x in xs]), torch.stack([y[:L] for y in ys])

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1)
    t0 = time.time()
    model.train()
    for step in range(1, a.steps + 1):
        x, y = batch()
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == a.steps:
            print(f'step {step:4d}  loss {loss.item():.3f}  ({time.time() - t0:.0f}s)', flush=True)

    shutil.copy(a.model, Path(a.model).with_name('model_prev.pt'))
    torch.save(dict(state=model.state_dict(), config=model.config, vocab=vocab), a.model)
    print(f'updated {a.model} (previous kept as model_prev.pt)')


if __name__ == '__main__':
    main()
