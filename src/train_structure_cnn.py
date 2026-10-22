"""Train-only controlled CNN ablations. No test access unless --predict-test.

Variants: raw, views (raw/NLM/boundary), stats (multistage mean/std),
fine_stats (same statistics, stride-1 first conv). Saved checkpoints include
all normalization values and configuration needed for inference.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import pandas as pd
import timm
import torch
from torch import nn
import torch.nn.functional as F
from threadpoolctl import threadpool_limits

import train_cnn as T
from evaluate_structure import folds_for, rmse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs' / 'structure_v1' / 'cnn'


class StatsNet(nn.Module):
    def __init__(self, fine=False, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model('resnet18.a1_in1k', pretrained=pretrained,
                                          in_chans=1, features_only=True, out_indices=(1, 2, 3, 4))
        if fine:
            self.backbone.conv1.stride = (1, 1)
        n = 2 * sum(self.backbone.feature_info.channels())
        self.head = nn.Sequential(nn.LayerNorm(n), nn.Linear(n, 128), nn.SiLU(),
                                  nn.Dropout(.2), nn.Linear(128, 1))

    def forward(self, x):
        stats = []
        for f in self.backbone(x):
            f = f.float()
            stats.extend([f.mean((-2, -1)), torch.sqrt(f.var((-2, -1), unbiased=False) + 1e-6)])
        return self.head(torch.cat(stats, 1))


def model_for(variant, pretrained=True):
    if variant in ['stats', 'fine_stats']:
        return StatsNet(fine=variant == 'fine_stats', pretrained=pretrained)
    return timm.create_model('resnet18.a1_in1k', pretrained=pretrained,
        in_chans=3 if variant == 'views' else 1, num_classes=1, drop_rate=.1, drop_path_rate=.1)


def load_images(split, ids, variant):
    raw = T.load_split(split, ids)
    if variant != 'views':
        return raw
    cache = np.load(ROOT / 'outputs' / 'structure_v1' / f'{split}_views.npz')
    assert np.array_equal(cache['ID'], np.asarray(ids))
    return torch.cat([raw, torch.from_numpy(cache['views'])], dim=1)


def augment(x, a):
    if x.shape[1] == 1:
        return T.augment(x, a)
    # Geometric operations act jointly on all channels. Brightness/contrast
    # change raw/NLM intensities together; extra noise is applied to raw only.
    B = len(x)
    transform = torch.randint(0, 8, (B,), device=x.device)
    out = torch.empty_like(x)
    for k in range(8):
        m = transform == k
        if m.any():
            out[m] = T.d4(x[m], k)
    x = T.rotate_arbitrary(out, a.rot_p)
    if a.blur_p:
        mask = torch.rand(B, device=x.device) < a.blur_p
        if mask.any():
            from torchvision.transforms.functional import gaussian_blur
            x[mask] = gaussian_blur(x[mask], 5, random.uniform(.3, 1.))
    c = 1 + (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * a.contrast
    b = (torch.rand(B, 1, 1, 1, device=x.device) * 2 - 1) * a.brightness
    mu = x[:, :2].mean((-2, -1), keepdim=True)
    x[:, :2] = (x[:, :2] - mu) * c + mu + b
    sig = torch.rand(B, 1, 1, 1, device=x.device) * a.noise / 255
    x[:, :1] += torch.randn_like(x[:, :1]) * sig
    return x.clamp(0, 1)


@torch.no_grad()
def predict(model, X, mean, std, dtype, tta=True, batch=32):
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        x = X[i:i + batch].to('cuda').float() / 255
        p = []
        for k in range(8) if tta else [0]:
            with torch.autocast('cuda', dtype=dtype):
                v = model((T.d4(x, k) - mean) / std)
            p.append(v.float().squeeze(1))
        preds.append(torch.stack(p).mean(0).cpu().numpy())
    return np.concatenate(preds)


def train_fold(k, X, y, fold, a, dst):
    tr, va = fold != k, fold == k
    Xtr, Xva = X[tr].to('cuda'), X[va]
    ymu, ysd = float(y[tr].mean()), float(y[tr].std())
    target = torch.tensor((y[tr] - ymu) / ysd, dtype=torch.float32, device='cuda')
    # Dataset statistics are derived solely from the outer training fold.
    xf = Xtr.float() / 255
    mean = xf.mean((0, 2, 3), keepdim=True)
    std = xf.std((0, 2, 3), keepdim=True).clamp_min(1e-4)
    del xf
    steps = len(Xtr) // a.batch
    histories = []
    for attempt in range(3):
        seed = a.seed * 100 + k + attempt * 7919
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        model = model_for(a.variant).to('cuda')
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.05)
        history = []
        step = 0
        for ep in range(a.epochs):
            model.train()
            perm = torch.randperm(len(Xtr), device='cuda')
            total_loss = 0.
            for b in range(steps):
                ix = perm[b * a.batch:(b + 1) * a.batch]
                x = (augment(Xtr[ix].float() / 255, a) - mean) / std
                opt.param_groups[0]['lr'] = T.cosine_lr(step, steps * a.epochs, steps, a.lr)
                opt.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=a.dtype):
                    pred = model(x).float().squeeze(1)
                    loss = F.mse_loss(pred, target[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.)
                opt.step()
                total_loss += float(loss.detach())
                step += 1
            if (ep + 1) % 10 == 0 or ep + 1 == a.epochs:
                p = predict(model, Xva, mean, std, a.dtype, tta=False, batch=a.batch) * ysd + ymu
                record = {'epoch': ep + 1, 'train_loss': total_loss / steps, 'val_rmse': rmse(y[va], p)}
                history.append(record)
                print(f'{a.variant} fold{k} {record}', flush=True)
        histories.append(history)
        if history[-1]['train_loss'] <= .8:
            break
        if attempt < 2:
            del model, opt
            torch.cuda.empty_cache()
    p = predict(model, Xva, mean, std, a.dtype, batch=a.batch) * ysd + ymu
    torch.save({'state_dict': model.cpu().state_dict(), 'mean': mean.cpu(), 'std': std.cpu(),
                'ymu': ymu, 'ysd': ysd, 'variant': a.variant, 'split_seed': a.split_seed,
                'seed': a.seed, 'epochs': a.epochs, 'lr': a.lr, 'batch': a.batch,
                'augmentation': {'noise': a.noise, 'rot_p': a.rot_p, 'blur_p': a.blur_p,
                                 'contrast': a.contrast, 'brightness': a.brightness}}, dst / f'fold{k}.pt')
    del model, opt, Xtr, target, mean, std
    torch.cuda.empty_cache()
    return p, {'rmse': rmse(y[va], p), 'attempts': histories}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', choices=['raw', 'views', 'stats', 'fine_stats'], required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split-seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--folds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    ap.add_argument('--predict-test', action='store_true')
    ap.add_argument('--resume', action='store_true')
    a = ap.parse_args()
    a.dtype = torch.bfloat16
    a.no_d4, a.rot_p, a.blur_p, a.contrast, a.brightness, a.noise = False, .5, .25, .1, .03, 16.
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True
    dst = OUT / f'{a.variant}_split{a.split_seed}_seed{a.seed}'
    dst.mkdir(parents=True, exist_ok=True)
    if a.predict_test:
        ss = pd.read_csv(ROOT / 'data' / 'sample_submission.csv')
        X = load_images('test', ss.ID, a.variant)
        preds = []
        for k in range(5):
            cp = torch.load(dst / f'fold{k}.pt', weights_only=False, map_location='cpu')
            model = model_for(cp['variant'], pretrained=False).to('cuda')
            model.load_state_dict(cp['state_dict'])
            preds.append(predict(model, X, cp['mean'].to('cuda'), cp['std'].to('cuda'), a.dtype,
                                 batch=a.batch) * cp['ysd'] + cp['ymu'])
            del model
            torch.cuda.empty_cache()
        ss.hardness = np.mean(preds, axis=0)
        assert np.isfinite(ss.hardness).all()
        ss.to_csv(dst / 'test_pred.csv', index=False)
        print('Inference saved', flush=True)
        return
    tr = pd.read_csv(ROOT / 'data' / 'train.csv')
    X = load_images('train', tr.ID, a.variant)
    y, fold = tr.hardness.to_numpy(), folds_for(tr, a.split_seed)
    pred = np.full(len(tr), np.nan)
    logs = {}
    if a.resume and (dst / 'oof.csv').exists():
        old = pd.read_csv(dst / 'oof.csv').set_index('ID').loc[tr.ID]
        assert np.array_equal(old.fold, fold)
        pred = old.pred.to_numpy()
        logs = json.loads((dst / 'log.json').read_text())['folds']
        expected = {'variant': a.variant, 'split_seed': a.split_seed, 'seed': a.seed,
                    'epochs': a.epochs, 'lr': a.lr, 'batch': a.batch}
        for completed in logs:
            cp = torch.load(dst / f'fold{completed}.pt', weights_only=False, map_location='cpu')
            if any(cp[key] != value for key, value in expected.items()):
                raise ValueError('Resume configuration differs from saved checkpoints; use a new experiment path.')
            mask = fold == int(completed)
            if not np.isfinite(pred[mask]).all():
                raise ValueError('Completed fold contains missing predictions.')
            del cp
    elif (dst / 'oof.csv').exists():
        raise FileExistsError(f'{dst}: use --resume or a new seed')
    t0 = time.time()
    with threadpool_limits(limits=2):
        for k in a.folds:
            if str(k) in logs:
                continue
            p, log = train_fold(k, X, y, fold, a, dst)
            pred[fold == k] = p
            logs[str(k)] = log
            pd.DataFrame({'ID': tr.ID, 'fold': fold, 'hardness': y, 'pred': pred}).to_csv(dst / 'oof.csv', index=False)
            valid = np.isfinite(pred)
            summary = {'variant': a.variant, 'split_seed': a.split_seed, 'seed': a.seed,
                       'epochs': a.epochs, 'lr': a.lr, 'batch': a.batch,
                       'n_predicted': int(valid.sum()), 'oof_rmse': rmse(y[valid], pred[valid]),
                       'folds': logs, 'minutes_this_run': (time.time()-t0)/60,
                       'train_cnn_sha256': hashlib.sha256(Path(T.__file__).read_bytes()).hexdigest()}
            (dst / 'log.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
            print(f'Completed {a.variant} fold{k}: {log["rmse"]:.4f}; OOF({valid.sum()}) {summary["oof_rmse"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
