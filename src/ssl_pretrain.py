"""train 이미지만 사용하는 자기지도 사전학습 (SimCLR, NT-Xent) — 정답(경도) 미사용, test 미사용.

같은 이미지의 서로 다른 두 '뷰'(경도와 무관한 변형만 적용)를 가깝게, 다른 이미지와는 멀게 학습한다.
- 뷰 변형: D4, 임의 회전, 노이즈, 블러, 밝기/대비, 확대 없는 무작위 크롭(시야만 변경)
- 확대·축소·전단은 결정립 크기·형상(경도 신호)을 바꾸므로 사용하지 않는다
- 검증 누수 방지: 폴드 k의 사전학습은 해당 폴드의 학습 이미지(400장)만 사용 → 같은 폴드의 미세조정에 사용
- 사전학습 직후 '고정 특징 + Ridge' 프로브로 표현 품질을 빠르게 비교 (같은 방식의 ImageNet 기준선과 비교: --epochs 0)

예) python src/ssl_pretrain.py --model resnet18.a1_in1k --epochs 200 --exp ssl_r18
산출물: outputs/ssl/<exp>/fold{k}.pt (timm 백본 state_dict), outputs/ssl/<exp>/probe.json
미세조정: python src/train_cnn.py --model resnet18 --init-weights "outputs/ssl/ssl_r18/fold{k}.pt" ...
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cnn import DATA, OUT, augment, d4, get_folds, load_split, random_crop  # noqa: E402


class SimCLR(nn.Module):
    def __init__(self, name, proj_dim=128):
        super().__init__()
        self.backbone = timm.create_model(name, pretrained=True, num_classes=0, in_chans=1)
        d = self.backbone.num_features
        self.proj = nn.Sequential(nn.Linear(d, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, proj_dim))

    def forward(self, x):
        return self.proj(self.backbone(x))


def nt_xent(z1, z2, t):
    z = F.normalize(torch.cat([z1, z2]), dim=1)
    n = z1.shape[0]
    sim = z @ z.T / t
    sim.fill_diagonal_(-1e4)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, target)


@torch.no_grad()
def embed(backbone, X, mean, std, bs=64):
    backbone.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        acc = 0
        for k in range(8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                acc = acc + backbone((d4(xb, k) - mean) / std).float()
        out.append((acc / 8).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet18.a1_in1k")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--crop", type=int, default=192)
    ap.add_argument("--noise", type=float, default=16.0)
    ap.add_argument("--rot-p", type=float, default=0.5)
    ap.add_argument("--blur-p", type=float, default=0.25)
    ap.add_argument("--contrast", type=float, default=0.1)
    ap.add_argument("--brightness", type=float, default=0.03)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--grad-ckpt", action="store_true")
    a = ap.parse_args()
    a.no_d4 = False
    torch.backends.cudnn.benchmark = True

    train = pd.read_csv(DATA / "train.csv")
    fold = get_folds(train)
    X = load_split("train", train.ID)
    y = train.hardness.values.astype(np.float64)
    out = OUT / "ssl" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    oof = np.zeros(len(y))
    t0 = time.time()
    done = []
    for k in (a.folds if a.folds is not None else range(5)):
        torch.manual_seed(a.seed + k)
        np.random.seed(a.seed + k)
        random.seed(a.seed + k)
        tr, va = fold != k, fold == k
        Xtr = X[tr].cuda()
        mean = (Xtr.float().mean() / 255).item()
        std = (Xtr.float().std() / 255).item()
        model = SimCLR(a.model).cuda()
        if a.grad_ckpt:
            model.backbone.set_grad_checkpointing(True)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
        n = len(Xtr)
        spe = max(1, n // a.batch)
        total = spe * a.epochs
        step = 0
        for ep in range(a.epochs):
            model.train()
            perm = torch.randperm(n, device="cuda")
            tl = 0.0
            for b in range(spe):
                idx = perm[b * a.batch:(b + 1) * a.batch]
                x = Xtr[idx].float().div_(255)
                v1 = (random_crop(augment(x.clone(), a), a.crop) - mean) / std
                v2 = (random_crop(augment(x.clone(), a), a.crop) - mean) / std
                lr = a.lr * min(1.0, (step + 1) / (spe * 5)) * 0.5 * (1 + math.cos(math.pi * step / total))
                for g in opt.param_groups:
                    g["lr"] = lr
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = nt_xent(model(v1).float(), model(v2).float(), a.temp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                tl += loss.item()
                step += 1
            if (ep + 1) % 50 == 0 or ep == a.epochs - 1:
                print(f"  fold{k} ssl ep{ep + 1} loss {tl / spe:.4f}", flush=True)
        torch.save(model.backbone.state_dict(), out / f"fold{k}.pt")
        # 프로브: 고정 백본 GAP 특징(D4 평균) + Ridge
        Etr = embed(model.backbone, X[tr], mean, std)
        Eva = embed(model.backbone, X[va], mean, std)
        rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 40))).fit(Etr, y[tr])
        oof[va] = rg.predict(Eva)
        r = float(np.sqrt(np.mean((oof[va] - y[va]) ** 2)))
        print(f"  fold{k} probe RMSE {r:.3f} ({time.time() - t0:.0f}s)", flush=True)
        done.append(k)
        del model, opt, Xtr
        torch.cuda.empty_cache()
    m = np.isin(fold, done)
    rmse = float(np.sqrt(np.mean((oof[m] - y[m]) ** 2)))
    with open(out / "probe.json", "w", encoding="utf-8") as f:
        json.dump({"exp": a.exp, "probe_rmse": rmse, "minutes": (time.time() - t0) / 60,
                   "args": {k: str(v) for k, v in vars(a).items()}}, f, ensure_ascii=False, indent=2)
    print(f"[{a.exp}] probe OOF RMSE {rmse:.3f} | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
