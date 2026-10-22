"""DINOv2 도메인 적응: train 이미지만으로 짧은 SimCLR 자기지도 학습 → 다층 특징 + Ridge (OOF·test).

- 폴드 k: 폴드 k의 학습 이미지(400장)로만 적응 (정답 미사용) → 같은 폴드 검증/ test 예측에 사용 (누수 없음)
- 원래 DINOv2 표현을 보존하기 위해 낮은 학습률·짧은 학습, 입력은 3채널 복제 + DINOv2 정규화 (probe_models.py와 동일)
- 스케일 일치: 프로브는 256→518(약 2배) 업샘플 → 적응 학습 뷰도 크롭 후 2배 업샘플(확대·축소 증강 아님)
- 평가: 적응된 백본으로 518 입력의 지정 층 평균·표준편차 특징(D4 4방향) → 폴드별 Ridge

예) python src/dino_adapt.py --model vit_base_patch14_dinov2.lvd142m --layers 0 1 2 3 4 5 6 7 8 9 10 11 --exp dinoB_adapt
산출물: outputs/emb_reg/<exp>/{oof.csv, test_pred.csv}
"""
from __future__ import annotations

import argparse
import math
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
from ssl_pretrain import nt_xent  # noqa: E402

PUB_STD = 16.56


@torch.no_grad()
def embed(m, mean, std, X, layers, size=518, bs=16):
    m.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        acc = 0
        for k in (0, 1, 4, 5):
            xi = F.interpolate(d4(xb, k), size=(size, size), mode="bilinear", align_corners=False)
            xi = (xi.repeat(1, 3, 1, 1) - mean) / std
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fs = m.forward_intermediates(xi, indices=layers, intermediates_only=True)
            acc = acc + torch.cat([torch.cat([z.float().mean((2, 3)), z.float().std((2, 3))], 1) for z in fs], 1)
        out.append((acc / 4).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_base_patch14_dinov2.lvd142m")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=2e-5, help="백본 학습률 (투영 헤드는 50배)")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--crop", type=int, default=182, help="원본 크롭 크기 → 2배 업샘플 (182→364=26×14)")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    a = ap.parse_args()
    a.no_d4, a.rot_p, a.blur_p, a.contrast, a.brightness, a.noise = False, 0.5, 0.25, 0.1, 0.03, 16.0
    torch.backends.cudnn.benchmark = True
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = get_folds(train)
    y = train.hardness.values.astype(np.float64)
    X, Xte = load_split("train", train.ID), load_split("test", test.ID)
    out = OUT / "emb_reg" / a.exp
    out.mkdir(parents=True, exist_ok=True)
    oof, tp = np.zeros(len(y)), np.zeros(len(test))
    t0 = time.time()
    done = []
    view = int(round(a.crop * 2 / 14)) * 14
    for k in (a.folds if a.folds is not None else range(5)):
        torch.manual_seed(42 + k)
        tr, va = fold != k, fold == k
        m = timm.create_model(a.model, pretrained=True, num_classes=0, img_size=view, dynamic_img_size=True).cuda()
        m.set_grad_checkpointing(True)
        cfg = m.pretrained_cfg
        mean = torch.tensor(cfg["mean"], device="cuda").view(1, 3, 1, 1)
        std = torch.tensor(cfg["std"], device="cuda").view(1, 3, 1, 1)
        head = nn.Sequential(nn.Linear(m.num_features, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                             nn.Linear(512, 128)).cuda()
        opt = torch.optim.AdamW([{"params": m.parameters(), "lr": a.lr},
                                 {"params": head.parameters(), "lr": a.lr * 50}], weight_decay=0.05)
        base_lr = [g["lr"] for g in opt.param_groups]
        Xtr = X[tr].cuda()
        n = len(Xtr)
        spe = max(1, n // a.batch)
        total = spe * a.epochs
        step = 0

        def prep(x):
            v = random_crop(augment(x.clone(), a), a.crop)
            v = F.interpolate(v, size=(view, view), mode="bilinear", align_corners=False)
            return (v.repeat(1, 3, 1, 1) - mean) / std

        for ep in range(a.epochs):
            m.train()
            head.train()
            perm = torch.randperm(n, device="cuda")
            tl = 0.0
            for b in range(spe):
                x = Xtr[perm[b * a.batch:(b + 1) * a.batch]].float().div_(255)
                f = min(1.0, (step + 1) / spe) * 0.5 * (1 + math.cos(math.pi * step / total))
                for g, bl in zip(opt.param_groups, base_lr):
                    g["lr"] = bl * f
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = nt_xent(head(m(prep(x))).float(), head(m(prep(x))).float(), a.temp)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                tl += loss.item()
                step += 1
            if (ep + 1) % 10 == 0:
                print(f"  fold{k} adapt ep{ep + 1} loss {tl / spe:.4f}", flush=True)
        del opt, head, Xtr
        torch.cuda.empty_cache()
        Etr, Eva, Ete = embed(m, mean, std, X[tr], a.layers), embed(m, mean, std, X[va], a.layers), \
            embed(m, mean, std, Xte, a.layers)
        rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 7, 60))).fit(Etr, y[tr])
        oof[va] = rg.predict(Eva)
        tp += rg.predict(Ete) / 5
        print(f"  fold{k} RMSE {np.sqrt(np.mean((oof[va] - y[va]) ** 2)):.3f} ({time.time() - t0:.0f}s)", flush=True)
        done.append(k)
        del m
        torch.cuda.empty_cache()
    msk = np.isin(fold, done)
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()
    r = np.sqrt(np.mean((oof[msk] - y[msk]) ** 2))
    rw = np.sqrt(np.mean(w[msk] * (oof[msk] - y[msk]) ** 2))
    if len(done) == 5:
        pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(out / "oof.csv", index=False)
        pd.DataFrame({"ID": test.ID, "hardness": tp}).to_csv(out / "test_pred.csv", index=False)
    print(f"[{a.exp}] OOF RMSE {r:.3f} | 재가중 {rw:.3f} | {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
