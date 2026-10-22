"""특징 수준 노이즈 증강 Ridge: train 이미지에 노이즈를 더한 사본의 DINOv2 특징을 같은 정답으로 Ridge 학습에 추가.

- 목적: 노이즈가 큰 이미지(σ≥10)에서 고정 특징 Ridge가 약한 문제 → Ridge가 노이즈에 민감한 특징 방향을 덜 쓰도록
- 누수 방지: 폴드 k의 Ridge는 폴드 k 학습 이미지의 원본 + 그 사본만 사용 (검증 이미지의 사본은 사용하지 않음)
- 사본: 추가 가우시안 노이즈 σ ~ U(lo, hi)/255 (+ 선택적 블러), 이미지·사본마다 고정 시드
- test 예측: 원본 test 이미지 (증강 없음)

예) python src/probe_aug.py --model vit_large_patch14_dinov2.lvd142m --layers 13 15 17 19 --copies 2 --lo 4 --hi 20
산출물: outputs/emb_reg/aug_<...>/{oof.csv, test_pred.csv}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from torchvision.transforms.functional import gaussian_blur

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_models import build, embed  # noqa: E402
from train_cnn import DATA, OUT, get_folds, load_split  # noqa: E402

PUB_STD = 16.56


def noisy_copy(X: torch.Tensor, c: int, lo: float, hi: float, blur_p: float) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 + c)
    x = X.float() / 255
    sig = (lo + (hi - lo) * torch.rand(len(x), 1, 1, 1, generator=g)) / 255
    x = x + torch.randn(x.shape, generator=g) * sig
    if blur_p > 0:
        m = torch.rand(len(x), generator=g) < blur_p
        if m.any():
            x[m] = gaussian_blur(x[m], kernel_size=5, sigma=0.8)
    return (x.clamp(0, 1) * 255).round().to(torch.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--layers", type=int, nargs="+", default=[13, 15, 17, 19])
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--lo", type=float, default=4.0)
    ap.add_argument("--hi", type=float, default=20.0)
    ap.add_argument("--blur-p", type=float, default=0.0)
    ap.add_argument("--copy-weight", type=float, default=1.0, help="사본 표본 가중치")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = get_folds(train)
    y = train.hardness.values.astype(np.float64)
    X, Xte = load_split("train", train.ID), load_split("test", test.ID)
    t0 = time.time()
    m, mean, std = build(a.model, a.size)
    E0 = embed(m, mean, std, X, a.size, 4, a.layers, a.bs)
    Ete = embed(m, mean, std, Xte, a.size, 4, a.layers, a.bs)
    Ec = [embed(m, mean, std, noisy_copy(X, c, a.lo, a.hi, a.blur_p), a.size, 4, a.layers, a.bs) for c in range(a.copies)]
    print(f"  embeddings ready ({time.time() - t0:.0f}s)", flush=True)
    oof, oof0, tp = np.zeros(len(y)), np.zeros(len(y)), np.zeros(len(test))
    for k in range(5):
        tr, va = fold != k, fold == k
        Xa = np.vstack([E0[tr]] + [e[tr] for e in Ec])
        ya = np.concatenate([y[tr]] * (1 + a.copies))
        wa = np.concatenate([np.ones(tr.sum())] + [np.full(tr.sum(), a.copy_weight)] * a.copies)
        sc = StandardScaler().fit(E0[tr])
        rg = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(sc.transform(Xa), ya, sample_weight=wa)
        oof[va] = rg.predict(sc.transform(E0[va]))
        tp += rg.predict(sc.transform(Ete)) / 5
        rg0 = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(sc.transform(E0[tr]), y[tr])  # 비교용: 증강 없음
        oof0[va] = rg0.predict(sc.transform(E0[va]))
    sig = pd.read_csv(OUT / "features/train_features.csv").set_index("ID").loc[train.ID, "q_noise_sigma"].values
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()

    def rep(p):
        g = [np.sqrt(np.mean((p[mm] - y[mm]) ** 2)) for mm in (sig < 10, (sig >= 10) & (sig < 14), sig >= 14)]
        return (f"σ<10 {g[0]:.2f} | σ10~14 {g[1]:.2f} | σ≥14 {g[2]:.2f} | 전체 {np.sqrt(np.mean((p - y) ** 2)):.3f} "
                f"| 재가중 {np.sqrt(np.mean(w * (p - y) ** 2)):.3f}")

    tag = (f"aug_{a.model.split('.')[0]}_L{'-'.join(map(str, a.layers))}_c{a.copies}_n{a.lo:g}-{a.hi:g}"
           f"{'_b' + str(a.blur_p) if a.blur_p else ''}{'_w' + str(a.copy_weight) if a.copy_weight != 1 else ''}")
    d = OUT / "emb_reg" / tag
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
    pd.DataFrame({"ID": test.ID, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
    print(f"  증강 없음: {rep(oof0)}")
    print(f"[{tag}] 증강: {rep(oof)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
