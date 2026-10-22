"""타일 확대 프로브: 이미지(디노이즈 캐시)를 2×2 사분면으로 나눠 각각 --size로 확대 → DINO 고정 특징(풀링) →
사분면 평균(+ 사분면 간 표준편차) → Ridge 5-Fold. 유효 해상도 2배(미세 결정립 관찰), 원본에선 노이즈도 확대돼 불리했으나 디노이즈 입력에선 다를 수 있음.

예) python src/probe_tiles.py --src outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz --layers 11 13 15 17
산출물: outputs/probe2/tiles_<model>_<size>_L.._P.._S<src>/{oof.csv, test_pred.csv, emb.npz}
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_models import build, embed  # noqa: E402
from train_cnn import DATA, OUT, get_folds  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--model", default="vit_large_patch14_dinov2.lvd142m")
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--layers", type=int, nargs="+", default=[11, 13, 15, 17])
    ap.add_argument("--pool", default="hetero3")
    ap.add_argument("--bs", type=int, default=8)
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = get_folds(train)
    y = train.hardness.values.astype(np.float64)
    z = np.load(ROOT / a.src)
    X, Xte = torch.from_numpy(z["train"]).unsqueeze(1), torch.from_numpy(z["test"]).unsqueeze(1)
    t0 = time.time()
    m, mean, std = build(a.model, a.size)
    H = X.shape[-1] // 2

    def feats(T):
        q = [embed(m, mean, std, T[..., i * H:(i + 1) * H, j * H:(j + 1) * H].contiguous(), a.size, 4, a.layers, a.bs, a.pool)
             for i in (0, 1) for j in (0, 1)]
        q = np.stack(q)  # 4, N, D
        return np.concatenate([q.mean(0), q.std(0)], 1)

    Etr, Ete = feats(X), feats(Xte)
    sec = time.time() - t0
    oof, tp = np.zeros(len(y)), np.zeros(len(test))
    D = Etr.shape[1] // 2
    res = {}
    for name, sl in (("mean", slice(0, D)), ("mean+std", slice(0, 2 * D))):
        o, t = np.zeros(len(y)), np.zeros(len(test))
        for k in range(5):
            tr, va = fold != k, fold == k
            rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 7, 60))).fit(Etr[tr][:, sl], y[tr])
            o[va] = rg.predict(Etr[va][:, sl])
            t += rg.predict(Ete[:, sl]) / 5
        res[name] = (o, t)
        print(f"  사분면 {name}: OOF {np.sqrt(np.mean((o - y) ** 2)):.3f}", flush=True)
    best = min(res, key=lambda k: np.mean((res[k][0] - y) ** 2))
    oof, tp = res[best]
    tag = f"tiles_{a.model}_{a.size}_L{'-'.join(map(str, a.layers))}_P{a.pool}_S{Path(a.src).stem}"
    d = OUT / "probe2" / tag
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "emb.npz", train=Etr.astype(np.float32), test=Ete.astype(np.float32))
    pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
    pd.DataFrame({"ID": test.ID, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
    print(f"[{tag}] 선택 {best} OOF {np.sqrt(np.mean((oof - y) ** 2)):.3f} ({sec:.0f}s)")


if __name__ == "__main__":
    main()
