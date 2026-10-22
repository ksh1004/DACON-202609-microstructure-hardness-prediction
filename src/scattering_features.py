"""웨이블릿 산란 변환(Scattering2D, Kymatio — BSD-3, 학습 없는 고정 변환) 질감 특징 + Ridge.

- J 스케일 × L 방향의 1·2차 산란 계수 → 채널별 공간 평균·표준편차 (log) → D4 4방향 평균
- 사전학습 모델이 아니므로 라이선스·외부 데이터 이슈 없음
- 주의: kymatio 0.3.0은 최신 scipy에서 3D 모듈 import가 실패하므로 2D 프런트엔드를 직접 import

실행: python src/scattering_features.py --J 5 --L 8   (CPU 약 10분, --device cuda 가능)
산출물: outputs/probe2/scattering_J{J}_L{L}/{emb.npz, oof.csv, test_pred.csv}
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from kymatio.scattering2d.frontend.torch_frontend import ScatteringTorch2D
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cnn import DATA, OUT, d4, get_folds, load_split  # noqa: E402

PUB_STD = 16.56


@torch.no_grad()
def feats(S, X, device, bs=16):
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].to(device).float().div_(255)
        acc = 0
        for k in (0, 1, 4, 5):
            c = S(d4(xb, k)[:, 0].contiguous())  # B, K, h, w (회전·전치 후 연속 메모리 필요)
            f = torch.cat([torch.log(c.mean((2, 3)) + 1e-6), torch.log(c.std((2, 3)) + 1e-6)], 1)
            acc = acc + f
        out.append((acc / 4).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=int, default=5)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = get_folds(train)
    y = train.hardness.values.astype(np.float64)
    X, Xte = load_split("train", train.ID), load_split("test", test.ID)
    S = ScatteringTorch2D(J=a.J, shape=(256, 256), L=a.L, max_order=2)
    if a.device == "cuda":
        S = S.cuda()
    t0 = time.time()
    Etr, Ete = feats(S, X, a.device), feats(S, Xte, a.device)
    oof, tp = np.zeros(len(y)), np.zeros(len(test))
    for k in range(5):
        tr, va = fold != k, fold == k
        rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 6, 50))).fit(Etr[tr], y[tr])
        oof[va] = rg.predict(Etr[va])
        tp += rg.predict(Ete) / 5
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()
    d = OUT / "probe2" / f"scattering_J{a.J}_L{a.L}"
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "emb.npz", train=Etr.astype(np.float32), test=Ete.astype(np.float32))
    pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
    pd.DataFrame({"ID": test.ID, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
    print(f"[scattering_J{a.J}_L{a.L}] dim={Etr.shape[1]} CV {np.sqrt(np.mean((oof - y) ** 2)):.3f} | "
          f"재가중 {np.sqrt(np.mean(w * (oof - y) ** 2)):.3f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
