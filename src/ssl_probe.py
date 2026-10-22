"""자기지도 사전학습 백본(폴드별)의 고정 특징 + Ridge → OOF·test 예측 생성 (재학습 없음).

폴드 k의 백본은 폴드 k의 학습 이미지로만 사전학습되었으므로, 같은 폴드의 검증 예측에 누수가 없다.
test 예측은 5개 폴드 (백본+Ridge)의 평균. test 이미지는 추론에만 사용한다.

특징: 각 stage 특징맵의 채널별 평균·표준편차 (다중 스케일), D4 8방향 평균
실행: python src/ssl_probe.py --exp ssl_r18_ep200 --model resnet18
산출물: outputs/ssl/<exp>/{oof.csv, test_pred.csv}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cnn import DATA, OUT, d4, get_folds, load_split  # noqa: E402


@torch.no_grad()
def embed(m, X, mean, std, bs=32):
    m.eval()
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].cuda().float().div_(255)
        acc = 0
        for k in range(8):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                fs = m((d4(xb, k) - mean) / std)
            acc = acc + torch.cat([torch.cat([z.float().mean((2, 3)), z.float().std((2, 3))], 1) for z in fs], 1)
        out.append((acc / 8).cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model", required=True, help="timm 아키텍처 이름 (예: resnet18)")
    a = ap.parse_args()
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    fold = get_folds(train)
    X = load_split("train", train.ID)
    Xte = load_split("test", test.ID)
    y = train.hardness.values.astype(np.float64)
    d = OUT / "ssl" / a.exp
    oof = np.zeros(len(y))
    tp = np.zeros(len(test))
    for k in range(5):
        tr, va = fold != k, fold == k
        m = timm.create_model(a.model, pretrained=False, features_only=True, in_chans=1).cuda()
        sd = torch.load(d / f"fold{k}.pt", map_location="cpu", weights_only=False)
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing, missing[:5]
        Xtr = X[tr]
        mean = (Xtr.float().mean() / 255).item()
        std = (Xtr.float().std() / 255).item()
        Etr, Eva, Ete = embed(m, Xtr, mean, std), embed(m, X[va], mean, std), embed(m, Xte, mean, std)
        rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 40))).fit(Etr, y[tr])
        oof[va] = rg.predict(Eva)
        tp += rg.predict(Ete) / 5
        print(f"  fold{k} RMSE {np.sqrt(np.mean((oof[va] - y[va]) ** 2)):.3f}", flush=True)
        del m
        torch.cuda.empty_cache()
    r = float(np.sqrt(np.mean((oof - y) ** 2)))
    pd.DataFrame({"ID": train.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
    pd.DataFrame({"ID": test.ID, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
    print(f"[{a.exp}] multi-scale probe OOF RMSE {r:.3f}")


if __name__ == "__main__":
    main()
