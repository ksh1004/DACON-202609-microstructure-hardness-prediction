"""이미지 캐시(npz, 예: U-Net 디노이즈 결과)에서 수작업 특징(features.py 106개 + structure_features.py 279개) 추출 → Ridge 5-Fold.

디노이즈 이미지에서는 고노이즈 이미지도 결정립 분할·상 분율 측정이 가능해진다. 각 모듈은 내부에서 σ를 추정해
추가 디노이즈를 하지만, 이미 깨끗한 입력이면 σ가 작아 거의 그대로 통과한다. 이미지별 고정 연산(학습 없음).

실행: python src/features_from_cache.py --src outputs/denoised/unet_cm7_am22_l1_i6000.npz --workers 8
산출물: outputs/features_cache/<src>_{train,test}.csv, outputs/emb_reg/hc_<src>/{oof.csv, test_pred.csv}
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
PUB_STD = 16.56


def _one(img: np.ndarray) -> dict:
    from features import extract_features
    from structure_features import extract
    f = extract_features(img)
    f2, _ = extract(img)
    f.update(f2)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/sample_submission.csv")
    z = np.load(ROOT / a.src)
    stem = Path(a.src).stem
    fd = ROOT / "outputs/features_cache"
    fd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    F = {}
    for split, ids in (("train", tr.ID), ("test", te.ID)):
        p = fd / f"{stem}_{split}.csv"
        if p.exists():
            F[split] = pd.read_csv(p)
            continue
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(_one, list(z[split]), chunksize=8))
        F[split] = pd.DataFrame(rows).assign(ID=ids.values)
        F[split].to_csv(p, index=False)
        print(f"  {split} 특징 {F[split].shape} ({time.time() - t0:.0f}s)", flush=True)
    cols = [c for c in F["train"].columns if c != "ID" and not c.startswith("nui_")]
    Xtr = F["train"][cols].replace([np.inf, -np.inf], np.nan)
    med = Xtr.median()
    Xtr = Xtr.fillna(med).values
    Xte = F["test"][cols].replace([np.inf, -np.inf], np.nan).fillna(med).values
    y = tr.hardness.values.astype(float)
    fold = pd.read_csv(ROOT / "outputs/folds.csv").set_index("ID").loc[tr.ID, "fold"].values
    oof, tp = np.zeros(len(y)), np.zeros(len(te))
    for k in range(5):
        t, v = fold != k, fold == k
        rg = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 40))).fit(Xtr[t], y[t])
        oof[v] = rg.predict(Xtr[v])
        tp += rg.predict(Xte) / 5
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()
    d = ROOT / "outputs/emb_reg" / f"hc_{stem}"
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": tr.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
    pd.DataFrame({"ID": te.ID, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
    print(f"[hc_{stem}] 특징 {len(cols)}개 Ridge OOF {np.sqrt(np.mean((oof - y) ** 2)):.3f} | "
          f"재가중 {np.sqrt(np.mean(w * (oof - y) ** 2)):.3f} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
