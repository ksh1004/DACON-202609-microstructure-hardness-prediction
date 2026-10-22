"""그룹 평균 앙상블 제출 파일 생성: 각 그룹 = 여러 결과 폴더(oof.csv·test_pred.csv)의 단순 평균, 최종 = 그룹 가중 평균.

예) python src/make_submission.py --name v9 --group cnn outputs/cnn/s512_convnext_tiny_s42 ... --group dino outputs/emb_reg/dinoL_all12 ...
    (--weights 생략 시 그룹 동일 가중)
산출물: submissions/sub_<name>.csv  (OOF RMSE·재가중 CV 출력)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PUB_STD = 16.56

# 현재 최고 제출(v7) 구성 — 다른 스크립트에서 재사용
V7_CNN = [f"outputs/cnn/s512_{m}_s{s}" for m in ("convnext_tiny", "effnet_b2", "resnet50") for s in (42, 43, 44)]
V7_DINO = ["outputs/probe2/vit_large_patch14_dinov2.lvd142m_518_L1-3-5-7-9-11-13-15-17-19-21-23_T8",
           "outputs/emb_reg/dinoL_13-15-17-19", "outputs/emb_reg/dinoL_17-19",
           "outputs/probe2/vit_base_patch14_dinov2.lvd142m_518_L12"]


def load(d: str, ids_tr, ids_te, full=False):
    d = ROOT / d
    o = pd.read_csv(d / "oof.csv").set_index("ID").loc[ids_tr, "pred"].values
    f = d / "test_pred_full.csv"
    t = pd.read_csv(f if full and f.exists() else d / "test_pred.csv").set_index("ID").loc[ids_te, "hardness"].values
    return o, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--group", nargs="+", action="append", required=True,
                    help="그룹이름 폴더1 폴더2 ... (폴더 자리에 @v7cnn·@v7dino 쓰면 v7 구성으로 확장)")
    ap.add_argument("--weights", type=float, nargs="+")
    ap.add_argument("--full", action="store_true", help="test_pred_full.csv(전체 500장 재적합)가 있는 폴더는 그것을 test 예측으로 사용")
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/sample_submission.csv")
    y = tr.hardness.values
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()
    gw = np.array(a.weights if a.weights else [1.0] * len(a.group), dtype=float)
    gw /= gw.sum()
    oof, tp = np.zeros(len(y)), np.zeros(len(te))
    alias = {"@v7cnn": V7_CNN, "@v7dino": V7_DINO}
    for (gname, *dirs), g in zip(a.group, gw):
        dirs = [x for d in dirs for x in alias.get(d, [d])]
        parts = [load(d, tr.ID, te.ID, a.full) for d in dirs]
        go, gt = np.mean([p[0] for p in parts], 0), np.mean([p[1] for p in parts], 0)
        nfull = sum((ROOT / d / "test_pred_full.csv").exists() for d in dirs) if a.full else 0
        print(f"  {gname:12s} w={g:.3f} n={len(dirs)} OOF {np.sqrt(np.mean((go - y) ** 2)):.3f}"
              + (f" (전체적합 test {nfull}/{len(dirs)})" if a.full else ""))
        oof += g * go
        tp += g * gt
    out = ROOT / "submissions" / f"sub_{a.name}.csv"
    pd.DataFrame({"ID": te.ID, "hardness": tp}).to_csv(out, index=False)
    print(f"[{a.name}] OOF {np.sqrt(np.mean((oof - y) ** 2)):.3f} | 재가중 {np.sqrt(np.mean(w * (oof - y) ** 2)):.3f} "
          f"| test mean {tp.mean():.2f} std {tp.std():.2f} → {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
