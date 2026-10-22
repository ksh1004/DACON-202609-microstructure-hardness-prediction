"""최종 블록 앙상블 구성 비교 & 제출 파일 생성.

블록 = 같은 모델의 시드 평균 (OOF·test 모두). 블록 간 결합은 단순 평균(가중치 학습 없음 — reports/model_catalog_ensemble.md §2.3).
OOF 단계에서는 각 샘플을 폴드 모델 1개(×시드 수)가 예측하지만 test는 5개 폴드 모델 평균이라 CNN이 test에서 더 강하다
(제출 v2 교훈). 따라서 CNN 비중이 큰 구성과 작은 구성을 함께 보고 LB로 교차 확인한다.

실행: python src/build_ensemble.py                 → 구성별 OOF RMSE 표
      python src/build_ensemble.py --write cnn_heavy --name sub_v3   → submissions/sub_v3.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
SEEDS = (42, 43, 44)


def _pair(oof_path: Path, test_path: Path, test_col="hardness"):
    o = pd.read_csv(oof_path)
    t = pd.read_csv(test_path)
    return o.pred.values, t[test_col].values


def cnn_block(prefix: str):
    """final_<prefix>_s<seed> 중 완료된 시드를 평균."""
    oo, tt, used = [], [], []
    for s in SEEDS:
        d = OUT / "cnn" / f"final_{prefix}_s{s}"
        if (d / "oof.csv").exists() and (d / "test_pred.csv").exists():
            o, t = _pair(d / "oof.csv", d / "test_pred.csv")
            oo.append(o)
            tt.append(t)
            used.append(s)
    if not oo:
        return None
    return np.mean(oo, 0), np.mean(tt, 0), used


def feat_block(names):
    pairs = [_pair(OUT / f"feat_{n}" / "oof.csv", OUT / f"feat_{n}" / "test_pred.csv") for n in names]
    return np.mean([p[0] for p in pairs], 0), np.mean([p[1] for p in pairs], 0)


def probe_block(models):
    pairs = [_pair(OUT / "probe" / m / "oof_emb.csv", OUT / "probe" / m / "test_emb.csv") for m in models]
    return np.mean([p[0] for p in pairs], 0), np.mean([p[1] for p in pairs], 0)


def load_blocks():
    base = pd.read_csv(OUT / "feat_ridge" / "oof.csv")
    y = base.hardness.values
    B, info = {}, {}
    B["ridge"] = feat_block(["ridge"])
    B["kernel"] = feat_block(["kernel_ridge", "gp", "svr"])
    B["probe"] = probe_block(["resnet50.a1_in1k", "vit_small_patch14_dinov2.lvd142m"])
    for name in ["resnet18", "convnext", "effnet"]:
        r = cnn_block(name)
        if r is not None:
            B[name] = r[:2]
            info[name] = r[2]
    return B, info, y


CONFIGS = {
    "cnn_only": ["resnet18", "convnext", "effnet"],
    "cnn_heavy": ["ridge", "resnet18", "convnext", "effnet"],            # CNN 75% (v1과 같은 비중)
    "balanced6": ["ridge", "kernel", "resnet18", "convnext", "effnet", "probe"],  # CNN 50% (v2 구조)
    "feat_cnn_half": None,  # (특징 2블록 평균 + CNN 3블록 평균) / 2
}


def combine(B, keys):
    if keys is None:
        f = np.mean([B["ridge"][0], B["kernel"][0]], 0), np.mean([B["ridge"][1], B["kernel"][1]], 0)
        c = (np.mean([B[k][0] for k in ["resnet18", "convnext", "effnet"]], 0),
             np.mean([B[k][1] for k in ["resnet18", "convnext", "effnet"]], 0))
        return (f[0] + c[0]) / 2, (f[1] + c[1]) / 2
    return np.mean([B[k][0] for k in keys], 0), np.mean([B[k][1] for k in keys], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", choices=list(CONFIGS), default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    B, info, y = load_blocks()
    rmse = lambda p: float(np.sqrt(np.mean((p - y) ** 2)))  # noqa: E731
    print("== 블록 (시드 평균)")
    for k, (o, _) in B.items():
        extra = f"  seeds={info[k]}" if k in info else ""
        print(f"  {k:10s} OOF {rmse(o):.3f}{extra}")
    print("== 구성")
    for name, keys in CONFIGS.items():
        o, _ = combine(B, keys)
        print(f"  {name:14s} OOF {rmse(o):.3f}")
    if a.write:
        ss = pd.read_csv(ROOT / "data" / "sample_submission.csv")
        _, t = combine(B, CONFIGS[a.write])
        sub = ss.copy()
        sub["hardness"] = t
        assert len(sub) == 1000 and sub.hardness.notna().all()
        path = ROOT / "submissions" / f"{a.name or 'sub_' + a.write}.csv"
        sub.to_csv(path, index=False)
        print("saved ->", path, sub.hardness.describe().round(2).to_dict())


if __name__ == "__main__":
    main()
