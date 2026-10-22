"""고정 폴드 OOF 예측들의 앙상블 평가 — 가중치를 '다른 폴드'에서 학습하는 중첩 CV로 낙관 편향 제거.

실행: python src/ensemble_eval.py
입력: outputs/feat_ridge/oof.csv, outputs/cnn/<exp>/oof.csv, outputs/probe/<model>/oof_emb.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

OUT = Path(__file__).resolve().parents[1] / "outputs"


def load_all() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    base = pd.read_csv(OUT / "feat_ridge" / "oof.csv")
    P = {"ridge_hc": base.pred.values}
    for d in sorted((OUT / "cnn").glob("scr_*")):
        if (d / "oof.csv").exists():
            P[d.name.replace("scr_", "ft_")] = pd.read_csv(d / "oof.csv").pred.values
    for d in sorted((OUT / "probe").glob("*")):
        f = d / "oof_emb.csv"
        if f.exists():
            P["probe_" + d.name.split(".")[0]] = pd.read_csv(f).pred.values
    return pd.DataFrame(P), base.hardness.values, base.fold.values


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def nested_nnls(X: np.ndarray, y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """폴드 k의 블렌드 가중치는 나머지 폴드 OOF로만 추정 (절편 없는 비음수 가중, 합=1 정규화)."""
    out = np.zeros(len(y))
    for k in np.unique(fold):
        a, b = fold != k, fold == k
        w, _ = nnls(X[a], y[a])
        w = w / w.sum() if w.sum() > 0 else np.full(X.shape[1], 1 / X.shape[1])
        out[b] = X[b] @ w
    return out


def main():
    P, y, fold = load_all()
    print("== 단일 모델 OOF RMSE")
    for c in P:
        print(f"  {c:28s} {rmse(P[c].values, y):.3f}")
    print("\n== 상관행렬")
    print(P.corr().round(2).to_string())
    print("\n== 앙상블 (중첩 CV)")
    groups = {
        "ridge+ft(all)": ["ridge_hc"] + [c for c in P if c.startswith("ft_")],
        "ft(all) only": [c for c in P if c.startswith("ft_")],
        "ridge+probe(all)": ["ridge_hc"] + [c for c in P if c.startswith("probe_")],
        "all": list(P.columns),
    }
    for name, cols in groups.items():
        if len(cols) < 2:
            continue
        X = P[cols].values
        print(f"  {name:22s} 단순평균 {rmse(X.mean(1), y):.3f} | NNLS(중첩) {rmse(nested_nnls(X, y, fold), y):.3f}")
    w, _ = nnls(P.values, y)
    print("\n== 전체 데이터 NNLS 가중치 (참고용)")
    print(pd.Series(w / w.sum(), index=P.columns).round(3).sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
