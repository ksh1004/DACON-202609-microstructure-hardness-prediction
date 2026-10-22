"""저장된 고정 특징(outputs/probe2/*/emb.npz)으로 회귀 모델·특징 결합 비교 (CPU, 수 초~수십 초).

- 회귀: Ridge(기본), Kernel Ridge(RBF), PLS, SVR — 하이퍼파라미터는 각 폴드 내부 CV로만 선택
- 특징 결합: 여러 emb.npz를 이어 붙임 (각각 표준화 후 블록 크기로 나눠 블록별 비중을 맞춤)
- test 예측: (a) 5개 폴드 모델 평균, (b) train 500장 전체로 재학습 — 둘 다 저장
- 평가: 일반 CV + Public 분산(16.56)에 맞춘 재가중 CV

예) python src/emb_regress.py --embs vit_base_patch14_dinov2.lvd142m_518_L12_T8 --models ridge krr
산출물: outputs/emb_reg/<tag>/{oof.csv, test_pred.csv, test_pred_full.csv}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
INNER = KFold(4, shuffle=True, random_state=0)
PUB_STD = 16.56


def ty(model):
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())


def make(name, g):
    """g: RBF 기본 gamma = 1 / (샘플 간 제곱거리 중앙값) — 특징 스케일과 무관하게 커널 폭을 맞춤."""
    return {
        "ridge": lambda: RidgeCV(alphas=np.logspace(-1, 7, 60)),
        "krr": lambda: GridSearchCV(ty(KernelRidge(kernel="rbf")),
                                    {"regressor__alpha": [0.03, 0.1, 0.3, 1.0],
                                     "regressor__gamma": [g * 0.1, g * 0.3, g, g * 3]},
                                    cv=INNER, scoring="neg_root_mean_squared_error"),
        "pls": lambda: GridSearchCV(PLSRegression(scale=False), {"n_components": [4, 8, 16, 32]},
                                    cv=INNER, scoring="neg_root_mean_squared_error"),
        "svr": lambda: GridSearchCV(ty(SVR(kernel="rbf")),
                                    {"regressor__C": [1, 3, 10], "regressor__gamma": [g * 0.3, g, g * 3],
                                     "regressor__epsilon": [0.1, 0.3]},
                                    cv=INNER, scoring="neg_root_mean_squared_error"),
    }[name]()


def load(tags):
    Xtr, Xte = [], []
    for t in tags:
        z = np.load(OUT / "probe2" / t / "emb.npz")
        sc = StandardScaler().fit(z["train"])  # 비지도 표준화 (정답 미사용) → 블록 크기로 나눠 블록 비중 균등화
        k = np.sqrt(z["train"].shape[1])
        Xtr.append(sc.transform(z["train"]) / k)
        Xte.append(sc.transform(z["test"]) / k)
    return np.hstack(Xtr), np.hstack(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embs", nargs="+", required=True, help="outputs/probe2 아래 태그들 (여러 개면 결합)")
    ap.add_argument("--models", nargs="+", default=["ridge"])
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    base = pd.read_csv(OUT / "feat_ridge" / "oof.csv")
    y, fold = base.hardness.values, base.fold.values
    ids_te = pd.read_csv(OUT / "probe2" / a.embs[0] / "test_pred.csv").ID
    Xtr, Xte = load(a.embs)
    sub = Xtr[np.random.default_rng(0).choice(len(Xtr), 200, replace=False)]
    d2 = ((sub[:, None, :] - sub[None, :, :]) ** 2).sum(-1)
    g = 1.0 / np.median(d2[np.triu_indices(len(sub), 1)])
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()
    for mname in a.models:
        oof, tp = np.zeros(len(y)), np.zeros(len(Xte))
        for k in range(5):
            tr, va = fold != k, fold == k
            m = make(mname, g).fit(Xtr[tr], y[tr])
            oof[va] = np.ravel(m.predict(Xtr[va]))
            tp += np.ravel(m.predict(Xte)) / 5
        full = np.ravel(make(mname, g).fit(Xtr, y).predict(Xte))
        r, rw = np.sqrt(np.mean((oof - y) ** 2)), np.sqrt(np.mean(w * (oof - y) ** 2))
        tag = (a.name or "+".join(a.embs)) + f"__{mname}"
        d = OUT / "emb_reg" / tag
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ID": base.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
        pd.DataFrame({"ID": ids_te, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
        pd.DataFrame({"ID": ids_te, "hardness": full}).to_csv(d / "test_pred_full.csv", index=False)
        print(f"{tag[:90]:90s} CV {r:.3f} | 재가중CV {rw:.3f}", flush=True)


if __name__ == "__main__":
    main()
