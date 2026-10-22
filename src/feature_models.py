"""수작업 특징(106개 중 nuisance 제외 103개) 기반 회귀 모델 비교 — 고정 폴드, 내부 CV 튜닝.

실행: python src/feature_models.py [--models ridge svr ...]
산출물: outputs/feat_<name>/{oof.csv, test_pred.csv}, outputs/feature_model_results.csv
주의: GridSearchCV n_jobs는 쓰지 않음 (joblib loky가 한글 경로에서 실패)
"""
from __future__ import annotations

import argparse
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import QuantileTransformer, SplineTransformer, StandardScaler
from sklearn.svm import SVR

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
INNER = KFold(4, shuffle=True, random_state=0)


def qt():
    return QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=0)


def scaled_y(model):
    return TransformedTargetRegressor(regressor=model, transformer=StandardScaler())


MODELS = {
    "ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
    "ridge_qt": lambda: make_pipeline(qt(), RidgeCV(alphas=np.logspace(-2, 4, 30))),
    "elasticnet": lambda: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=INNER,
                                                                         max_iter=20000, random_state=0)),
    "pls": lambda: GridSearchCV(make_pipeline(StandardScaler(), PLSRegression()),
                                {"plsregression__n_components": [2, 4, 6, 8, 12]}, cv=INNER,
                                scoring="neg_root_mean_squared_error"),
    "spline_ridge": lambda: make_pipeline(qt(), SplineTransformer(n_knots=4, degree=3),
                                          RidgeCV(alphas=np.logspace(0, 5, 30))),
    "svr": lambda: GridSearchCV(make_pipeline(qt(), scaled_y(SVR())),
                                {"transformedtargetregressor__regressor__C": [0.3, 1, 3, 10],
                                 "transformedtargetregressor__regressor__gamma": [0.001, 0.003, 0.01],
                                 "transformedtargetregressor__regressor__epsilon": [0.1, 0.3]},
                                cv=INNER, scoring="neg_root_mean_squared_error"),
    "kernel_ridge": lambda: GridSearchCV(make_pipeline(qt(), scaled_y(KernelRidge(kernel="rbf"))),
                                         {"transformedtargetregressor__regressor__alpha": [0.1, 0.3, 1, 3],
                                          "transformedtargetregressor__regressor__gamma": [0.0005, 0.001, 0.003, 0.01]},
                                         cv=INNER, scoring="neg_root_mean_squared_error"),
    "gp": lambda: make_pipeline(qt(), GaussianProcessRegressor(
        ConstantKernel(1.0) * RBF(length_scale=20.0, length_scale_bounds=(1, 1e3)) + WhiteKernel(0.5),
        normalize_y=True, n_restarts_optimizer=2, random_state=0)),
    "mlp": lambda: make_pipeline(qt(), scaled_y(MLPRegressor(hidden_layer_sizes=(64,), alpha=1.0, learning_rate_init=1e-3,
                                                             max_iter=2000, early_stopping=True,
                                                             validation_fraction=0.15, random_state=0))),
    "lgbm": lambda: LGBMRegressor(n_estimators=700, learning_rate=0.02, num_leaves=15, min_child_samples=12,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                                  verbose=-1, random_state=0),
    "extratrees": lambda: ExtraTreesRegressor(800, min_samples_leaf=2, max_features=0.5, n_jobs=-1, random_state=0),
    "random_forest": lambda: RandomForestRegressor(800, min_samples_leaf=2, max_features=0.4, n_jobs=-1,
                                                   random_state=0),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data" / "train.csv").merge(pd.read_csv(OUT / "folds.csv"), on="ID")
    ftr = pd.read_csv(OUT / "features" / "train_features.csv").set_index("ID").loc[tr.ID]
    fte = pd.read_csv(OUT / "features" / "test_features.csv")
    feats = [c for c in ftr.columns if not c.startswith("nui_")]
    X, Xt, y, fold = ftr[feats].values, fte[feats].values, tr.hardness.values, tr.fold.values
    res_path = OUT / "feature_model_results.csv"
    rows = pd.read_csv(res_path).to_dict("records") if res_path.exists() else []
    for name in a.models:
        t = time.time()
        oof, tp = np.zeros(len(y)), np.zeros(len(Xt))
        for k in range(5):
            tri, vai = fold != k, fold == k
            m = MODELS[name]().fit(X[tri], y[tri])
            oof[vai] = np.ravel(m.predict(X[vai]))
            tp += np.ravel(m.predict(Xt)) / 5
        r = float(np.sqrt(np.mean((oof - y) ** 2)))
        d = OUT / f"feat_{name}"
        d.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ID": tr.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
        pd.DataFrame({"ID": fte.ID, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
        rows = [x for x in rows if x["model"] != name] + [{"model": name, "oof_rmse": round(r, 3),
                                                            "sec": round(time.time() - t, 1)}]
        print(f"{name:14s} OOF RMSE {r:.3f}  ({time.time() - t:.1f}s)", flush=True)
        pd.DataFrame(rows).sort_values("oof_rmse").to_csv(res_path, index=False)  # 모델마다 저장


if __name__ == "__main__":
    main()
