"""저장된 다층 특징(emb.npz)에서 일부 층만 골라 Ridge → OOF·test 예측 (CPU).

emb 레이아웃: 층마다 [채널 평균(C), 채널 표준편차(C)] 를 이어 붙인 형태 (probe_models.py, pool=meanstd).
--per-layer 를 주면 층별 단독 성능을 출력한다.

예) python src/layer_select.py --emb vit_large_patch14_dinov2.lvd142m_518_L1-3-5-7-9-11-13-15-17-19-21-23_T8 \
        --all-layers 1 3 5 7 9 11 13 15 17 19 21 23 --use 13 15 17 19 --name dinoL_13-19
산출물: outputs/emb_reg/<name>/{oof.csv, test_pred.csv, test_pred_full.csv}
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(__file__).resolve().parents[1] / "outputs"
PUB_STD = 16.56


def ridge():
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 7, 60)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--all-layers", type=int, nargs="+", required=True, help="emb에 저장된 층 번호 순서")
    ap.add_argument("--use", type=int, nargs="*", default=None)
    ap.add_argument("--per-layer", action="store_true")
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    base = pd.read_csv(OUT / "feat_ridge" / "oof.csv")
    y, fold = base.hardness.values, base.fold.values
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()
    z = np.load(OUT / "probe2" / a.emb / "emb.npz")
    E, Et = z["train"], z["test"]
    nl = len(a.all_layers)
    C = E.shape[1] // (2 * nl)
    assert C * 2 * nl == E.shape[1], "meanstd 레이아웃이 아님"

    def cols(sel):
        return np.concatenate([np.arange(a.all_layers.index(l) * 2 * C, (a.all_layers.index(l) + 1) * 2 * C) for l in sel])

    def cv(X):
        oof = np.zeros(len(y))
        for k in range(5):
            tr, va = fold != k, fold == k
            oof[va] = ridge().fit(X[tr], y[tr]).predict(X[va])
        return oof

    if a.per_layer:
        res = []
        for l in a.all_layers:
            o = cv(E[:, cols([l])])
            res.append(f"{l}:{np.sqrt(np.mean((o - y) ** 2)):.2f}")
        print("층별 단독 CV:", " ".join(res))
    if a.use:
        c = cols(a.use)
        X, Xt = E[:, c], Et[:, c]
        oof = cv(X)
        tp = np.mean([ridge().fit(X[fold != k], y[fold != k]).predict(Xt) for k in range(5)], 0)
        full = ridge().fit(X, y).predict(Xt)
        name = a.name or f"{a.emb}__use{'-'.join(map(str, a.use))}"
        d = OUT / "emb_reg" / name
        d.mkdir(parents=True, exist_ok=True)
        ids = pd.read_csv(OUT / "probe2" / a.emb / "test_pred.csv").ID
        pd.DataFrame({"ID": base.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
        pd.DataFrame({"ID": ids, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
        pd.DataFrame({"ID": ids, "hardness": full}).to_csv(d / "test_pred_full.csv", index=False)
        print(f"[{name}] 층 {a.use}: CV {np.sqrt(np.mean((oof - y) ** 2)):.3f} | 재가중 {np.sqrt(np.mean(w * (oof - y) ** 2)):.3f}")


if __name__ == "__main__":
    main()
