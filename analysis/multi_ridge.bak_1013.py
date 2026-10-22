"""여러 특징 묶음(고정 모델 임베딩 npz · 수작업 특징 csv)을 상대 분산 가중으로 한 Ridge에 결합 (block_ridge.py의 일반화).

각 묶음은 폴드 train 통계로 표준화(±6 절단)한 뒤, 묶음 전체 분산(커널 trace)이 첫 묶음의 G배가 되도록 배율을 곱한다.
--group 형식:
  emb:<probe2 폴더>:<G>[:<층 위치 목록, 쉼표>:<층당 채널 C>:<블록 수 nb>[:<블록 위치 목록>]]   예) emb:outputs/probe2/<tag>:1:1,2,3,4:1024:6
  csv:<features_cache 접두어>:<G>:<열 정규식>                              예) csv:outputs/features_cache/phase_<stem>:0.4:pf_
산출물: outputs/emb_reg/<name>/{oof.csv, test_pred.csv, test_pred_full.csv}
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
PUB_STD = 16.56


def load_group(spec, tr, te):
    kind, rest = spec.split(":", 1)
    if kind == "emb":
        parts = rest.split(":")
        path, g = parts[0], float(parts[1])
        z = np.load(ROOT / path / "emb.npz")
        A, B = z["train"].astype(np.float64), z["test"].astype(np.float64)
        if len(parts) > 2:
            layers, C, nb = [int(x) for x in parts[2].split(",")], int(parts[3]), int(parts[4])
            blocks = [int(x) for x in parts[5].split(",")] if len(parts) > 5 else list(range(nb))
            nl = A.shape[1] // (C * nb)
            A = A.reshape(len(A), nl, nb, C)[:, layers][:, :, blocks].reshape(len(A), -1)
            B = B.reshape(len(B), nl, nb, C)[:, layers][:, :, blocks].reshape(len(B), -1)
        return g, A, B, f"{Path(path).name[:40]} ({A.shape[1]})"
    pre, g, rx = rest.split(":", 2)
    A = pd.read_csv(ROOT / f"{pre}_train.csv").set_index("ID").loc[tr.ID]
    B = pd.read_csv(ROOT / f"{pre}_test.csv").set_index("ID").loc[te.ID]
    cols = [c for c in A.columns if re.match(rx, c)]
    f = lambda D: D[cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float64)
    return float(g), f(A), f(B), f"{Path(pre).name[:40]}::{rx} ({len(cols)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", required=True)
    ap.add_argument("--name", required=True)
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/sample_submission.csv")
    y = tr.hardness.values.astype(np.float64)
    fold = pd.read_csv(ROOT / "outputs/folds.csv").set_index("ID").loc[tr.ID, "fold"].values
    groups = [load_group(s, tr, te) for s in a.group]
    for g, A, _, desc in groups:
        print(f"  G={g:g} {desc}")

    def design(t):
        outA, outB, ref = [], [], None
        for g, A, B, _ in groups:
            med = np.nanmedian(A[t], 0)
            A2, B2 = np.where(np.isnan(A), med, A), np.where(np.isnan(B), med, B)
            mu, sd = A2[t].mean(0), A2[t].std(0)
            keep = sd > 1e-10
            ZA = np.clip((A2[:, keep] - mu[keep]) / sd[keep], -6, 6)
            ZB = np.clip((B2[:, keep] - mu[keep]) / sd[keep], -6, 6)
            if ref is None:
                ref, s = keep.sum(), 1.0
            else:
                s = np.sqrt(g * ref / keep.sum())
            outA.append(ZA * s)
            outB.append(ZB * s)
        return np.hstack(outA), np.hstack(outB)

    oof, tp = np.zeros(len(y)), np.zeros(len(te))
    for k in range(5):
        t, v = fold != k, fold == k
        A, B = design(t)
        r = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(A[t], y[t])
        oof[v] = r.predict(A[v])
        tp += r.predict(B) / 5
    A, B = design(np.ones(len(y), bool))
    tf = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(A, y).predict(B)
    d = ROOT / "outputs/emb_reg" / a.name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": tr.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
    pd.DataFrame({"ID": te.ID, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
    pd.DataFrame({"ID": te.ID, "hardness": tf}).to_csv(d / "test_pred_full.csv", index=False)
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w13 = np.exp(-0.5 * ((y - y.mean()) / 13) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    rm = lambda ww: np.sqrt(np.average((oof - y) ** 2, weights=ww))
    print(f"[{a.name}] OOF {rm(None):.3f} | 재가중 {rm(w):.3f} | pw13 {rm(w13):.3f}")


if __name__ == "__main__":
    main()
