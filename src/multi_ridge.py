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
from sklearn.linear_model import Ridge, RidgeCV

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
    ap.add_argument("--lq-gain", type=float, default=1.0,
                    help="저품질 모드(원본 노이즈 σ≥10) 이미지의 표준화된 임베딩 특징에 곱하는 배율 — 저품질 렌더링의 대비 압축(≈0.77배)으로 "
                         "특징 편차가 줄어든 것을 학습 전에 되돌린다 (이미지별 고정 연산, train·test 동일)")
    ap.add_argument("--lq-gain-csv", type=float, default=1.0, help="csv 묶음에 적용할 저품질 배율 (기본 1 = 미적용)")
    ap.add_argument("--lq-split", action="store_true",
                    help="모델 2개 적합: 고품질 이미지는 배율 없는 모델, 저품질 이미지는 --lq-gain 모델의 예측을 사용 "
                         "(배율을 키우면 저품질 예측은 좋아지고 고품질 예측은 조금 나빠지므로 모드별로 나눠 씀)")
    ap.add_argument("--alpha-rel", type=float, default=None,
                    help="Ridge 벌점을 LOO로 고르지 않고 고정: α = 값 × (중심화한 train 특징의 행별 제곱합 평균). "
                         "구성원 하나만 보면 LOO가 고른 α(상대값 0.04~0.15)가 최선이지만, 여러 구성원을 평균하는 블렌드에서는 "
                         "덜 정규화한 구성원(0.01)이 더 낫다 — 평균이 분산을 줄여 주고 구성원 간 오차 상관이 낮아진다")
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/sample_submission.csv")
    y = tr.hardness.values.astype(np.float64)
    fold = pd.read_csv(ROOT / "outputs/folds.csv").set_index("ID").loc[tr.ID, "fold"].values
    groups = [load_group(s, tr, te) for s in a.group]
    for g, A, _, desc in groups:
        print(f"  G={g:g} {desc}")
    # 저품질 모드 판정은 조명 보정(flatfield_cache.py)과 같은 기준: 원본 이미지의 노이즈 σ 추정값 ≥ 10
    lq = [pd.read_csv(ROOT / f"outputs/features/{n}_features.csv").set_index("ID").loc[d.ID, "q_noise_sigma"].values >= 10
          for n, d in (("train", tr), ("test", te))]
    gains = [a.lq_gain if s.startswith("emb:") else a.lq_gain_csv for s in a.group]
    if any(g != 1.0 for g in gains):
        print(f"  저품질 배율 {gains} (train {lq[0].sum()}장, test {lq[1].sum()}장)")

    def design(t):
        outA, outB, cols, ref, c0 = [], [], [], None, 0
        for (g, A, B, _), lg in zip(groups, gains):
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
            cols.append((c0, c0 + ZA.shape[1], lg))
            c0 += ZA.shape[1]
        return np.hstack(outA), np.hstack(outB), cols

    def gained(A, B, cols):  # 저품질 행의 표준화 특징에 묶음별 배율 적용
        A, B = A.copy(), B.copy()
        for c0, c1, lg in cols:
            if lg != 1.0:
                A[lq[0], c0:c1] *= lg
                B[lq[1], c0:c1] *= lg
        return A, B

    use_gain = any(g != 1.0 for g in gains)
    variants = [False, True] if (a.lq_split and use_gain) else [use_gain]
    oofs = {v_: np.zeros(len(y)) for v_ in variants}
    tps = {v_: np.zeros(len(te)) for v_ in variants}
    tfs = {}
    for k in list(range(5)) + [None]:  # None = 전체 500장 적합
        t = fold != k if k is not None else np.ones(len(y), bool)
        A0, B0, cols = design(t)
        for v_ in variants:
            A, B = gained(A0, B0, cols) if v_ else (A0, B0)
            if a.alpha_rel is None:
                r = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(A[t], y[t])
            else:
                r = Ridge(alpha=a.alpha_rel * ((A[t] - A[t].mean(0)) ** 2).sum() / t.sum()).fit(A[t], y[t])
            if k is None:
                tfs[v_] = r.predict(B)
            else:
                oofs[v_][fold == k] = r.predict(A[fold == k])
                tps[v_] += r.predict(B) / 5
    if len(variants) == 2:
        oof, tp, tf = np.where(lq[0], oofs[True], oofs[False]), np.where(lq[1], tps[True], tps[False]), np.where(lq[1], tfs[True], tfs[False])
    else:
        oof, tp, tf = oofs[variants[0]], tps[variants[0]], tfs[variants[0]]
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
