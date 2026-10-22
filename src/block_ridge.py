"""블록 가중 Ridge: emb.npz(층 × 통계블록 × 채널 배치)를 블록별 표준화 후 블록마다 배율을 곱해 Ridge.

배율 < 1 이면 그 블록에 더 강한 규제를 주는 효과 (예: 불균일성 블록은 평균·표준편차보다 잡음이 많아 0.5배가 유리).
probe_models.py 풀링별 블록 순서: meanstd=[mean,std] / hetero=[mean,std,h2,h4] / hetero3=[mean,std,h2,h4,h8,hstd]

예) python src/block_ridge.py --src outputs/probe2/<tag> --nb 6 --mult 1 1 .5 .5 .5 .5 --name dinoL1319_het3_m05
--extra G SPEC...: 수작업 특징 그룹 추가 (SPEC = <features_cache csv 접두어>::<열 정규식>, 접두어+_train/_test.csv).
  폴드별 표준화(±6 절단) 후 그룹 전체 분산(커널 trace)이 DINO 부분의 G배가 되도록 배율 → 같은 Ridge에 결합.
  예) --extra 0.4 outputs/features_cache/phase_<stem>::pf_ outputs/features_cache/<stem>::ph_
산출물: outputs/emb_reg/<name>/{oof.csv, test_pred.csv}
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PUB_STD = 16.56


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="emb.npz가 있는 폴더")
    ap.add_argument("--nb", type=int, required=True, help="층당 통계 블록 수")
    ap.add_argument("--mult", type=float, nargs="+", required=True, help="블록별 배율 (nb개)")
    ap.add_argument("--layers", type=int, nargs="+", help="사용할 층 위치(0부터, emb 안의 순서). 생략 시 전체")
    ap.add_argument("--C", type=int, default=0, help="층당 채널 수 (생략 시 폴더 이름으로: ViT-S 384/B 768/L 1024/g 1536)")
    ap.add_argument("--name", required=True)
    ap.add_argument("--extra", nargs="+", action="append", default=[], metavar="G SPEC",
                    help="수작업 특징 그룹: 상대 분산 G 다음 '<csv 접두어>::<열 정규식>' 1개 이상 (여러 번 지정 가능)")
    ap.add_argument("--full", action="store_true", help="test 예측을 전체 500장 재적합으로도 저장 (test_pred_full.csv)")
    a = ap.parse_args()
    assert len(a.mult) == a.nb
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/sample_submission.csv")
    y = tr.hardness.values.astype(np.float64)
    fold = pd.read_csv(ROOT / "outputs/folds.csv").set_index("ID").loc[tr.ID, "fold"].values
    z = np.load(ROOT / a.src / "emb.npz")
    Etr, Ete = z["train"].astype(np.float64), z["test"].astype(np.float64)
    C = a.C or {"vit_s": 384, "vit_b": 768, "vit_l": 1024, "vit_g": 1536}[Path(a.src).name[:5]]
    nl = Etr.shape[1] // (a.nb * C)
    assert nl * a.nb * C == Etr.shape[1], f"차원 불일치: {Etr.shape[1]} ≠ 층{nl}×블록{a.nb}×C{C}"
    R = lambda E: E.reshape(len(E), nl, a.nb, C)[:, a.layers] if a.layers else E.reshape(len(E), nl, a.nb, C)
    Xtr, Xte = R(Etr), R(Ete)
    nl_used = Xtr.shape[1]
    m = np.tile(np.repeat(np.array(a.mult), C), nl_used)
    Xtr, Xte = Xtr.reshape(len(Xtr), -1), Xte.reshape(len(Xte), -1)
    groups = []  # (상대분산 G, train 특징, test 특징)
    for g, *specs in a.extra:
        ftr, fte = [], []
        for sp in specs:
            pre, rx = sp.split("::")
            A = pd.read_csv(ROOT / f"{pre}_train.csv").set_index("ID").loc[tr.ID]
            B = pd.read_csv(ROOT / f"{pre}_test.csv").set_index("ID").loc[te.ID]
            cols = [c for c in A.columns if re.match(rx, c)]
            ftr.append(A[cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float64))
            fte.append(B[cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float64))
        groups.append((float(g), np.hstack(ftr), np.hstack(fte)))
        print(f"  추가 그룹 G={g}: {groups[-1][1].shape[1]}개 특징")
    dvar = float(np.sum(m ** 2))

    def design(t):
        """train 마스크 t의 통계로 변환한 (train 전체, test) 설계행렬"""
        sc = StandardScaler().fit(Xtr[t])
        A, B = [sc.transform(Xtr) * m], [sc.transform(Xte) * m]
        for g, F, Ft in groups:
            med = np.nanmedian(F[t], 0)
            F2, Ft2 = np.where(np.isnan(F), med, F), np.where(np.isnan(Ft), med, Ft)
            mu, sd = F2[t].mean(0), F2[t].std(0)
            keep = sd > 1e-10
            s_ = np.sqrt(g * dvar / keep.sum())
            A.append(np.clip((F2[:, keep] - mu[keep]) / sd[keep], -6, 6) * s_)
            B.append(np.clip((Ft2[:, keep] - mu[keep]) / sd[keep], -6, 6) * s_)
        return np.hstack(A), np.hstack(B)

    oof, tp = np.zeros(len(y)), np.zeros(len(te))
    for k in range(5):
        t, v = fold != k, fold == k
        A, B = design(t)
        r = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(A[t], y[t])
        oof[v] = r.predict(A[v])
        tp += r.predict(B) / 5
    w = np.exp(-0.5 * ((y - y.mean()) / PUB_STD) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
    w /= w.mean()
    d = ROOT / "outputs/emb_reg" / a.name
    d.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": tr.ID, "fold": fold, "hardness": y, "pred": oof}).to_csv(d / "oof.csv", index=False)
    pd.DataFrame({"ID": te.ID, "hardness": tp}).to_csv(d / "test_pred.csv", index=False)
    if a.full:  # 전체 500장 재적합 (폴드 평균 대신): 학습곡선상 400→500이 OOF로 확인 안 되는 이득
        A, B = design(np.ones(len(y), bool))
        r = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(A, y)
        tf = r.predict(B)
        pd.DataFrame({"ID": te.ID, "hardness": tf}).to_csv(d / "test_pred_full.csv", index=False)
        print(f"  전체적합 α={r.alpha_:.3g} | 폴드평균과 차이 RMS {np.sqrt(np.mean((tf - tp) ** 2)):.3f}, 상관 {np.corrcoef(tf, tp)[0, 1]:.4f}")
    print(f"[{a.name}] 층 {nl_used}/{nl} C={C} 배율 {a.mult} → OOF {np.sqrt(np.mean((oof - y) ** 2)):.3f} "
          f"| 재가중 {np.sqrt(np.mean(w * (oof - y) ** 2)):.3f}")


if __name__ == "__main__":
    main()
