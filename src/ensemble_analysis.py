"""앙상블 심층 분석 — 모든 결합 방식은 '중첩 CV'(가중치를 다른 폴드 OOF로 추정)로 평가해 낙관 편향을 제거한다.

분석 항목
 1) 다양성 분해 (Krogh & Vedelsby): 앙상블 MSE = 개별 평균 MSE − 평균 모호도(ambiguity)
 2) 앙상블 크기별 성능: 무작위 k개 부분집합 단순평균의 RMSE 분포
 3) 결합 방식 비교: 단순평균 / 계열 균등평균 / NNLS 가중 / Ridge 스태킹 / 탐욕 전진선택(Caruana)
 4) 시드 앙상블 효과 (scr_<model>_s<seed> 가 있으면)
실행: python src/ensemble_analysis.py   → reports/figures/fig_ens_*.png, outputs/ensemble_analysis.json
"""
from __future__ import annotations

import json
import re
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from sklearn.linear_model import RidgeCV

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
FIG = ROOT / "reports" / "figures"
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

FAMILY = {"feat": "특징기반", "ft": "미세조정CNN", "probe": "고정임베딩"}


def load(include_seeds=False):
    base = pd.read_csv(OUT / "feat_ridge" / "oof.csv") if (OUT / "feat_ridge").exists() else None
    y, fold = base.hardness.values, base.fold.values
    P, T = {}, {}
    for d in sorted(OUT.glob("feat_*")):
        if (d / "oof.csv").exists():
            P[d.name] = pd.read_csv(d / "oof.csv").pred.values
            T[d.name] = pd.read_csv(d / "test_pred.csv").hardness.values
    for d in sorted((OUT / "cnn").glob("scr_*")):
        if not (d / "oof.csv").exists():
            continue
        is_seed = re.search(r"_s\d+$", d.name) is not None
        if is_seed and not include_seeds:
            continue
        P[d.name.replace("scr_", "ft_")] = pd.read_csv(d / "oof.csv").pred.values
        T[d.name.replace("scr_", "ft_")] = pd.read_csv(d / "test_pred.csv").hardness.values
    for d in sorted((OUT / "probe").glob("*")):
        if (d / "oof_emb.csv").exists():
            k = "probe_" + d.name.split(".")[0]
            P[k] = pd.read_csv(d / "oof_emb.csv").pred.values
            T[k] = pd.read_csv(d / "test_emb.csv").hardness.values
    return pd.DataFrame(P), pd.DataFrame(T), y, fold


def rmse(p, y):
    return float(np.sqrt(np.mean((p - y) ** 2)))


def fam(c):
    return c.split("_")[0]


# ------------------------------------------------------------------ 결합기 (fit on train-fold OOF → apply)
def comb_mean(Xa, ya, Xb):
    return Xb.mean(1)


def comb_nnls(Xa, ya, Xb):
    w, _ = nnls(Xa, ya)
    w = w / w.sum() if w.sum() > 0 else np.full(Xa.shape[1], 1 / Xa.shape[1])
    return Xb @ w


def comb_ridge_stack(Xa, ya, Xb):
    m = RidgeCV(alphas=np.logspace(-1, 5, 40)).fit(Xa, ya)
    return m.predict(Xb)


def comb_greedy(Xa, ya, Xb, max_iter=25):
    """Caruana 전진 선택 (복원 추출) — 선택 빈도가 곧 가중치."""
    cnt = np.zeros(Xa.shape[1])
    cur = np.zeros(len(ya))
    for it in range(max_iter):
        best, bj = None, None
        for j in range(Xa.shape[1]):
            p = (cur * it + Xa[:, j]) / (it + 1)
            s = rmse(p, ya)
            if best is None or s < best:
                best, bj = s, j
        cnt[bj] += 1
        cur = (cur * it + Xa[:, bj]) / (it + 1)
    w = cnt / cnt.sum()
    return Xb @ w


def nested(comb, X, y, fold):
    out = np.zeros(len(y))
    for k in np.unique(fold):
        a, b = fold != k, fold == k
        out[b] = comb(X[a], y[a], X[b])
    return out


def family_mean(P: pd.DataFrame):
    fams = sorted({fam(c) for c in P.columns})
    return np.mean([P[[c for c in P.columns if fam(c) == f]].mean(1).values for f in fams], axis=0)


def main():
    res = {}
    P, T, y, fold = load()
    cols = list(P.columns)
    X = P.values

    # 1) 다양성 분해
    ens = X.mean(1)
    avg_mse = float(np.mean([(X[:, j] - y) ** 2 for j in range(X.shape[1])]))
    ambiguity = float(np.mean([(X[:, j] - ens) ** 2 for j in range(X.shape[1])]))
    res["decomposition_all"] = {"n_models": len(cols), "avg_individual_rmse": avg_mse ** 0.5,
                                "avg_individual_mse": avg_mse, "ambiguity": ambiguity,
                                "ensemble_mse": float(np.mean((ens - y) ** 2)), "ensemble_rmse": rmse(ens, y)}

    # 2) 크기별 곡선
    rng = np.random.default_rng(0)
    sizes = list(range(1, len(cols) + 1))
    curve = {}
    for k in sizes:
        combs = list(combinations(range(len(cols)), k))
        if len(combs) > 300:
            combs = [tuple(rng.choice(len(cols), k, replace=False)) for _ in range(300)]
        vals = [rmse(X[:, list(c)].mean(1), y) for c in combs]
        curve[k] = (float(np.mean(vals)), float(np.min(vals)), float(np.max(vals)))
    res["size_curve"] = curve

    # 3) 결합 방식
    methods = {"단순평균(전체)": lambda: nested(comb_mean, X, y, fold),
               "계열 균등평균": lambda: family_mean(P),
               "NNLS 가중": lambda: nested(comb_nnls, X, y, fold),
               "Ridge 스태킹": lambda: nested(comb_ridge_stack, X, y, fold),
               "탐욕 전진선택": lambda: nested(comb_greedy, X, y, fold)}
    comb_res = {}
    for name, f in methods.items():
        comb_res[name] = rmse(f(), y)
    best_single = min(cols, key=lambda c: rmse(P[c].values, y))
    comb_res[f"(참고) 단일 최고: {best_single}"] = rmse(P[best_single].values, y)
    res["combiners"] = comb_res
    res["singles"] = {c: rmse(P[c].values, y) for c in cols}

    # 4) 시드 앙상블
    Ps, _, _, _ = load(include_seeds=True)
    seed_groups = {}
    for c in Ps.columns:
        m = re.match(r"(ft_.+?)(_s\d+)?$", c)
        if m and c.startswith("ft_"):
            seed_groups.setdefault(m.group(1), []).append(c)
    seed_res = {}
    for base, members in seed_groups.items():
        if len(members) >= 2:
            singles = [rmse(Ps[c].values, y) for c in members]
            seed_res[base] = {"n_seeds": len(members), "single_mean": float(np.mean(singles)),
                              "single_min": float(np.min(singles)), "single_max": float(np.max(singles)),
                              "seed_avg": rmse(Ps[members].mean(1).values, y),
                              "seed_corr_mean": float(np.mean([np.corrcoef(Ps[a], Ps[b])[0, 1]
                                                               for a, b in combinations(members, 2)]))}
    res["seed_ensembles"] = seed_res

    # 5) 큐레이션 조합 (계열별 대표만, 시드는 평균해 1개 모델로 취급)
    def seedavg(base):
        mem = seed_groups.get(base, [base])
        return Ps[mem].mean(1).values

    eff_cands = [c for c in ("ft_efficientnet_b0", "ft_efficientnet_b0_lr5e4") if c in Ps.columns]
    eff = min(eff_cands, key=lambda c: rmse(Ps[c].values, y))  # 변형 중 OOF가 좋은 쪽을 대표로
    blocks = {
        "ridge": P["feat_ridge"].values,
        "kernel": P[["feat_kernel_ridge", "feat_gp", "feat_svr"]].mean(1).values,
        "resnet18": seedavg("ft_resnet18"),
        "convnext": seedavg("ft_convnext_nano"),
        "effnet": Ps[eff].values,
        "resnet50ft": seedavg("ft_resnet50"),
        "probe": P[["probe_resnet50", "probe_vit_small_patch14_dinov2"]].mean(1).values,
    }
    configs = {
        "A. Ridge + CNN 4종 (제출 v1 구성)": ["ridge", "resnet18", "convnext", "effnet", "resnet50ft"],
        "B. Ridge + CNN 3종 (ResNet-50 제외)": ["ridge", "resnet18", "convnext", "effnet"],
        "C. 특징2 + CNN3 + 프로브": ["ridge", "kernel", "resnet18", "convnext", "effnet", "probe"],
        "D. 계열 균등: 특징 / CNN / 프로브": None,
    }
    cur = {}
    for name, keys in configs.items():
        if keys is None:
            f_ = np.mean([blocks["ridge"], blocks["kernel"]], 0)
            c_ = np.mean([blocks["resnet18"], blocks["convnext"], blocks["effnet"]], 0)
            p = (f_ + c_ + blocks["probe"]) / 3
        else:
            p = np.mean([blocks[k] for k in keys], 0)
        cur[name] = rmse(p, y)
    res["curated"] = cur
    res["efficientnet_used"] = eff

    # ---------------- 그림
    fig, ax = plt.subplots(1, 3, figsize=(21, 5.8))
    s = pd.Series(res["singles"]).sort_values()
    colors = {"feat": "#2563eb", "ft": "#dc2626", "probe": "#16a34a"}
    ax[0].barh(s.index[::-1], s.values[::-1], color=[colors[fam(c)] for c in s.index[::-1]])
    ax[0].axvline(comb_res["단순평균(전체)"], color="k", ls="--", label=f"전체 단순평균 {comb_res['단순평균(전체)']:.2f}")
    ax[0].set(xlim=(12.5, max(s.values) + 0.3), title="개별 모델 OOF RMSE (파랑=특징, 빨강=미세조정, 초록=고정임베딩)",
              xlabel="RMSE")
    ax[0].legend(loc="lower right")
    ks = np.array(sizes)
    mean_c = np.array([curve[k][0] for k in sizes])
    ax[1].plot(ks, mean_c, "-o", color="#2563eb", label="무작위 k개 평균의 평균")
    ax[1].fill_between(ks, [curve[k][1] for k in sizes], [curve[k][2] for k in sizes], alpha=0.2, label="최소~최대")
    ax[1].set(title="앙상블 크기별 RMSE (단순평균)", xlabel="모델 수 k", ylabel="OOF RMSE")
    ax[1].legend()
    cr = pd.Series(comb_res)
    ax[2].barh(cr.index[::-1], cr.values[::-1], color="#93c5fd")
    for i, v in enumerate(cr.values[::-1]):
        ax[2].text(v + 0.01, i, f"{v:.3f}", va="center")
    ax[2].set(xlim=(cr.min() - 0.3, cr.max() + 0.25), title="결합 방식 비교 (중첩 CV)", xlabel="OOF RMSE")
    plt.tight_layout()
    plt.savefig(FIG / "fig_ens_overview.png", dpi=110, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 9.5))
    cm = P.corr()
    im = ax.imshow(cm, cmap="RdBu_r", vmin=0.6, vmax=1.0)
    ax.set_xticks(range(len(cols)), cols, rotation=90, fontsize=8)
    ax.set_yticks(range(len(cols)), cols, fontsize=8)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{cm.values[i, j]:.2f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("모델 간 OOF 예측 상관 (낮을수록 앙상블 다양성↑)")
    plt.tight_layout()
    plt.savefig(FIG / "fig_ens_correlation.png", dpi=110, bbox_inches="tight")
    plt.close()

    with open(OUT / "ensemble_analysis.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in res.items() if k != "size_curve"}, ensure_ascii=False, indent=1))
    print("size curve (mean):", {k: round(v[0], 3) for k, v in curve.items()})


if __name__ == "__main__":
    main()
