"""합성 미세구조 경도 예측 — EDA 파이프라인.

실행: (프로젝트 루트에서)  .venv/Scripts/python eda/run_eda.py
산출물:
  outputs/features/{train,test}_features.csv   수작업 특징 캐시
  reports/figures/*.png                         EDA 그림
  reports/eda_summary.json                      보고서용 핵심 수치
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import stats
from skimage import segmentation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import debug_maps, extract_features  # noqa: E402

warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 100

DATA = ROOT / "data"
FEAT_DIR = ROOT / "outputs" / "features"
FIG = ROOT / "reports" / "figures"
FEAT_DIR.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 42
C_TRAIN, C_TEST, C_ACC = "#2563eb", "#f97316", "#dc2626"
SUMMARY: dict = {}


def load_img(split: str, id_: str) -> np.ndarray:
    return np.array(Image.open(DATA / split / f"{id_}.png"))


def _feat_one(split, id_):
    return {"ID": id_, **extract_features(load_img(split, id_))}


def get_features(split: str, ids: list[str]) -> pd.DataFrame:
    path = FEAT_DIR / f"{split}_features.csv"
    if path.exists():
        df = pd.read_csv(path)
        if len(df) == len(ids):
            return df
    t = time.time()
    # joblib(loky)은 임시폴더 경로에 비ASCII(한글 사용자명)가 있으면 실패 → 표준 ProcessPool 사용
    with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 2)) as ex:
        rows = list(ex.map(_feat_one, [split] * len(ids), ids, chunksize=8))
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"[features] {split}: {len(df)} imgs, {df.shape[1] - 1} feats, {time.time() - t:.1f}s")
    return df


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=110, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------- 1. 타깃
def analyze_target(tr):
    y = tr["hardness"].values
    q1, q3 = np.percentile(y, [25, 75])
    iqr = q3 - q1
    out = (y < q1 - 1.5 * iqr) | (y > q3 + 1.5 * iqr)
    SUMMARY["target"] = {
        "n": int(len(y)), "mean": float(y.mean()), "std": float(y.std(ddof=1)), "min": float(y.min()),
        "p05": float(np.percentile(y, 5)), "median": float(np.median(y)), "p95": float(np.percentile(y, 95)),
        "max": float(y.max()), "skew": float(stats.skew(y)), "kurtosis": float(stats.kurtosis(y)),
        "skew_log": float(stats.skew(np.log(y))), "shapiro_p": float(stats.shapiro(y).pvalue),
        "n_iqr_outliers": int(out.sum()), "iqr_outlier_ids": tr.loc[out, "ID"].tolist(),
        "rmse_mean_baseline": float(y.std(ddof=0)),
    }
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.3))
    ax[0].hist(y, bins=40, color=C_TRAIN, alpha=0.75, density=True, edgecolor="white")
    xs = np.linspace(y.min() - 5, y.max() + 5, 300)
    ax[0].plot(xs, stats.gaussian_kde(y)(xs), color="k", lw=1.5, label="KDE")
    ax[0].plot(xs, stats.norm.pdf(xs, y.mean(), y.std()), "--", color=C_ACC, lw=1.2, label="정규분포 적합")
    ax[0].axvline(y.mean(), color="gray", ls=":")
    ax[0].set(title=f"경도 분포 (n={len(y)}, mean={y.mean():.1f}, std={y.std():.1f})", xlabel="hardness (HV)")
    ax[0].legend()
    ax[1].boxplot(y, vert=False, widths=0.5, patch_artist=True, boxprops=dict(facecolor="#bfdbfe"))
    ax[1].scatter(y, np.random.default_rng(0).normal(1, 0.04, len(y)), s=6, alpha=0.4, color=C_TRAIN)
    ax[1].set(title=f"Box plot (IQR 이상치 {out.sum()}개)", xlabel="hardness (HV)", yticks=[])
    stats.probplot(y, dist="norm", plot=ax[2])
    ax[2].set_title(f"Q-Q plot (skew={stats.skew(y):.2f}, kurt={stats.kurtosis(y):.2f})")
    ax[2].get_lines()[0].set(markersize=3, color=C_TRAIN)
    savefig("fig01_target_distribution.png")


# --------------------------------------------------------------------------- 2. 샘플
def plot_samples(tr):
    s = tr.sort_values("hardness").reset_index(drop=True)
    n = len(s)
    groups = [("최저 경도", list(range(6))), ("중앙값 부근", list(range(n // 2 - 3, n // 2 + 3))),
              ("최고 경도", list(range(n - 6, n)))]
    fig, ax = plt.subplots(3, 6, figsize=(18, 9.6))
    for r, (name, idx) in enumerate(groups):
        for c, i in enumerate(idx):
            row = s.iloc[i]
            ax[r, c].imshow(load_img("train", row.ID), cmap="gray", vmin=0, vmax=255)
            ax[r, c].set_title(f"{row.ID[-6:]} | {row.hardness:.1f}", fontsize=10)
            ax[r, c].axis("off")
        ax[r, 0].text(-0.08, 0.5, name, transform=ax[r, 0].transAxes, rotation=90, va="center", ha="right",
                      fontsize=13, fontweight="bold")
    savefig("fig02_samples_by_hardness.png")


def plot_pipeline(example="TRAIN_000162"):
    img = load_img("train", example)
    m = debug_maps(img)
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.8))
    ax[0].imshow(m["raw"], cmap="gray", vmin=0, vmax=255)
    ax[0].set_title(f"① 원본 (노이즈 σ={m['sigma']:.1f})")
    ax[1].imshow(m["denoised"], cmap="gray", vmin=0, vmax=255)
    ax[1].set_title("② TV 디노이즈 (구간별 상수 복원)")
    im = ax[2].imshow(m["rel"], cmap="turbo", vmin=-0.1, vmax=0.6)
    ax[2].set_title(f"③ 상대 어둡기 (기지={m['matrix']:.0f})\n0: 기지, 0.2~0.4: 2상, >0.4: 기공")
    plt.colorbar(im, ax=ax[2], fraction=0.046)
    ov = np.dstack([m["denoised"] / 255] * 3)
    ov[m["pore_mask"]] = [1, 0, 0]
    ax[3].imshow(ov)
    ax[3].set_title("④ 기공 검출 (top-hat + 최암부 65%)")
    ax[4].imshow(segmentation.mark_boundaries(m["denoised"] / 255, m["watershed"], color=(1, 1, 0)))
    ax[4].set_title("⑤ Watershed 결정립 분할")
    for a in ax:
        a.axis("off")
    savefig("fig05_feature_pipeline.png")


# --------------------------------------------------------------------------- 3. 영상 품질/nuisance
def analyze_quality(tr, te):
    cols = [("q_noise_sigma", "노이즈 σ (추정)"), ("q_blur_raw", "블러 지수 (0=선명)"),
            ("q_contrast", "대비 (p99-p1)"), ("q_mean", "평균 밝기")]
    fig, ax = plt.subplots(2, 4, figsize=(20, 8.5))
    res = {}
    for j, (c, lab) in enumerate(cols):
        a = ax[0, j]
        bins = np.linspace(min(tr[c].min(), te[c].min()), max(tr[c].max(), te[c].max()), 35)
        a.hist(tr[c], bins=bins, alpha=0.6, density=True, color=C_TRAIN, label="train")
        a.hist(te[c], bins=bins, alpha=0.5, density=True, color=C_TEST, label="test")
        ks = stats.ks_2samp(tr[c], te[c])
        a.set(title=f"{lab}\nKS={ks.statistic:.3f}, p={ks.pvalue:.2f}")
        a.legend()
        b = ax[1, j]
        rho, p = stats.spearmanr(tr[c], tr["hardness"])
        b.scatter(tr[c], tr["hardness"], s=8, alpha=0.5, color=C_TRAIN)
        b.set(xlabel=lab, ylabel="hardness", title=f"vs 경도: Spearman ρ={rho:.3f} (p={p:.2g})")
        res[c] = {"ks": float(ks.statistic), "ks_p": float(ks.pvalue), "rho": float(rho), "rho_p": float(p),
                  "train_min": float(tr[c].min()), "train_max": float(tr[c].max()),
                  "test_min": float(te[c].min()), "test_max": float(te[c].max())}
    savefig("fig03_image_quality.png")
    SUMMARY["quality"] = res


def analyze_orientation(tr):
    """방향 각도(nuisance)가 경도와 무관한지 → 회전 증강 정당성 검증."""
    y = tr["hardness"].values
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.5))
    res = {}
    for a, (c, lab) in zip(ax[:2], [("nui_st_angle", "구조텐서 주방향"), ("nui_acf_angle", "자기상관 주방향")]):
        th = tr[c].values
        X = np.c_[np.ones_like(th), np.cos(2 * th), np.sin(2 * th)]
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        # 순열 검정
        rng = np.random.default_rng(SEED)
        null = []
        for _ in range(2000):
            yp = rng.permutation(y)
            bp, *_ = np.linalg.lstsq(X, yp, rcond=None)
            null.append(1 - ((yp - X @ bp) ** 2).sum() / ((yp - yp.mean()) ** 2).sum())
        p = float((np.array(null) >= r2).mean())
        coh = tr["an_coherence"].values
        sc = a.scatter(np.degrees(th), y, c=coh, cmap="viridis", s=10, alpha=0.7)
        plt.colorbar(sc, ax=a, label="coherence (정렬 강도)")
        a.set(xlabel=f"{lab} (deg)", ylabel="hardness", title=f"{lab} vs 경도\n원형-선형 R²={r2:.4f}, 순열 p={p:.3f}")
        res[c] = {"r2": float(r2), "perm_p": p}
    rho, p = stats.spearmanr(tr["an_coherence"], y)
    ax[2].scatter(tr["an_coherence"], y, s=10, alpha=0.6, color=C_TRAIN)
    ax[2].set(xlabel="an_coherence (정렬 정도, 회전불변)", ylabel="hardness",
              title=f"정렬 정도 vs 경도: ρ={rho:.3f} (p={p:.2g})")
    res["an_coherence"] = {"rho": float(rho), "p": float(p)}
    savefig("fig04_orientation_nuisance.png")
    SUMMARY["orientation"] = res


# --------------------------------------------------------------------------- 4. 특징-타깃 관계
GROUP_NAMES = {"q": "영상품질", "ph": "상분율", "pore": "기공", "gr": "결정립", "sp": "스펙트럼",
               "an": "이방성", "het": "불균일성", "tx": "텍스처"}
GROUP_COLORS = {"q": "#9ca3af", "ph": "#2563eb", "pore": "#dc2626", "gr": "#16a34a", "sp": "#9333ea",
                "an": "#f59e0b", "het": "#0891b2", "tx": "#78716c"}


def group_of(c):
    return c.split("_")[0]


def analyze_correlations(tr, feats):
    y = tr["hardness"]
    rows = []
    for c in feats:
        x = tr[c]
        if x.std() == 0:
            continue
        rs, ps = stats.spearmanr(x, y)
        rp, _ = stats.pearsonr(x, y)
        rows.append({"feature": c, "group": group_of(c), "spearman": rs, "spearman_p": ps, "pearson": rp})
    cor = pd.DataFrame(rows)
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(tr[cor.feature].values, y.values, random_state=SEED, n_neighbors=5)
    cor["mutual_info"] = mi
    cor["abs_s"] = cor.spearman.abs()
    cor = cor.sort_values("abs_s", ascending=False).reset_index(drop=True)
    cor.to_csv(ROOT / "outputs" / "feature_target_correlation.csv", index=False)

    top = cor.head(25).iloc[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(17, 8.5))
    ax[0].barh(top.feature, top.spearman, color=[GROUP_COLORS[g] for g in top.group])
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set(title="경도와 Spearman 상관 상위 25개 특징", xlabel="Spearman ρ")
    handles = [plt.Rectangle((0, 0), 1, 1, color=GROUP_COLORS[g]) for g in GROUP_NAMES]
    ax[0].legend(handles, [f"{g}_ : {n}" for g, n in GROUP_NAMES.items()], loc="lower right", fontsize=8)
    gbest = cor.groupby("group").agg(max_abs_s=("abs_s", "max"), max_mi=("mutual_info", "max")).reindex(
        list(GROUP_NAMES)).dropna()
    xs = np.arange(len(gbest))
    ax[1].bar(xs - 0.2, gbest.max_abs_s, 0.4, label="max |Spearman|", color="#60a5fa")
    ax[1].bar(xs + 0.2, gbest.max_mi, 0.4, label="max Mutual Info", color="#fb923c")
    ax[1].set_xticks(xs, [f"{GROUP_NAMES[g]}\n({g}_)" for g in gbest.index])
    ax[1].set(title="특징 그룹별 최대 연관도")
    ax[1].legend()
    savefig("fig06_feature_correlation.png")

    # 상위 특징 산점도 + 구간 평균
    top9 = cor.head(9).feature.tolist()
    fig, ax = plt.subplots(3, 3, figsize=(16, 13))
    for a, c in zip(ax.ravel(), top9):
        x = tr[c].values
        a.scatter(x, y, s=8, alpha=0.45, color=GROUP_COLORS[group_of(c)])
        qb = pd.qcut(x, 10, duplicates="drop")
        gm = pd.DataFrame({"x": x, "y": y, "b": qb}).groupby("b", observed=True).mean()
        a.plot(gm.x, gm.y, "-o", color="k", ms=4, lw=1.5, label="10분위 평균")
        r = cor.set_index("feature").loc[c]
        a.set(title=f"{c}\nρ={r.spearman:.3f}, MI={r.mutual_info:.3f}", ylabel="hardness")
        a.legend(fontsize=8)
    savefig("fig07_top_feature_scatter.png")

    # 상위 20 특징 간 상관 (다중공선성)
    t20 = cor.head(20).feature.tolist()
    cm = tr[t20].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(t20)), t20, rotation=90, fontsize=8)
    ax.set_yticks(range(len(t20)), t20, fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("상위 20개 특징 간 Spearman 상관 (다중공선성 확인)")
    savefig("fig08_feature_collinearity.png")

    SUMMARY["correlation_top15"] = cor.head(15)[["feature", "spearman", "pearson", "mutual_info"]].round(4).to_dict("records")
    SUMMARY["group_best"] = gbest.round(4).reset_index().to_dict("records")
    SUMMARY["n_sig_features_p001"] = int((cor.spearman_p < 0.001).sum())
    SUMMARY["n_features"] = int(len(cor))
    return cor


def _partial_spearman(x, y, Z):
    rk = stats.rankdata
    Zr = np.c_[np.ones(len(x)), np.column_stack([rk(z) for z in Z.T])]
    ex = rk(x) - Zr @ np.linalg.lstsq(Zr, rk(x), rcond=None)[0]
    ey = rk(y) - Zr @ np.linalg.lstsq(Zr, rk(y), rcond=None)[0]
    return stats.pearsonr(ex, ey)


def analyze_multivariate(tr):
    """주요 동인(2상 분율·정렬도)을 통제한 편상관 → 결정립 크기·기공·분포 효과를 분리."""
    y = tr["hardness"].values
    ctrl_cols = ["ph_dark_frac_0.08", "an_local_coherence"]
    ctrl = tr[ctrl_cols].values
    clean = (tr.q_noise_sigma < tr.q_noise_sigma.median()).values
    feats = [("gr_area_mean", "결정립 평균 면적(watershed)"), ("sp_raw_centroid", "스펙트럼 중심주파수(↑=미세립)"),
             ("gr_lb_density", "결정립계 선 밀도"), ("gr_area_cv", "결정립 크기 CV(불균일성)"),
             ("gr_area_p90_p10", "결정립 크기 p90/p10"), ("pore_area_frac", "기공 면적분율"),
             ("pore_count", "기공 개수"), ("q_noise_sigma", "노이즈 σ"), ("q_blur_raw", "블러 지수")]
    rows = []
    for c, lab in feats:
        r, _ = stats.spearmanr(tr[c], y)
        pr, pp = _partial_spearman(tr[c].values, y, ctrl)
        prc, ppc = _partial_spearman(tr.loc[clean, c].values, y[clean], ctrl[clean])
        rows.append(dict(feature=c, label=lab, rho=r, partial=pr, partial_p=pp, partial_clean=prc, partial_clean_p=ppc))
    pc = pd.DataFrame(rows)
    pc.to_csv(ROOT / "outputs" / "partial_correlation.csv", index=False)

    fig, ax = plt.subplots(1, 2, figsize=(19, 6))
    ys = np.arange(len(pc))[::-1]
    ax[0].barh(ys + 0.27, pc.rho, 0.27, color="#cbd5e1", label="단순 Spearman ρ")
    ax[0].barh(ys, pc.partial, 0.27, color="#2563eb", label="편상관 (2상분율·정렬도 통제)")
    ax[0].barh(ys - 0.27, pc.partial_clean, 0.27, color="#16a34a", label="편상관 — 저노이즈 절반(n=250)")
    for yy, p in zip(ys, pc.partial_p):
        ax[0].text(0.36, yy, "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s.",
                   va="center", fontsize=9)
    ax[0].set_yticks(ys, [f"{lab}\n({c})" for c, lab in zip(pc.feature, pc.label)], fontsize=9)
    ax[0].axvline(0, color="k", lw=0.8)
    ax[0].set(xlim=(-0.45, 0.42), title="결정립 크기·기공·분포 효과: 주요 동인 통제 후 편상관")
    ax[0].legend(loc="lower left", fontsize=9)
    sc = ax[1].scatter(tr["ph_dark_frac_0.08"], tr["an_local_coherence"], c=y, cmap="viridis", s=14)
    plt.colorbar(sc, ax=ax[1], label="hardness")
    ax[1].set(xlabel="ph_dark_frac_0.08 (2상 면적분율 프록시)", ylabel="an_local_coherence (정렬도)",
              title="두 주요 동인 평면에서의 경도 — 우상단으로 갈수록 경도↑ (가산 구조)")
    savefig("fig13_partial_correlation.png")
    SUMMARY["partial_corr"] = pc.drop(columns="label").round(4).to_dict("records")


# --------------------------------------------------------------------------- 5. train/test 분포 이동
def analyze_shift(tr, te, feats):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    ks = pd.DataFrame([{"feature": c, "ks": stats.ks_2samp(tr[c], te[c]).statistic,
                        "p": stats.ks_2samp(tr[c], te[c]).pvalue} for c in feats]).sort_values("ks", ascending=False)
    # Test must never be used to fit a model, including a domain classifier.
    # Descriptive KS statistics are not used for model/feature selection.
    scaler = StandardScaler().fit(tr[feats])
    Ztr, Zte = scaler.transform(tr[feats]), scaler.transform(te[feats])
    pca = PCA(2, random_state=SEED).fit(Ztr)
    P = np.concatenate([pca.transform(Ztr), pca.transform(Zte)])
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.6))
    t15 = ks.head(15).iloc[::-1]
    ax[0].barh(t15.feature, t15.ks, color=[C_ACC if p < 0.01 else "#93c5fd" for p in t15.p])
    ax[0].set(title=f"Train vs Test KS 통계량 상위 15\n(빨강: p<0.01, 총 {int((ks.p < 0.01).sum())}/{len(ks)}개)",
              xlabel="KS statistic")
    ax[1].scatter(P[len(tr):, 0], P[len(tr):, 1], s=7, alpha=0.4, color=C_TEST, label="test (1000)")
    ax[1].scatter(P[: len(tr), 0], P[: len(tr), 1], s=7, alpha=0.6, color=C_TRAIN, label="train (500)")
    ax[1].set(title="PCA (scaler/PCA 모두 train에만 적합)", xlabel="PC1", ylabel="PC2")
    ax[1].legend()
    sc = ax[2].scatter(P[: len(tr), 0], P[: len(tr), 1], c=tr.hardness, cmap="viridis", s=10)
    plt.colorbar(sc, ax=ax[2], label="hardness")
    ax[2].set(title=f"PCA 공간의 경도 (PC1 설명분산 {pca.explained_variance_ratio_[0]:.0%}, "
                    f"PC2 {pca.explained_variance_ratio_[1]:.0%})", xlabel="PC1", ylabel="PC2")
    savefig("fig09_train_test_shift.png")
    SUMMARY["shift"] = {"adversarial_auc": None, "test_model_fitting": False,
                        "n_ks_p01": int((ks.p < 0.01).sum()),
                        "ks_top5": ks.head(5).round(4).to_dict("records")}


# --------------------------------------------------------------------------- 6. 중복/근접 중복
def analyze_duplicates(tr, te):
    def thumbs(split, ids):
        out = []
        for i in ids:
            im = Image.fromarray(load_img(split, i)).resize((32, 32), Image.BILINEAR)
            a = np.asarray(im, dtype=np.float32)
            out.append(a)
        return np.stack(out)

    def norm(a):
        a = a.reshape(len(a), -1)
        a = a - a.mean(1, keepdims=True)
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-6)

    A = thumbs("train", tr.ID)
    B = thumbs("test", te.ID)
    trans = [lambda x: x, lambda x: np.rot90(x, 1, (1, 2)), lambda x: np.rot90(x, 2, (1, 2)),
             lambda x: np.rot90(x, 3, (1, 2)), lambda x: x[:, :, ::-1], lambda x: x[:, ::-1, :],
             lambda x: np.transpose(x, (0, 2, 1)), lambda x: np.rot90(x, 1, (1, 2))[:, :, ::-1]]
    An = norm(A)
    best_tt = np.full(len(B), -1.0)
    best_rr = np.full(len(A), -1.0)
    for t in trans:
        Bt = norm(np.ascontiguousarray(t(B)))
        best_tt = np.maximum(best_tt, (Bt @ An.T).max(1))
        At = norm(np.ascontiguousarray(t(A)))
        S = At @ An.T
        if t is trans[0]:
            np.fill_diagonal(S, -1)
        best_rr = np.maximum(best_rr, S.max(1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(best_rr, bins=40, alpha=0.6, color=C_TRAIN, label="train→train (자기 제외)", density=True)
    ax.hist(best_tt, bins=40, alpha=0.5, color=C_TEST, label="test→train", density=True)
    ax.set(title="32×32 썸네일 최대 상관 (8가지 회전/반전 포함)", xlabel="최대 정규화 상관")
    ax.legend()
    savefig("fig10_near_duplicates.png")
    SUMMARY["duplicates"] = {"max_test_train": float(best_tt.max()), "max_train_train": float(best_rr.max()),
                             "n_test_gt_0.9": int((best_tt > 0.9).sum()), "n_train_gt_0.9": int((best_rr > 0.9).sum()),
                             "median_test_train": float(np.median(best_tt))}


# --------------------------------------------------------------------------- 7. 베이스라인 CV
def analyze_baselines(tr, feats, cor):
    from lightgbm import LGBMRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    y = tr["hardness"].values
    X = tr[feats].values

    def lgbm():
        return LGBMRegressor(n_estimators=700, learning_rate=0.02, num_leaves=15, min_child_samples=12,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=1.0,
                             verbose=-1, random_state=SEED)

    models = {
        "Mean": None,
        "Ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))),
        "RandomForest": lambda: RandomForestRegressor(500, min_samples_leaf=2, max_features=0.4, n_jobs=-1,
                                                      random_state=SEED),
        "LightGBM": lgbm,
    }

    def cv(make, Xm, repeats=3):
        scores, oof_all = [], np.zeros((repeats, len(y)))
        for r in range(repeats):
            for a, b in KFold(5, shuffle=True, random_state=SEED + r).split(Xm):
                if make is None:
                    p = np.full(len(b), y[a].mean())
                else:
                    m = make().fit(Xm[a], y[a])
                    p = m.predict(Xm[b])
                oof_all[r, b] = p
            scores.append(np.sqrt(np.mean((oof_all[r] - y) ** 2)))
        return np.array(scores), oof_all.mean(0)

    res, oofs = {}, {}
    for name, mk in models.items():
        s, o = cv(mk, X)
        res[name] = (float(s.mean()), float(s.std()))
        oofs[name] = o
        print(f"[baseline] {name}: RMSE {s.mean():.3f} ± {s.std():.3f}")

    # 그룹별 단독 성능 (LightGBM)
    grp = {}
    for g in GROUP_NAMES:
        cols = [i for i, c in enumerate(feats) if group_of(c) == g]
        if cols:
            s, _ = cv(lgbm, X[:, cols], repeats=1)
            grp[g] = float(s.mean())
    # 영상품질 제외 효과
    noq = [i for i, c in enumerate(feats) if group_of(c) != "q"]
    s_noq, _ = cv(lgbm, X[:, noq], repeats=3)
    res["LightGBM (q_ 제외)"] = (float(s_noq.mean()), float(s_noq.std()))

    best = min((k for k in models if k != "Mean"), key=lambda k: res[k][0])
    oof = oofs[best]
    m = lgbm().fit(X, y)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=feats).sort_values(ascending=False)

    fig, ax = plt.subplots(2, 2, figsize=(17, 12))
    names = list(res)
    ax[0, 0].bar(range(len(names)), [res[n][0] for n in names], yerr=[res[n][1] for n in names],
                 color=["#9ca3af", "#60a5fa", "#34d399", "#2563eb", "#1e40af"][: len(names)], capsize=4)
    for i, n in enumerate(names):
        ax[0, 0].text(i, res[n][0] + 0.3, f"{res[n][0]:.2f}", ha="center")
    ax[0, 0].set_xticks(range(len(names)), names, rotation=15)
    ax[0, 0].set(title="수작업 특징 베이스라인 5-Fold×3 CV RMSE (낮을수록 좋음)", ylabel="RMSE (HV)")
    gs = pd.Series(grp).sort_values()
    ax[0, 1].barh([f"{GROUP_NAMES[g]} ({g}_)" for g in gs.index], gs.values, color=[GROUP_COLORS[g] for g in gs.index])
    ax[0, 1].axvline(res["Mean"][0], color="k", ls="--", label=f"평균예측 {res['Mean'][0]:.2f}")
    ax[0, 1].set(title="특징 그룹 단독 LightGBM CV RMSE", xlabel="RMSE")
    ax[0, 1].legend()
    lim = [y.min() - 5, y.max() + 5]
    ax[1, 0].scatter(y, oof, s=10, alpha=0.6, color=C_TRAIN)
    ax[1, 0].plot(lim, lim, "k--", lw=1)
    r2 = 1 - ((y - oof) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    ax[1, 0].set(xlim=lim, ylim=lim, xlabel="실제 경도", ylabel="OOF 예측", title=f"{best} OOF 예측 vs 실제 (R²={r2:.3f})")
    t20 = imp.head(20).iloc[::-1]
    ax[1, 1].barh(t20.index, t20.values, color=[GROUP_COLORS[group_of(c)] for c in t20.index])
    ax[1, 1].set(title="LightGBM 특징 중요도 (gain) 상위 20", xlabel="gain")
    savefig("fig11_baseline_models.png")

    # 오차 분석: 노이즈/극단값과 잔차
    resid = oof - y
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
    for a, (c, lab) in zip(ax[:2], [("q_noise_sigma", "노이즈 σ"), ("q_blur_raw", "블러 지수")]):
        qb = pd.qcut(tr[c], 5)
        g = pd.DataFrame({"b": qb, "ae": np.abs(resid)}).groupby("b", observed=True).ae.agg(["mean", "count"])
        rmse_b = pd.DataFrame({"b": qb, "se": resid ** 2}).groupby("b", observed=True).se.mean() ** 0.5
        a.bar(range(len(rmse_b)), rmse_b.values, color="#93c5fd")
        a.set_xticks(range(len(rmse_b)), [f"{iv.left:.2f}~\n{iv.right:.2f}" for iv in rmse_b.index], fontsize=8)
        a.set(title=f"{lab} 5분위별 OOF RMSE", ylabel="RMSE")
    qy = pd.qcut(y, 5)
    bias = pd.DataFrame({"b": qy, "r": resid}).groupby("b", observed=True).r.mean()
    ax[2].bar(range(len(bias)), bias.values, color=["#f87171" if v < 0 else "#60a5fa" for v in bias.values])
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(range(len(bias)), [f"{iv.left:.0f}~{iv.right:.0f}" for iv in bias.index], fontsize=8)
    ax[2].set(title="실제 경도 5분위별 평균 잔차 (예측-실제)\n→ 양 극단 회귀-평균 편향", ylabel="평균 잔차")
    savefig("fig12_error_analysis.png")

    SUMMARY["baselines"] = {k: {"rmse": v[0], "std": v[1]} for k, v in res.items()}
    SUMMARY["group_only_rmse"] = grp
    SUMMARY["best_baseline"] = best
    SUMMARY["best_oof_r2"] = float(r2)
    SUMMARY["importance_top10"] = imp.head(10).round(1).to_dict()
    SUMMARY["extreme_bias"] = {str(k): float(v) for k, v in bias.items()}


def main():
    tr = pd.read_csv(DATA / "train.csv")
    te = pd.read_csv(DATA / "sample_submission.csv")[["ID"]]
    ftr = get_features("train", tr.ID.tolist())
    fte = get_features("test", te.ID.tolist())
    tr = tr.merge(ftr, on="ID")
    te = te.merge(fte, on="ID")
    feats = [c for c in ftr.columns if c != "ID" and not c.startswith("nui_")]
    SUMMARY["data"] = {"n_train": len(tr), "n_test": len(te), "img_shape": list(load_img("train", tr.ID[0]).shape),
                       "dtype": str(load_img("train", tr.ID[0]).dtype)}

    analyze_target(tr)
    plot_samples(tr)
    plot_pipeline()
    analyze_quality(tr, te)
    analyze_orientation(tr)
    cor = analyze_correlations(tr, feats)
    analyze_multivariate(tr)
    analyze_shift(tr, te, feats)
    analyze_duplicates(tr, te)
    analyze_baselines(tr, feats, cor)

    with open(ROOT / "reports" / "eda_summary.json", "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, ensure_ascii=False, indent=2, default=float)
    print("done ->", FIG)


if __name__ == "__main__":
    main()
