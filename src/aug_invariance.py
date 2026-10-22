"""증강 변환의 '레이블 보존성' 검사 — 변환 전후 수작업 특징과 Ridge 예측 경도의 변화를 측정한다.

- 허용 후보: 반전, 90° 회전, 임의 각도 회전, 노이즈, 블러, 대비
- 금지 후보: 전단(shear), 확대(zoom), 비등방 늘이기(stretch) → 결정립 크기·형상·정렬도를 바꿈
- 판단 근거: EDA에서 방향은 경도와 무관, 정렬도·2상 분율은 경도와 관련 (reports/EDA_report.md §6~7)

실행: python src/aug_invariance.py  → reports/figures/fig_aug_examples.png, fig_aug_invariance.png,
                                      outputs/aug_invariance.csv
"""
from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from features import extract_features  # noqa: E402

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
N_IMG = 60
KEY = ["ph_dark_frac_0.08", "an_local_coherence", "an_coherence", "gr_area_cv", "sp_raw_centroid",
       "pore_area_frac", "gr_lb_intercept_len", "q_noise_sigma"]


def _affine(a, mat, zoom_center=True):
    """출력 좌표→입력 좌표 행렬로 중심 기준 아핀 변환 (reflect 패딩, 1차 보간)."""
    c = np.array(a.shape) / 2.0
    off = c - mat @ c
    return ndi.affine_transform(a, mat, offset=off, order=1, mode="reflect")


def transforms():
    rng = np.random.default_rng(0)
    return {
        "원본": lambda a: a,
        "좌우반전": lambda a: a[:, ::-1],
        "90° 회전": lambda a: np.rot90(a),
        "임의 회전 37°": lambda a: _affine(a, np.array([[np.cos(0.646), -np.sin(0.646)], [np.sin(0.646), np.cos(0.646)]])),
        "노이즈 σ+8": lambda a: a + rng.normal(0, 8, a.shape),
        "블러 σ0.8": lambda a: ndi.gaussian_filter(a, 0.8),
        "대비 ×1.1": lambda a: (a - a.mean()) * 1.1 + a.mean(),
        "[금지] 전단 0.25": lambda a: _affine(a, np.array([[1.0, 0.25], [0.0, 1.0]])),
        "[금지] 확대 1.3배": lambda a: _affine(a, np.eye(2) / 1.3),
        "[금지] 가로 늘이기 1.3": lambda a: _affine(a, np.array([[1.0, 0.0], [0.0, 1 / 1.3]])),
    }


def _job(args):
    id_, tname = args
    a = np.array(Image.open(ROOT / "data" / "train" / f"{id_}.png")).astype(np.float64)
    b = transforms()[tname](a)
    b = np.clip(np.round(b), 0, 255).astype(np.uint8)
    return {"ID": id_, "aug": tname, **extract_features(b)}


def main():
    tr = pd.read_csv(ROOT / "data" / "train.csv")
    ftr = pd.read_csv(ROOT / "outputs" / "features" / "train_features.csv")
    feats = [c for c in ftr.columns if c != "ID" and not c.startswith("nui_")]
    ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
    ridge.fit(ftr.set_index("ID").loc[tr.ID, feats].values, tr.hardness.values)

    ids = tr.sample(N_IMG, random_state=0).ID.tolist()
    names = list(transforms())
    jobs = [(i, t) for i in ids for t in names]
    cache = ROOT / "outputs" / "aug_invariance_features.csv"
    if cache.exists():
        df = pd.read_csv(cache)
    else:
        with ProcessPoolExecutor(max_workers=max(1, (os.cpu_count() or 2) - 4)) as ex:
            rows = list(ex.map(_job, jobs, chunksize=4))
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
    df["pred"] = ridge.predict(df[feats].values)
    base = df[df["aug"] == "원본"].set_index("ID")
    sd = ftr[KEY].std()
    out = []
    for t in names[1:]:
        d = df[df["aug"] == t].set_index("ID").loc[ids]
        r = {"aug": t, "pred_abs_change_HV": float(np.mean(np.abs(d.pred - base.loc[ids, "pred"])))}
        for k in KEY:
            r[k] = float(np.median(np.abs(d[k] - base.loc[ids, k])) / sd[k])
        out.append(r)
    res = pd.DataFrame(out)
    res.to_csv(ROOT / "outputs" / "aug_invariance.csv", index=False)
    print(res.round(3).to_string())

    # 그림 1: 변환 예시
    ex_id = "TRAIN_000162"
    a = np.array(Image.open(ROOT / "data" / "train" / f"{ex_id}.png")).astype(np.float64)
    fig, ax = plt.subplots(2, 5, figsize=(20, 8.6))
    for axx, t in zip(ax.ravel(), names):
        axx.imshow(np.clip(transforms()[t](a), 0, 255), cmap="gray", vmin=0, vmax=255)
        axx.set_title(t, fontsize=13, color="#b91c1c" if t.startswith("[금지]") else "black")
        axx.axis("off")
    plt.suptitle("증강 변환 예시 (빨강 [금지] = 결정립 크기·형상·정렬도를 바꾸므로 금지)", fontsize=14)
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "figures" / "fig_aug_examples.png", dpi=90, bbox_inches="tight")
    plt.close()

    # 그림 2: 레이블 보존성
    fig, ax = plt.subplots(1, 2, figsize=(19, 5.5), gridspec_kw={"width_ratios": [1, 1.6]})
    col = ["#b91c1c" if t.startswith("[금지]") else "#2563eb" for t in res["aug"]]
    ax[0].barh(res["aug"][::-1], res.pred_abs_change_HV[::-1], color=col[::-1])
    for i, v in enumerate(res.pred_abs_change_HV[::-1]):
        ax[0].text(v + 0.1, i, f"{v:.2f}", va="center")
    ax[0].set(title="변환 후 Ridge 예측 경도의 평균 변화 |Δ| (HV)\n(작을수록 레이블 보존적)", xlabel="HV")
    M = res.set_index("aug")[KEY]
    im = ax[1].imshow(M.values, cmap="Reds", vmin=0, vmax=1.0, aspect="auto")
    ax[1].set_xticks(range(len(KEY)), KEY, rotation=30, ha="right", fontsize=9)
    ax[1].set_yticks(range(len(M)), M.index)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax[1].text(j, i, f"{M.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.colorbar(im, ax=ax[1], fraction=0.03, label="|Δ특징| 중앙값 / 데이터셋 표준편차")
    ax[1].set_title("핵심 특징의 변화량 (데이터셋 표준편차 단위)")
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "figures" / "fig_aug_invariance.png", dpi=100, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
