"""조밀한 상(phase) 분율 측정 — 디노이즈(+조명보정) 이미지 캐시(npz)에서 이미지별 고정 연산 (학습·정답 미사용).

근거(2026-10-07): DINO 커널 + 수작업 ph_ 특징(15개, 디노이즈+조명보정 이미지) 결합이 11.327 → 10.949.
DINO는 2상(어두운 결정립) 면적분율을 정밀하게 재지 못하고, 디노이즈 이미지에서는 이 측정이 정확해진다.
기존 ph_ 는 임계값 5개·고정 상대어둡기라 저품질 모드(대비 ≈0.77배)에서 분율이 줄어드는 문제 → 조밀 CDF + 대비 적응형.

- closing 반지름 r ∈ --radii (0 = 없음): 얇은 결정립계 선 제거 후 기지(최빈 밝기) 대비 상대 어둡기 rel=(m−x)/m
  · rel CDF: rel > t 면적분율 (t = −0.20 … 0.60, 0.025 간격) · rel 분위수 · 어두운 질량
  · 대비 정규화 rel/s (s = 기지 주변 분포 폭이 아니라 rel 의 p99−p50 = 가장 어두운 상까지의 폭) CDF
- multi-Otsu(3·4 클래스) 클래스 면적분율 (이미지별 적응 임계값)

실행: python src/phase_features.py --src outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz
산출물: outputs/features_cache/phase_<stem>_{train,test}.csv
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.signal import find_peaks
from skimage import filters, morphology

ROOT = Path(__file__).resolve().parents[1]
T_ABS = np.round(np.arange(-0.20, 0.6001, 0.025), 3)
T_NORM = np.round(np.arange(-1.0, 1.501, 0.1), 2)
QS = (1, 2, 5, 10, 25, 50, 75, 90, 95, 98, 99, 99.5)
RADII = (0, 2, 3, 5)


def matrix_level(x: np.ndarray, mode: str = "mode") -> float:
    """기지 밝기. mode: 평활 히스토그램 최빈값 / bright: 최빈값의 30% 이상인 봉우리 중 가장 밝은 것
    (2상이 기지만큼 많은 쌍봉 이미지에서 최빈값이 어두운 봉우리로 뒤집히는 불안정 방지)"""
    h, b = np.histogram(x, bins=256, range=(0, 256))
    hs = ndi.gaussian_filter1d(h.astype(float), 2)
    if mode == "bright":
        pk, _ = find_peaks(hs, prominence=hs.max() * 0.05)
        pk = [p for p in pk if hs[p] >= 0.3 * hs.max()]
        if pk:
            return float(b[max(pk)] + 0.5)
    return float(b[hs.argmax()] + 0.5)


def local_bg(c: np.ndarray, block: int) -> np.ndarray:
    """국소 기지 밝기 지도: block×block 칸마다 상위 60% 분위(기지가 다수상) → 3×3 중앙값 → 가우시안 확대"""
    n = c.shape[0] // block
    g = np.percentile(c[:n * block, :n * block].reshape(n, block, n, block).transpose(0, 2, 1, 3).reshape(n, n, -1), 60, axis=2)
    g = ndi.median_filter(g, 3, mode="nearest")
    up = ndi.zoom(g, c.shape[0] / n, order=1, mode="nearest")
    return ndi.gaussian_filter(up, block / 2, mode="nearest")


def _one(img: np.ndarray, radii=RADII, bg: int = 0, matrix: str = "mode") -> dict:
    x = img.astype(np.float64)
    f = {}
    for r in radii:
        c = morphology.closing(x, morphology.disk(r)) if r > 0 else x
        if bg:  # 국소 기지 밝기 기준 (조명 잔여 경사에 강건) → 전역 최빈값으로 다시 맞춤
            lb = local_bg(c, bg)
            c = c / lb * float(np.median(lb))
        m = matrix_level(c, matrix)
        rel = (m - c) / m
        p = f"pf_r{r}_"
        for t in T_ABS:
            f[f"{p}gt{t:+.3f}"] = float((rel > t).mean())
        qv = np.percentile(rel, QS)
        for q, v in zip(QS, qv):
            f[f"{p}q{q}"] = float(v)
        f[f"{p}mass"] = float(np.clip(rel, 0, None).mean())
        s = max(qv[QS.index(99)] - qv[QS.index(50)], 1e-3)
        f[f"{p}scale"] = s
        rn = (rel - qv[QS.index(50)]) / s
        for t in T_NORM:
            f[f"{p}n{t:+.2f}"] = float((rn > t).mean())
    for k in (3, 4):
        th = filters.threshold_multiotsu(x.astype(np.uint8) if x.max() > 1 else x, classes=k)
        cls = np.digitize(x, th)
        for j in range(k):
            f[f"pf_otsu{k}_c{j}"] = float((cls == j).mean())
        for j, t in enumerate(th):
            f[f"pf_otsu{k}_t{j}"] = float(t) / matrix_level(x, matrix)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--bg", type=int, default=0, help="국소 기지 밝기 칸 크기(px, 0=전역 최빈값)")
    ap.add_argument("--radii", type=int, nargs="+", default=list(RADII))
    ap.add_argument("--matrix", choices=["mode", "bright"], default="mode", help="기지 밝기 기준 (matrix_level 참고)")
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/sample_submission.csv")
    z = np.load(ROOT / a.src)
    stem = Path(a.src).stem + (f"_bg{a.bg}" if a.bg else "") + ("_mb" if a.matrix == "bright" else "")
    fn = partial(_one, radii=tuple(a.radii), bg=a.bg, matrix=a.matrix)
    fd = ROOT / "outputs/features_cache"
    fd.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for split, ids in (("train", tr.ID), ("test", te.ID)):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(fn, list(z[split]), chunksize=16))
        df = pd.DataFrame(rows).assign(ID=ids.values)
        df.to_csv(fd / f"phase_{stem}_{split}.csv", index=False)
        print(f"[phase_{stem}] {split} {df.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
