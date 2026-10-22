"""2점 상관함수(two-point statistics) 특징 — 디노이즈(+조명보정) 이미지 캐시(npz), 이미지별 고정 연산 (학습·정답 미사용).

미세구조 정보학의 표준 기술자: 상 지시함수 b(x)=[rel(x)>t] 의 자기상관 S2(r) = P(b(x)=1, b(x+r)=1) 를 방향 평균.
S2(0)=분율, S2(r)의 감쇠 길이 = 2상 덩어리 크기·간격. 회전 불변(방향 평균) → 절대 방향 무관 규칙과 일치.
- tp_t{t}_r{r}: 정규화 (S2(r) − f²)/(f − f²), r = 1…64 px (로그 간격), t = 상대어둡기 임계값 (closing r=2 후)
- tp_t{t}_aniso: 가로·세로가 아닌 '최대/최소 방향' S2(4) 비 (정렬도)
- tg_r{r}: 그레이 이미지 정규화 자기상관 (결정립 크기 척도)

실행: python src/twopoint_features.py --src outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz
산출물: outputs/features_cache/twopt_<stem>_{train,test}.csv
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import morphology

from phase_features import matrix_level

ROOT = Path(__file__).resolve().parents[1]
TS = (0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35)
RS = (1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64)
N = 256
_yy, _xx = np.mgrid[-N // 2:N // 2, -N // 2:N // 2]
RAD = np.sqrt(_yy ** 2 + _xx ** 2)
ANG = np.arctan2(_yy, _xx) % np.pi


def autocorr(a: np.ndarray) -> np.ndarray:
    """제로패딩 FFT 자기상관 (비주기), 겹치는 픽셀 수로 정규화, 중심이 (N/2, N/2)"""
    p = np.zeros((2 * N, 2 * N))
    p[:N, :N] = a
    F = np.fft.rfft2(p)
    ac = np.fft.irfft2(F * np.conj(F), s=p.shape)
    o = np.zeros((2 * N, 2 * N))
    o[:N, :N] = 1
    Fo = np.fft.rfft2(o)
    cnt = np.fft.irfft2(Fo * np.conj(Fo), s=p.shape)
    ac = np.fft.fftshift(ac / np.maximum(cnt, 1))
    return ac[N - N // 2:N + N // 2, N - N // 2:N + N // 2]


def radial(ac: np.ndarray) -> np.ndarray:
    return np.array([ac[(RAD >= r - 0.5) & (RAD < r + 0.5)].mean() for r in RS])


def _one(img: np.ndarray) -> dict:
    x = img.astype(np.float64)
    f = {}
    c = morphology.closing(x, morphology.disk(2))
    m = matrix_level(c)
    rel = (m - c) / m
    for t in TS:
        b = (rel > t).astype(np.float64)
        fr = b.mean()
        if fr < 1e-4 or fr > 1 - 1e-4:
            for r in RS:
                f[f"tp_t{t}_r{r}"] = 0.0
            f[f"tp_t{t}_aniso"] = 1.0
            continue
        ac = (autocorr(b) - fr ** 2) / (fr - fr ** 2)
        for r, v in zip(RS, radial(ac)):
            f[f"tp_t{t}_r{r}"] = float(v)
        ring = (RAD >= 3.5) & (RAD < 4.5)
        h, _ = np.histogram(ANG[ring], bins=8, range=(0, np.pi), weights=ac[ring])
        n, _ = np.histogram(ANG[ring], bins=8, range=(0, np.pi))
        prof = h / np.maximum(n, 1)
        f[f"tp_t{t}_aniso"] = float((prof.max() + 1e-3) / (prof.min() + 1e-3))
    g = x - x.mean()
    acg = autocorr(g) / (g.var() + 1e-9)
    for r, v in zip(RS, radial(acg)):
        f[f"tg_r{r}"] = float(v)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/sample_submission.csv")
    z = np.load(ROOT / a.src)
    stem = Path(a.src).stem
    t0 = time.time()
    for split, ids in (("train", tr.ID), ("test", te.ID)):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(_one, list(z[split]), chunksize=16))
        df = pd.DataFrame(rows).assign(ID=ids.values)
        df.to_csv(ROOT / "outputs/features_cache" / f"twopt_{stem}_{split}.csv", index=False)
        print(f"[twopt_{stem}] {split} {df.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
