"""원본(디노이즈 전) 이미지의 조밀 스펙트럼·자기상관 곡선 — 이미지별 고정 연산 (학습·정답 미사용).

근거: 백색 노이즈는 파워스펙트럼에 평평한 바닥으로만 더해지고 자기상관에서는 지연 0에만 나타난다.
→ 원본 PSD에서 바닥을 빼면 디노이저가 결정립계를 지어내는 저품질 모드에서도 구조 스펙트럼(결정립 규모)을 편향 없이 잴 수 있다.
저품질 모드(q_noise_sigma ≥ 10)는 가우시안 블러 σb≈1px 이므로 MTF²(= exp(−(2πf)²σb²))로 나눠 보정한다 (--blur, 고정 규칙).

- rp_psd{i}: 방사 평균 PSD(바닥 제거, 블러 보정)를 로그 간격 주파수 구간별로 합한 뒤 전체 신호 파워로 정규화한 것의 로그
- rp_acf{l}: 신호 자기상관(바닥 제거 PSD의 역변환, 방사 평균)의 지연 l 값 / 지연 0 값
- rp_cum{q}: 신호 파워 누적 분위 주파수

실행: python src/raw_psd_features.py
산출물: outputs/features_cache/rawpsd_{train,test}.csv
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
N = 256
RMAX = 64  # 주기 4px까지 (그 위는 블러 보정 시 노이즈 증폭)
EDGES = np.unique(np.round(np.geomspace(1, RMAX + 1, 25)).astype(int))
LAGS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 64)
_yy, _xx = np.indices((N, N))
_R = np.hypot(_yy - N // 2, _xx - N // 2)
_RI = _R.astype(int)
_CNT = np.maximum(np.bincount(_RI.ravel()), 1)
_WIN = np.outer(np.hanning(N), np.hanning(N))  # 가장자리 불연속 누설 방지 (백색 바닥은 평평하게 유지됨)


def _one(args, blur: float = 1.0) -> dict:
    path, sigma = args
    x = np.asarray(Image.open(path), dtype=np.float64)
    z = x - x.mean()
    P = np.abs(np.fft.fftshift(np.fft.fft2(z * _WIN))) ** 2 / (_WIN ** 2).sum()
    floor = np.median(P[_R > 140])  # 모서리(주기 < 1.8px)는 구조 신호가 거의 없음 → 노이즈 바닥
    S = P - floor
    if sigma >= 10 and blur > 0:  # 저품질 모드 블러 보정 (RMAX 밖은 사용하지 않음)
        S = S / np.exp(-((2 * np.pi * np.minimum(_R, RMAX) / N) ** 2) * blur ** 2)
    S[_R > RMAX] = 0
    S[N // 2, N // 2] = 0
    prof = np.bincount(_RI.ravel(), S.ravel()) / _CNT  # 방사 평균 (음수 허용: 평균하면 편향 없음)
    ring = prof[:RMAX + 1] * (2 * np.pi * np.arange(RMAX + 1))  # 고리 전체 파워
    tot = max(ring[1:].sum(), 1e-9)
    f = {"rp_logpow": float(np.log(tot)), "rp_floor_sigma": float(np.sqrt(max(floor, 0)))}
    for i, (lo, hi) in enumerate(zip(EDGES[:-1], EDGES[1:])):
        f[f"rp_psd{i:02d}"] = float(np.log(max(ring[lo:hi].sum() / tot, 1e-4)))
    cum = np.cumsum(np.clip(ring[1:], 0, None))
    cum = cum / max(cum[-1], 1e-9)
    for q in (10, 25, 50, 75, 90):
        f[f"rp_cum{q}"] = float(np.log(1 + np.searchsorted(cum, q / 100)))
    acf = np.fft.fftshift(np.real(np.fft.ifft2(np.fft.ifftshift(S))))
    ap = np.bincount(_RI.ravel(), acf.ravel()) / _CNT
    a0 = max(ap[0], 1e-9)
    for l in LAGS:
        f[f"rp_acf{l:02d}"] = float(ap[l] / a0)
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blur", type=float, default=1.0, help="저품질 모드 블러 σ(px) 보정값 (0 = 보정 없음)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    tag = "rawpsd" + ("" if a.blur == 1.0 else f"_b{a.blur:g}")
    fd = ROOT / "outputs/features_cache"
    fd.mkdir(parents=True, exist_ok=True)
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        ids = pd.read_csv(ROOT / "data" / csv).ID
        sig = pd.read_csv(ROOT / f"outputs/features/{split}_features.csv").set_index("ID").loc[ids, "q_noise_sigma"].values
        jobs = [(ROOT / "data" / split / f"{i}.png", s) for i, s in zip(ids, sig)]
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(partial(_one, blur=a.blur), jobs, chunksize=32))
        pd.DataFrame(rows).assign(ID=ids.values).to_csv(fd / f"{tag}_{split}.csv", index=False)
        print(f"[{tag}] {split} {len(rows)}장 × {len(rows[0])}열", flush=True)


if __name__ == "__main__":
    main()
