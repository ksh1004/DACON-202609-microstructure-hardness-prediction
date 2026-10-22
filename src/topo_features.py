"""위상(topology) 특징: 상대 어둡기 임계값별 연결성 — 어두운 상(2상·결정립계)·밝은 영역의 조각 수, 오일러 수, 조각 크기, 경계 길이.

상대 어둡기 r = 1 − I / m (m = 이미지의 70% 분위 밝기 = 기지 수준, phase_features의 dark_map과 같은 기준).
임계값 r_t (0.0~0.5, 20개)마다 어두운 집합 D = {r > r_t}:
  - n_d: D의 8-연결 조각 수, n_b: 여집합의 4-연결 조각 수 (D 안의 구멍 + 바깥)
  - euler: n_d − (n_b − 1) (평면 위상: 구멍 수 ≈ 밝은 조각 수 − 1)
  - msz: D 조각 평균 면적(화소), per: D 경계 화소 비율(4-이웃 중 하나라도 밖)
두 규모: 디노이즈 이미지 그대로 / 가우시안 σ2 평활. 모든 값은 이미지 하나만 보고 계산 (정답·test 통계 미사용).

실행: python src/topo_features.py --src outputs/denoised/<stem>.npz
산출물: outputs/features_cache/topo_<stem>_{train,test}.csv (열 접두어 tp_)
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
THR = np.round(np.linspace(0.0, 0.5, 20), 4)
S8 = np.ones((3, 3), bool)


def one(img: np.ndarray) -> dict:
    out = {}
    for sc, x in (("s0", img.astype(np.float64)), ("s2", ndi.gaussian_filter(img.astype(np.float64), 2))):
        m = np.quantile(x, 0.7) + 1e-6
        r = 1 - x / m
        for t in THR:
            D = r > t
            nd = ndi.label(D, structure=S8)[1]
            nb = ndi.label(~D)[1]
            area = D.sum()
            inner = ndi.binary_erosion(D)
            k = f"{sc}_{t:.3f}"
            out[f"tp_nd_{k}"] = nd / 1000
            out[f"tp_nb_{k}"] = nb / 1000
            out[f"tp_eu_{k}"] = (nd - (nb - 1)) / 1000
            out[f"tp_msz_{k}"] = np.log1p(area / max(nd, 1))
            out[f"tp_per_{k}"] = (area - inner.sum()) / D.size
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    a = ap.parse_args()
    z = np.load(ROOT / a.src)
    stem = Path(a.src).stem
    t0 = time.time()
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        ids = pd.read_csv(ROOT / "data" / csv).ID
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(one, list(z[split]), chunksize=8))
        df = pd.DataFrame(rows)
        df.insert(0, "ID", ids.values)
        out = ROOT / "outputs/features_cache" / f"topo_{stem}_{split}.csv"
        df.to_csv(out, index=False)
        print(f"[topo_{stem}] {split} {df.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
