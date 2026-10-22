"""조명 불균일 보정 캐시: 디노이즈 캐시에서 이미지별 밝기 평면(선형 경사) 또는 넓은 저역(가우시안) 성분을 제거.

근거(2026-10-06): 저품질 촬영 모드(노이즈 σ≥10)는 블러·대비 감소와 함께 조명 경사/얼룩이 더해짐(평면 경사 중앙값 13.6→17.9,
σ32 저역 표준편차 4.3→5.4) → hetero3(영역 간 불균일성) 특징을 오염. 고품질 모드의 저역 성분은 경도와 상관(ρ 0.29)이라 신호.
모드는 원본 이미지의 노이즈 추정치(features.py q_noise_sigma ≥ --sig-th)로 이미지마다 판별(test도 동일 규칙, 학습 없음).

예) python src/flatfield_cache.py --src outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz --mode plane --only-low
산출물: outputs/denoised/<src>_ff<mode>[<σ>][_low].npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
WIN = 31  # matrix 모드 클로징 창 (main에서 --win으로 설정)


def correct(d: np.ndarray, mode: str, s: float) -> np.ndarray:
    x = d.astype(np.float64)
    if mode == "plane":
        yy, xx = np.mgrid[0:x.shape[0], 0:x.shape[1]] / (x.shape[0] - 1)
        M = np.c_[np.ones(x.size), xx.ravel(), yy.ravel()]
        c = np.linalg.lstsq(M, x.ravel(), rcond=None)[0]
        x = x - (c[1] * xx + c[2] * yy) + (c[1] + c[2]) / 2
    elif mode == "matrix":
        bg = ndi.gaussian_filter(ndi.grey_closing(x, size=(WIN, WIN), mode="reflect"), s, mode="reflect")
        x = x - bg + bg.mean()
    else:
        lp = ndi.gaussian_filter(x, s, mode="reflect")
        x = x - lp + lp.mean()
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def stretch(d: np.ndarray, k: float) -> np.ndarray:
    if k == 1.0:
        return d
    x = d.astype(np.float64)
    return np.clip(np.round(x.mean() + k * (x - x.mean())), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--mode", choices=["plane", "lowpass", "matrix"], default="plane",
                    help="matrix: 그레이 클로징(창 --win)으로 어두운 2상·기공을 지운 기지상 밝기 지도를 σ로 평활 → 조명으로 보고 제거 (2상 밀도의 지역 차이는 보존)")
    ap.add_argument("--win", type=int, default=31)
    ap.add_argument("--sigma", type=float, default=48.0)
    ap.add_argument("--only-low", action="store_true", help="저품질 모드 이미지(σ≥--sig-th)만 보정")
    ap.add_argument("--sig-th", type=float, default=10.0)
    ap.add_argument("--contrast", type=float, default=1.0, help="보정 대상 이미지의 대비 배율(이미지 평균 기준), 저품질 대비 감소(≈0.77배) 보상용")
    ap.add_argument("--plane-all", action="store_true", help="추가로 모든 이미지에서 평면(선형 경사) 제거 (보정 대상은 평면 제거 후 --mode 적용)")
    a = ap.parse_args()
    global WIN
    WIN = a.win
    z = np.load(ROOT / a.src)
    out = {}
    for split, csv in (("train", "train"), ("test", "test")):
        ids = pd.read_csv(ROOT / "data" / ("train.csv" if split == "train" else "sample_submission.csv")).ID
        sig = pd.read_csv(ROOT / f"outputs/features/{csv}_features.csv").set_index("ID").loc[ids, "q_noise_sigma"].values
        do = sig >= a.sig_th if a.only_low else np.ones(len(ids), bool)
        imgs = [correct(d, "plane", 0) if a.plane_all else d for d in z[split]]
        out[split] = np.stack([stretch(correct(d, a.mode, a.sigma), a.contrast) if f else d for d, f in zip(imgs, do)])
        print(f"  {split}: 보정 {do.sum()}/{len(do)}장")
    tag = f"_ff{a.mode}{'' if a.mode == 'plane' else int(a.sigma)}{f'w{a.win}' if a.mode == 'matrix' else ''}{'_low' if a.only_low else ''}{'_pall' if a.plane_all else ''}{f'_k{a.contrast:g}' if a.contrast != 1 else ''}{f'_th{a.sig_th:g}' if a.sig_th != 10 else ''}"
    p = ROOT / "outputs/denoised" / (Path(a.src).stem + tag + ".npz")
    np.savez_compressed(p, **out)
    print(f"[{p.name}]")


if __name__ == "__main__":
    main()
