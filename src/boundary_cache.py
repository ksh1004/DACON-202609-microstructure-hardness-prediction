"""결정립계 강조 이미지 캐시: 디노이즈 이미지의 black top-hat(closing − 원본)으로 얇은 어두운 선(결정립계)만 남긴다.

2상(넓은 어두운 결정립)·기공(큰 검은 반점)은 구조요소보다 커서 closing으로 채워지지 않으므로 대부분 사라진다
→ 밝기 대비와 분리된 결정립 크기·형상·배열 정보만 담은 입력. 이미지별 고정 연산(학습 없음).
출력은 흰 바탕에 어두운 선: 255 − clip(gain · tophat).

실행: python src/boundary_cache.py --src outputs/denoised/nlm_c0.8.npz --r 2 --gain 4
산출물: outputs/denoised/bnd_<src>_r<r>_g<gain>.npz  (probe_models.py --src 로 사용)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from skimage.morphology import black_tophat, disk

ROOT = Path(__file__).resolve().parents[1]


def boundary(x: np.ndarray, r: int, gain: float) -> np.ndarray:
    th = black_tophat(x, disk(r)).astype(np.float32)
    return (255 - np.clip(gain * th, 0, 255)).round().astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="outputs/denoised/nlm_c0.8.npz")
    ap.add_argument("--r", type=int, default=2)
    ap.add_argument("--gain", type=float, default=4.0)
    a = ap.parse_args()
    z = np.load(ROOT / a.src)
    out = {k: np.stack([boundary(x, a.r, a.gain) for x in z[k]]) for k in ("train", "test")}
    p = ROOT / "outputs" / "denoised" / f"bnd_{Path(a.src).stem}_r{a.r}_g{a.gain:g}.npz"
    np.savez_compressed(p, **out)
    print(f"[{p.name}] train {out['train'].shape} test {out['test'].shape}")


if __name__ == "__main__":
    main()
