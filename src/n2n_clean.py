"""고품질 이미지의 잔여 노이즈 제거 (Noisier2Noise 보정, 재학습 없음).

발견(2026-10-07): 부트스트랩 디노이저는 σ<7 이미지를 거의 그대로 통과시킨다(잔차 std 0.05, train·test 동일 → 암기 아님).
정답 자체에 σ0 노이즈가 있어 'σ0 수준 노이즈 = 신호'로 학습했기 때문. 고품질(σ<th) 이미지는 노이즈 σ4~7이 남은 채 DINO·상분율에 들어감.
Noisier2Noise: 디노이저 f ≈ E[y | y+m] (y = 깨끗+n, m ~ n과 같은 분포)이면 E[깨끗 | z] = 2 f(z) − z (z = y+m).
→ 원본 y에 추정 σ0(Haar MAD)만큼 노이즈를 더해 z를 만들고 2 f(z) − z 를 K번 평균. 저품질(σ≥th)은 기존 출력 유지.
이미지별 고정 연산(난수 시드 고정), test는 추론만.

실행: python src/n2n_clean.py --th 10 --k 8
산출물: outputs/denoised/unet_<base>_n2n<th>.npz
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_cnn import DATA, OUT, load_split  # noqa: E402
from train_denoiser import UNet, denoise, haar_sigma  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/denoiser_cm7_am22_l1_i10000_boot.pt")
    ap.add_argument("--base", default="cm7_am22_l1_i10000_boot", help="저품질 이미지에 쓸 기존 디노이즈 캐시 태그")
    ap.add_argument("--th", type=float, default=10.0, help="이 σ 미만 이미지에 Noisier2Noise 보정")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--scale", type=float, default=1.0, help="추가 노이즈 σ = scale × 추정 σ0")
    a = ap.parse_args()
    net = UNet().cuda()
    net.load_state_dict(torch.load(ROOT / a.weights, map_location="cuda"))
    base = np.load(OUT / "denoised" / f"unet_{a.base}.npz")
    out = {}
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        ids = pd.read_csv(DATA / csv).ID
        sig = pd.read_csv(OUT / f"features/{split}_features.csv").set_index("ID").loc[ids, "q_noise_sigma"].values
        X = load_split(split, ids)
        res = base[split].copy()
        idx = np.where(sig < a.th)[0]
        g = torch.Generator().manual_seed(0)
        y = X[idx].float()
        s0 = haar_sigma(y / 255).view(-1, 1, 1, 1) * 255 * a.scale
        acc = torch.zeros_like(y)
        for _ in range(a.k):
            z = (y + torch.randn(y.shape, generator=g) * s0).clamp(0, 255).round()
            fz = denoise(net, z.to(torch.uint8)).float()
            acc += 2 * fz - z
        res[idx] = (acc / a.k).clamp(0, 255).round().to(torch.uint8).squeeze(1).numpy()
        out[split] = res
        r = (y.squeeze(1).numpy() - res[idx]).std(axis=(1, 2))
        print(f"  {split}: {len(idx)}장 보정, 잔차 std 중앙값 {np.median(r):.2f} (추정 σ0 중앙값 {np.median(sig[idx]):.2f})", flush=True)
    tag = f"{a.base}_n2n{a.th:g}" + (f"_s{a.scale:g}" if a.scale != 1 else "") + (f"_k{a.k}" if a.k != 8 else "")
    np.savez_compressed(OUT / "denoised" / f"unet_{tag}.npz", **out)
    print(f"[unet_{tag}] 저장")


if __name__ == "__main__":
    main()
