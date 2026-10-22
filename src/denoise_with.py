"""저장된 디노이저 가중치로 이미지 캐시(npz 또는 원본 PNG)를 디노이즈 (추론만, D4 평균).

예) python src/denoise_with.py --weights weights/denoiser_cm7_am22_l1_i10000_boot.pt --src outputs/denoised/raw_fflowpass48_low.npz --tag boot_on_rawff48
산출물: outputs/denoised/unet_<tag>.npz
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
from train_denoiser import UNet, denoise  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", default=None, help="입력 캐시 npz (없으면 원본 PNG)")
    ap.add_argument("--width", type=int, default=48)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    if a.src:
        z = np.load(ROOT / a.src)
        X, Xte = torch.from_numpy(z["train"]).unsqueeze(1), torch.from_numpy(z["test"]).unsqueeze(1)
    else:
        X = load_split("train", pd.read_csv(DATA / "train.csv").ID)
        Xte = load_split("test", pd.read_csv(DATA / "sample_submission.csv").ID)
    m = UNet(tuple(a.width * 2 ** i for i in range(4))).cuda()
    m.load_state_dict(torch.load(ROOT / a.weights, map_location="cuda"))
    p = OUT / "denoised" / f"unet_{a.tag}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, train=denoise(m, X).squeeze(1).numpy(), test=denoise(m, Xte).squeeze(1).numpy())
    print(f"[unet_{a.tag}] 저장 {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
