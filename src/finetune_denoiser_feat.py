"""디노이저 특징 일치 미세조정: 노이즈 → 디노이저 출력의 DINOv2 고정 특징이 깨끗한 정답 이미지의 특징과 같아지도록 학습.

배경: 경도 모델(Ridge)은 디노이즈 이미지의 DINOv2-L hetero3 특징(층 11~17, 448)을 쓴다. 픽셀 손실로만 학습한 디노이저는
고노이즈에서 구조를 '그럴듯하게' 지어내 특징이 깨끗한 이미지와 달라지고(σ≥10 오차 ≈13.4), 이 차이를 직접 줄이는 것이 목적.
- 시작점: 부트스트랩 디노이저 가중치 (--init)
- 정답: 같은 부트스트랩 정답 이미지(--target-src, σ<7, train만). 정답의 D4 8방향 특징을 미리 계산(차원별 표준화 통계 포함)
- 손실: L1(픽셀) + λ·(표준화 특징 차이)² — λ는 첫 배치에서 두 손실이 같은 크기가 되도록 자동 설정 × --lam
- DINO는 고정(파라미터 학습 없음), 기울기만 디노이저로 전달. 경도 정답 미사용. test 이미지는 추론만.

예) python src/finetune_denoiser_feat.py --init weights/denoiser_cm7_am22_l1_i10000_boot.pt --iters 2000
산출물: weights/denoiser_<tag>.pt, outputs/denoised/unet_<tag>.npz
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_models import build, pool  # noqa: E402
from train_cnn import DATA, OUT, d4, load_split  # noqa: E402
from train_denoiser import UNet, add_noise, denoise  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True)
    ap.add_argument("--target-src", default="outputs/denoised/unet_cm7_am22_l1_i6000.npz")
    ap.add_argument("--clean-max", type=float, default=7.0)
    ap.add_argument("--add-max", type=float, default=22.0)
    ap.add_argument("--add-min", type=float, default=0.0)
    ap.add_argument("--layers", type=int, nargs="+", default=[11, 13, 15, 17])
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lam", type=float, default=1.0, help="특징 손실 상대 가중 (첫 배치 기준 L1과 같은 크기 = 1)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    train = pd.read_csv(DATA / "train.csv")
    test = pd.read_csv(DATA / "sample_submission.csv")
    sig = pd.read_csv(OUT / "features/train_features.csv").set_index("ID").loc[train.ID, "q_noise_sigma"].values
    X, Xte = load_split("train", train.ID), load_split("test", test.ID)
    ci = np.where(sig < a.clean_max)[0]
    rng = np.random.default_rng(0)  # train_denoiser.py와 같은 정답/검증 분할
    rng.shuffle(ci)
    nval = max(4, len(ci) // 10)
    vi, ti = ci[:nval], ci[nval:]
    T = torch.from_numpy(np.load(ROOT / a.target_src)["train"]).unsqueeze(1)
    C = T[ti].cuda().float().div_(255)
    V = T[vi].cuda().float().div_(255)
    model = UNet().cuda()
    model.load_state_dict(torch.load(ROOT / a.init, map_location="cuda"))
    dino, mean, std = build("vit_large_patch14_dinov2.lvd142m", a.size)
    for p_ in dino.parameters():
        p_.requires_grad_(False)
    dino.set_grad_checkpointing(True)

    def feats(x):  # x: B,1,256,256 [0,1] → hetero3 특징 (기울기 유지)
        xi = F.interpolate(x, size=(a.size, a.size), mode="bilinear", align_corners=False)
        xi = (xi.repeat(1, 3, 1, 1) - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fs = dino.forward_intermediates(xi, indices=a.layers, intermediates_only=True, stop_early=True)
        return torch.cat([pool(z, "hetero3") for z in fs], 1)

    t0 = time.time()
    with torch.no_grad():
        Fc = torch.stack([torch.cat([feats(d4(C[i:i + 16], k)) for i in range(0, len(C), 16)]) for k in range(8)], 1)  # N,8,D
        FcV = torch.stack([feats(d4(V, k)) for k in range(8)], 1)
    mu, sd = Fc.flatten(0, 1).mean(0), Fc.flatten(0, 1).std(0) + 1e-4
    print(f"정답 {len(ti)}장 특징 {tuple(Fc.shape)} 계산 ({time.time() - t0:.0f}s)", flush=True)

    def feat_loss(f, fc):
        return (((f - fc) / sd) ** 2).mean()

    @torch.no_grad()
    def evaluate():
        model.eval()
        out = {}
        for s in (10, 20):
            g = torch.Generator(device="cuda").manual_seed(s)
            xn = ((V + torch.randn(V.shape, device="cuda", generator=g) * s / 255).clamp(0, 1) * 255).round() / 255
            with torch.autocast("cuda", dtype=torch.bfloat16):
                dv = model(xn).float().clamp(0, 1)
            out[s] = (10 * math.log10(1 / F.mse_loss(dv, V).item()), feat_loss(feats(dv), FcV[:, 0]).item())
        model.train()
        return " | ".join(f"σ{s}: PSNR {p:.2f} 특징차 {f:.3f}" for s, (p, f) in out.items())

    print(f"  시작 검증: {evaluate()}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    lam = None
    model.train()
    for it in range(a.iters):
        lr = a.lr * min(1.0, (it + 1) / 100) * 0.5 * (1 + math.cos(math.pi * it / a.iters))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = torch.randint(0, len(C), (a.batch,), device="cuda")
        k = int(np.random.randint(8))
        y = d4(C[idx], k)
        s = (a.add_min + (a.add_max - a.add_min) * torch.rand(len(y), 1, 1, 1, device="cuda")) / 255
        x = ((y + torch.randn_like(y) * s).clamp(0, 1) * 255).round() / 255
        with torch.autocast("cuda", dtype=torch.bfloat16):
            den = model(x).float()
        lp = (den - y).abs().mean()
        lf = feat_loss(feats(den.clamp(0, 1)), Fc[idx, k])
        if lam is None:
            lam = a.lam * lp.item() / max(lf.item(), 1e-8)
            print(f"  λ = {lam:.4g} (L1 {lp.item():.4f}, 특징 {lf.item():.4f})", flush=True)
        loss = lp + lam * lf
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if (it + 1) % 250 == 0:
            print(f"  it{it + 1} L1 {lp.item():.4f} 특징 {lf.item():.4f} | 검증 {evaluate()} ({time.time() - t0:.0f}s)", flush=True)
    tag = a.tag or f"featft_lam{a.lam:g}_i{a.iters}"
    torch.save(model.state_dict(), ROOT / "weights" / f"denoiser_{tag}.pt")
    Dtr, Dte = denoise(model, X), denoise(model, Xte)
    p = OUT / "denoised" / f"unet_{tag}.npz"
    np.savez_compressed(p, train=Dtr.squeeze(1).numpy(), test=Dte.squeeze(1).numpy())
    print(f"[unet_{tag}] 저장 {p.relative_to(ROOT)} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
