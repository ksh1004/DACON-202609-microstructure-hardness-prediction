"""저품질 모드 합성 사본 (train 증강, 정답 = 원본 경도): 고품질 train 이미지 → 대비↓·블러·노이즈 열화 → 같은 처리 경로
(부트스트랩 디노이저 → σ48 저역 조명 보정 → DINOv2-L het3@448 특징 + 상분율 특징).

근거(2026-10-07): 오차의 57%가 저품질 모드(σ≥10). 예전 합성 사본 실험은 조명 보정 발견 전이라 실제 저품질과 DINO로 AUC 0.993
구분(주원인 조명 얼룩) → 실패. 지금은 실제 저품질도 조명 보정 후 특징을 쓰므로 합성(조명 없음)과 차이가 줄었을 수 있다.
test 이미지는 사용하지 않는다 (train 원본 열화만).

실행: python src/synlow_aug.py --clean-max 10 --copies 2
산출물: outputs/synlow/<tag>/{emb.npz(train 사본 특징), phase.csv, meta.csv(원본 ID·열화 파라미터)}
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flatfield_cache import correct  # noqa: E402
from phase_features import _one as phase_one  # noqa: E402
from probe_models import build, embed  # noqa: E402
from train_cnn import DATA, OUT, load_split  # noqa: E402
from train_denoiser import UNet, denoise, gblur  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-max", type=float, default=10.0, help="원본으로 쓸 고품질 이미지 σ 상한")
    ap.add_argument("--copies", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--c-lo", type=float, default=0.65)
    ap.add_argument("--c-hi", type=float, default=0.95)
    ap.add_argument("--b-lo", type=float, default=0.6)
    ap.add_argument("--b-hi", type=float, default=1.4)
    ap.add_argument("--n-lo", type=float, default=10.0)
    ap.add_argument("--n-hi", type=float, default=20.0)
    ap.add_argument("--tone", action="store_true", help="분위수 매핑 톤 곡선(outputs/synlow/tone_map.npy) + 비대칭 조명장 (열화 모델 v2)")
    ap.add_argument("--illum-a", type=float, nargs=2, default=[1.0, 3.0])
    ap.add_argument("--illum-k", type=float, default=1.0)
    a = ap.parse_args()
    tag = f"cm{a.clean_max:g}_x{a.copies}_s{a.seed}" + ("_tone" if a.tone else "")
    d = OUT / "synlow" / tag
    d.mkdir(parents=True, exist_ok=True)
    tr = pd.read_csv(DATA / "train.csv")
    sig = pd.read_csv(OUT / "features/train_features.csv").set_index("ID").loc[tr.ID, "q_noise_sigma"].values
    src = np.where(sig < a.clean_max)[0]
    X = load_split("train", tr.ID)[src].float().div(255).cuda()  # N,1,H,W
    g = torch.Generator(device="cuda").manual_seed(a.seed)
    rows, imgs = [], []
    if a.tone:  # 열화 모델 v2 (2026-10-07 진단): 톤 곡선(어두운 쪽 압축) → 가우시안 블러 → 양의 왜도 저주파 조명장 → 노이즈
        from scipy import ndimage as ndi
        qh, ql = np.load(OUT / "synlow/tone_map.npy")
        Xn = X.squeeze(1).cpu().numpy() * 255
        for c in range(a.copies):
            rng = np.random.default_rng(a.seed * 1000 + c)
            out = []
            for i, x in zip(src, Xn):
                m = x.mean(); y = m + np.interp(x - m, qh, ql)
                sb = rng.uniform(a.b_lo, a.b_hi); y = ndi.gaussian_filter(y, sb)
                z = ndi.gaussian_filter(rng.normal(size=x.shape), rng.uniform(32, 64), mode="wrap"); z = (z - z.mean()) / z.std()
                f = np.exp(a.illum_k * z) if a.illum_k else z; f = (f - f.mean()) / f.std()
                A = rng.uniform(*a.illum_a); sn = rng.uniform(a.n_lo, a.n_hi)
                y = y + A * f + rng.normal(0, sn, x.shape)
                out.append(np.clip(np.round(y), 0, 255).astype(np.uint8))
                rows.append({"ID": tr.ID[i], "copy": c, "contrast": float("nan"), "blur": sb, "noise": sn, "illum": A})
            imgs.append(torch.from_numpy(np.stack(out)).unsqueeze(1))
    for c in (range(a.copies) if not a.tone else []):
        B = len(X)
        r = lambda: torch.rand(B, device="cuda", generator=g)
        con = a.c_lo + (a.c_hi - a.c_lo) * r()
        sb = a.b_lo + (a.b_hi - a.b_lo) * r()
        sn = (a.n_lo + (a.n_hi - a.n_lo) * r()) / 255
        m = X.mean((1, 2, 3), keepdim=True)
        yb = gblur(m + con.view(B, 1, 1, 1) * (X - m), sb)
        yb = yb + torch.randn(X.shape, device="cuda", generator=g) * sn.view(B, 1, 1, 1)
        imgs.append((yb.clamp(0, 1) * 255).round().to(torch.uint8).cpu())
        rows += [{"ID": tr.ID[i], "copy": c, "contrast": float(cc), "blur": float(bb), "noise": float(nn * 255)}
                 for i, cc, bb, nn in zip(src, con.cpu(), sb.cpu(), sn.cpu())]
    Xd = torch.cat(imgs)
    t0 = time.time()
    net = UNet().cuda()
    net.load_state_dict(torch.load(ROOT / "weights/denoiser_cm7_am22_l1_i10000_boot.pt", map_location="cuda"))
    D = denoise(net, Xd).squeeze(1).numpy()
    del net
    torch.cuda.empty_cache()
    D = np.stack([correct(x, "lowpass", 48.0) for x in D])  # 저품질 모드 → 모두 σ48 조명 보정 (실제 경로와 동일)
    np.savez_compressed(d / "images.npz", noisy=Xd.squeeze(1).numpy(), denoised=D)
    print(f"[synlow {tag}] 사본 {len(D)}장 디노이즈·보정 ({time.time() - t0:.0f}s)", flush=True)
    model, mean, std = build("vit_large_patch14_dinov2.lvd142m", 448)
    E = embed(model, mean, std, torch.from_numpy(D).unsqueeze(1), 448, 4, [9, 11, 13, 15, 17, 19, 21], 8, "hetero3")
    np.savez_compressed(d / "emb.npz", train=E.astype(np.float32))
    del model
    torch.cuda.empty_cache()
    with ProcessPoolExecutor(max_workers=6) as ex:
        ph = list(ex.map(phase_one, list(D), chunksize=16))
    pd.DataFrame(ph).to_csv(d / "phase.csv", index=False)
    pd.DataFrame(rows).to_csv(d / "meta.csv", index=False)
    print(f"[synlow {tag}] DINO {E.shape} · 상분율 특징 완료 ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
