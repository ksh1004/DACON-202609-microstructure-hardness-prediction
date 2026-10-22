"""저품질 이미지용 플러그앤플레이 블러 복원(DPIR; Zhang et al., TPAMI 2021) — 사전 모델은 공개 사전학습 DRUNet(gray, MIT).
저품질 모드(원본 노이즈 σ≥10)는 가우시안 블러 σb≈1.0px(MTF로 측정)가 들어 있으므로, HQS 반복으로 블러를 되돌리면서 디노이즈한다:
  x ← argmin ||k*x − y||² + ρ_i ||x − z||²  (FFT 닫힌 해),   z ← DRUNet(x, σ_i),   σ_i: 49 → 이미지 노이즈 σ 로 로그 감소
고품질(σ<10)은 블러가 없으므로 일반 DRUNet 디노이즈만. 산출물: outputs/denoised/dpir_b<σb>.npz (train/test uint8)
사용: python dpir_deblur.py [--blur 1.0 --iters 8 --lamda 0.23]"""
import argparse, sys
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--blur", type=float, default=1.0); ap.add_argument("--iters", type=int, default=8); ap.add_argument("--lamda", type=float, default=0.23)
ap.add_argument("--sig-th", type=float, default=10.0); a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, pandas as pd, torch
from PIL import Image
from drunet_net import load as load_drunet
m = load_drunet(ROOT / "weights/pretrained_denoiser/drunet_gray.pth")
N = 256
def gauss_otf(sb):
    r = int(np.ceil(3 * sb)); xx = np.arange(-r, r + 1); g = np.exp(-xx ** 2 / (2 * sb * sb)); k2 = np.outer(g, g); k2 /= k2.sum()
    K = np.zeros((N, N)); K[:k2.shape[0], :k2.shape[1]] = k2; K = np.roll(K, (-r, -r), (0, 1)); return torch.tensor(np.fft.fft2(K), dtype=torch.complex64, device="cuda")
FB = gauss_otf(a.blur); F2B = (FB.abs() ** 2).real
@torch.no_grad()
def denoise(x, s):            # x (B,1,H,W) [0,1], s (B,) σ/255
    return m(torch.cat([x, s.view(-1, 1, 1, 1).expand(-1, 1, x.shape[-2], x.shape[-1])], 1))
@torch.no_grad()
def dpir(yb, sn):             # yb (B,1,N,N) [0,1], sn (B,) 노이즈 σ/255
    sig = torch.stack([torch.logspace(np.log10(49 / 255), float(torch.log10(s.clamp_min(1 / 255))), a.iters, device="cuda") for s in sn], 1)   # (iters, B)
    FBFy = FB.conj() * torch.fft.fft2(yb); x = yb.clone()
    for i in range(a.iters):
        rho = (a.lamda * sn ** 2 / sig[i] ** 2).view(-1, 1, 1, 1)
        x = torch.fft.ifft2((FBFy + rho * torch.fft.fft2(x)) / (F2B + rho)).real
        x = denoise(x.clamp(0, 1), sig[i])
    return x
out = {}
for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
    ids = pd.read_csv(ROOT / "data" / csv).ID; sg = pd.read_csv(ROOT / f"outputs/features/{split}_features.csv").set_index("ID").loc[ids, "q_noise_sigma"].values
    X = torch.from_numpy(np.stack([np.array(Image.open(ROOT / f"data/{split}/{i}.png").convert("L")) for i in ids])).unsqueeze(1).float().div(255); res = []
    for i in range(0, len(X), 16):
        xb = X[i:i + 16].cuda(); s = torch.tensor(sg[i:i + 16] / 255, dtype=torch.float32, device="cuda"); lq = torch.tensor(sg[i:i + 16] >= a.sig_th, device="cuda")
        r = denoise(xb, s)
        if lq.any(): r[lq] = dpir(xb[lq], s[lq])
        res.append((r.clamp(0, 1) * 255).round().byte().squeeze(1).cpu())
    out[split] = torch.cat(res).numpy(); print(split, out[split].shape, flush=True)
name = f"dpir_b{a.blur:g}"; np.savez_compressed(ROOT / f"outputs/denoised/{name}.npz", **out); print("→", name)
