"""공개 사전학습 디노이저(DRUNet gray, KAIR, MIT / SCUNet gray, Apache-2.0)를 train 고품질 이미지로 미세조정 → 새 디노이즈 경로.
학습 데이터는 train 이미지만 (test·라벨 미사용). 2차 증류: --target-src 에 1차 미세조정 출력을 주고 같은 사전학습 가중치에서 다시 미세조정.
정답: --target-src의 train 이미지 중 q_noise_sigma < --clean-max (기본 DRUNet이 정리한 σ<10, 291장).
입력: 정답 + 가우시안 노이즈 σ~U(0, add_max), 노이즈 지도 = 실제 σ. 추론: 이미지별 추정 σ(q_noise_sigma), D4 평균.
산출물: outputs/denoised/drunetft_<tag>.npz, weights/pretrained_denoiser/drunetft_<tag>.pt"""
import sys, math, time, argparse
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, torch
from PIL import Image
from train_cnn import d4
from drunet_net import load

ap = argparse.ArgumentParser()
ap.add_argument("--target-src", default="outputs/denoised/drunet.npz"); ap.add_argument("--clean-max", type=float, default=10)
ap.add_argument("--iters", type=int, default=6000); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--crop", type=int, default=128)
ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--add-max", type=float, default=22); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", required=True); ap.add_argument("--arch", choices=["drunet", "scunet"], default="drunet"); ap.add_argument("--init", default=None)
a = ap.parse_args(); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); torch.backends.cudnn.benchmark = True
ids = {"train": pd.read_csv(ROOT / "data/train.csv").ID, "test": pd.read_csv(ROOT / "data/sample_submission.csv").ID}
sg = {s: pd.read_csv(ROOT / f"outputs/features/{s}_features.csv").set_index("ID").loc[ids[s], "q_noise_sigma"].values for s in ids}
ci = np.where(sg["train"] < a.clean_max)[0]; rng.shuffle(ci); nval = len(ci) // 10; vi, ti = ci[:nval], ci[nval:]
T = torch.from_numpy(np.load(ROOT / a.target_src)["train"]).unsqueeze(1)
C = T[ti].cuda().float().div_(255); V = T[vi].cuda().float().div_(255)
print(f"정답 {len(ci)}장 (학습 {len(ti)} / 검증 {len(vi)})", flush=True)
if a.arch == "scunet":
    from scunet_net import load as sload; m = sload(ROOT / (a.init or "weights/pretrained_denoiser/scunet_gray_25.pth")).train()
else:
    m = load(ROOT / (a.init or "weights/pretrained_denoiser/drunet_gray.pth")).train()
NB = a.arch == "drunet"  # 노이즈 지도 입력 여부
for p in m.parameters(): p.requires_grad_(True)
opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0)

def noisy(y, g=None):
    s = torch.rand(len(y), 1, 1, 1, device=y.device, generator=g) * a.add_max / 255
    x = ((y + torch.randn(y.shape, device=y.device, generator=g) * s).clamp(0, 1) * 255).round() / 255
    return torch.cat([x, s.expand_as(x)], 1) if NB else x

gv = torch.Generator(device="cuda").manual_seed(123); Vn = noisy(V, gv)
def val():
    m.eval()
    with torch.no_grad(): pv = torch.cat([m(Vn[i:i + 8]) for i in range(0, len(Vn), 8)])
    m.train(); return 10 * math.log10(1 / torch.mean((pv - V) ** 2).item())
print(f"  시작 검증 PSNR 입력 {10 * math.log10(1 / torch.mean((Vn[:, :1] - V) ** 2).item()):.2f} → 출력 {val():.2f}", flush=True)
t0 = time.time()
for it in range(a.iters):
    lr = a.lr * min(1.0, (it + 1) / 200) * 0.5 * (1 + math.cos(math.pi * it / a.iters))
    for g in opt.param_groups: g["lr"] = lr
    idx = torch.randint(0, len(C), (a.batch,), device="cuda"); oy, ox = np.random.randint(0, 256 - a.crop + 1, 2)
    y = d4(C[idx, :, oy:oy + a.crop, ox:ox + a.crop], int(np.random.randint(8)))
    with torch.autocast("cuda", dtype=torch.bfloat16): p = m(noisy(y)).float()
    loss = (p - y).abs().mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    if (it + 1) % 1000 == 0: print(f"  it{it + 1} loss {loss.item():.4f} 검증 PSNR {val():.2f} ({time.time() - t0:.0f}s)", flush=True)
(ROOT / "weights/pretrained_denoiser").mkdir(parents=True, exist_ok=True); torch.save(m.state_dict(), ROOT / f"weights/pretrained_denoiser/{a.arch}ft_{a.tag}.pt")
m.eval(); e = torch.arange(16.).view(1, 1, 4, 4); inv = {k: next(j for j in range(8) if torch.equal(d4(d4(e, k), j), e)) for k in range(8)}; out = {}
for split in ("train", "test"):
    X = torch.from_numpy(np.stack([np.array(Image.open(ROOT / f"data/{split}/{i}.png").convert("L")) for i in ids[split]])).unsqueeze(1).float().div(255); res = []
    with torch.no_grad():
        bs = 32 if NB else 8
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].cuda(); nm = torch.tensor(sg[split][i:i + bs] / 255, dtype=torch.float32, device="cuda").view(-1, 1, 1, 1).expand(-1, 1, *xb.shape[-2:])
            acc = sum(d4(m(torch.cat([d4(xb, k), nm], 1) if NB else d4(xb, k)), inv[k]) for k in range(8)) / 8
            res.append((acc.clamp(0, 1) * 255).round().byte().squeeze(1).cpu())
    out[split] = torch.cat(res).numpy()
np.savez_compressed(ROOT / f"outputs/denoised/{a.arch}ft_{a.tag}.npz", **out); print(f"[{a.arch}ft_{a.tag}] 저장 ({time.time() - t0:.0f}s)", flush=True)
