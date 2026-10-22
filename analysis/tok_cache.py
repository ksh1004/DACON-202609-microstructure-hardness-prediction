"""토큰 수준 실험용 캐시: 고정 모델의 토큰(특징맵 화소)을 층별 PCA(k차원, train 토큰 표본으로 적합·라벨 미사용)로 줄여 D4 8보기 전부 저장.
풀링 방식(공분산·토큰 군집 히스토그램 등)을 GPU 재추출 없이 바꿔 가며 시험하기 위한 것. σ48 보정 train 이미지.
산출물: outputs/tok_cache/<tag>.npz  (tok: [layer][view] (500, T, k) float16, dm: [view] (500, T) float16 상 지도, mean/comp: PCA)
사용: python tok_cache.py --model vit_large_patch14_dinov2.lvd142m --size 448 --layers 11 13 15 17 --k 64 --tag dinoL
      python tok_cache.py --model cnn:densenet121.tv_in1k@2 --size 512 --k 48 --tag dn121"""
import argparse, sys, time
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--size", type=int, default=448); ap.add_argument("--layers", type=int, nargs="+", default=[0])
ap.add_argument("--k", type=int, default=64); ap.add_argument("--tag", required=True); ap.add_argument("--bs", type=int, default=8); ap.add_argument("--grid", type=int, default=0, help="토큰 격자를 이 크기로 평균 축소(0=그대로)")
a = ap.parse_args()
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
import numpy as np, torch, torch.nn.functional as F
from probe_models import build, dark_map
from train_cnn import d4
S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
X = torch.from_numpy(np.load(ROOT / f"outputs/denoised/{S48}.npz")["train"]).unsqueeze(1); N = len(X)
m, mean, std = build(a.model, a.size); torch.manual_seed(0)
@torch.no_grad()
def fwd(i, k):
    xb = X[i:i + a.bs].cuda().float().div_(255); xk = d4(xb, k)
    xi = (F.interpolate(xk, size=(a.size, a.size), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1) - mean) / std
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fs = m.forward_intermediates(xi, indices=a.layers, intermediates_only=True)
    fs = [f.float() for f in fs]
    if a.grid: fs = [F.adaptive_avg_pool2d(f, a.grid) for f in fs]
    g = fs[0].shape[-1]; return [f.flatten(2).transpose(1, 2) for f in fs], dark_map(xk, g).flatten(1)      # (B, T, C), (B, T)
t0 = time.time(); samp = [[] for _ in a.layers]
for i in range(0, N, a.bs):                                   # 1) PCA용 토큰 표본 (보기 0, 이미지당 128개)
    fs, _ = fwd(i, 0)
    for j, f in enumerate(fs):
        idx = torch.randint(0, f.shape[1], (f.shape[0], 128), device=f.device); samp[j].append(torch.gather(f, 1, idx.unsqueeze(-1).expand(-1, -1, f.shape[2])).flatten(0, 1).cpu())
P = []
for j in range(len(a.layers)):
    Z = torch.cat(samp[j]).double(); mu = Z.mean(0); U, S, Vt = torch.linalg.svd(Z - mu, full_matrices=False)
    P.append((mu.float().cuda(), Vt[:a.k].T.float().cuda())); print(f"층 {a.layers[j]}: 토큰 {tuple(Z.shape)}, 상위 {a.k}성분 분산 비율 {(S[:a.k] ** 2).sum() / (S ** 2).sum():.3f}", flush=True)
tok = [[[] for _ in range(8)] for _ in a.layers]; dm = [[] for _ in range(8)]
for k in range(8):                                             # 2) 8보기 전부 투영해 저장
    for i in range(0, N, a.bs):
        fs, d = fwd(i, k); dm[k].append(d.half().cpu())
        for j, f in enumerate(fs): tok[j][k].append(((f - P[j][0]) @ P[j][1]).half().cpu())
    print(f"보기 {k} {time.time() - t0:.0f}s", flush=True)
out = ROOT / "outputs/tok_cache"; out.mkdir(exist_ok=True)
np.savez(out / f"{a.tag}.npz", tok=np.stack([np.stack([torch.cat(v).numpy() for v in L]) for L in tok]), dm=np.stack([torch.cat(v).numpy() for v in dm]),
         layers=np.array(a.layers), mean=np.stack([p[0].cpu().numpy() for p in P]), comp=np.stack([p[1].cpu().numpy() for p in P]))
print("DONE", a.tag, np.load(out / f"{a.tag}.npz")["tok"].shape)
