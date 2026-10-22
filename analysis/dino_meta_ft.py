"""DINOv2-L의 정규화 층(LayerNorm 가중치·편향)만 '메타 Ridge' 손실로 조정 — 고정 특징 + Ridge 구조는 그대로 두고 특징 추출기를 이 데이터에 맞춘다.
보통의 미세조정은 Ridge가 train을 보간해 손실이 0이 되어 배울 것이 없거나(LP-FT) 과적합했다. 여기서는 매 스텝 폴드 train 이미지를 내부 5분할해
'일부로 맞춘 Ridge가 나머지를 얼마나 맞히는가'를 손실로 쓴다(analysis/meta_ft.py의 CNN 판과 같은 방식).
특징: σ48 U-Net 경로 이미지, 448 입력, 블록 11·13·15·17 출력의 phase2m 풀링(불균일성 6 + 상 조건부 4 + 기여량 1 블록), 스텝마다 무작위 D4 보기 1개.
평가: 조정된 모델로 8보기 평균 특징 → 폴드 train으로 Ridge(고정 λ) → 폴드 밖 오차 (스텝 0 = 고정 특징).
사용: python dino_meta_ft.py --folds 0 1 --steps 15 --lr 3e-4"""
import argparse, sys, time
from pathlib import Path
ap = argparse.ArgumentParser(); ap.add_argument("--folds", type=int, nargs="+", default=[0]); ap.add_argument("--steps", type=int, default=15); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--eval-every", type=int, default=5); ap.add_argument("--lam", type=float, default=0.01); ap.add_argument("--chunk", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--src", default="unet_cm7_am22_l1_i10000_boot_fflowpass48_low"); ap.add_argument("--tag", default="")
a = ap.parse_args(); sys.argv = sys.argv[:1]
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, torch, torch.nn.functional as F, timm
from probe_models import dark_map
from train_cnn import d4
from blend_eval import *
torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed); dev = "cuda"; LAY = [11, 13, 15, 17]; SIZE = 448
m = timm.create_model("vit_large_patch14_dinov2.lvd142m", pretrained=True, num_classes=0, img_size=SIZE).cuda().eval()
for p in m.parameters(): p.requires_grad_(False)
tun = [p for n, p in m.named_parameters() if ("norm" in n) and n.startswith("blocks.") and int(n.split(".")[1]) <= max(LAY)]
for p in tun: p.requires_grad_(True)
init_state = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad or True}
cfg = m.pretrained_cfg; mean = torch.tensor(cfg["mean"], device=dev).view(1, 3, 1, 1); std = torch.tensor(cfg["std"], device=dev).view(1, 3, 1, 1)
X = torch.from_numpy(np.load(ROOT / f"outputs/denoised/{a.src}.npz")["train"]).unsqueeze(1).cuda(); N = len(X)
print(f"조정 파라미터 {sum(p.numel() for p in tun) / 1e3:.0f}k ({len(tun)}개 텐서)", flush=True)
def sstd(x, dim): return (x.var(dim) + 1e-8).sqrt()
def pool(z, dm):
    """z (B,C,h,w) float, dm (B,1,h,w) → phase2m 풀링 (B, 11C)"""
    zf = z.flatten(2); fs = [zf.mean(2), sstd(zf, 2)]
    for g in (2, 4, 8): fs.append(sstd(F.adaptive_avg_pool2d(z, g).flatten(2), 2))
    m1, m2 = F.adaptive_avg_pool2d(z, 4), F.adaptive_avg_pool2d(z * z, 4); fs.append(sstd(((m2 - m1 * m1).clamp_min(0) + 1e-8).sqrt().flatten(2), 2))
    w = dm.float().flatten(1).unsqueeze(1); ph = []
    for ww in (w, 1 - w):
        s = ww.sum(2, keepdim=True).clamp_min(1e-3); mu = (zf * ww).sum(2, keepdim=True) / s; ph.append(mu.squeeze(2)); ph.append(((((zf - mu) ** 2 * ww).sum(2, keepdim=True) / s) + 1e-8).sqrt().squeeze(2))
    fs += [ph[0], ph[2], ph[1], ph[3], (zf * w).mean(2)]; return torch.cat(fs, 1)
def fwd(idx, k, grad):
    xb = X[idx].to(dev).float().div_(255); xk = d4(xb, k); xi = (F.interpolate(xk, size=(SIZE, SIZE), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1) - mean) / std
    with torch.set_grad_enabled(grad), torch.autocast("cuda", dtype=torch.bfloat16):
        fs = m.forward_intermediates(xi, indices=LAY, intermediates_only=True)
    dm = dark_map(xk, fs[0].shape[-1])
    return torch.cat([pool(f.float(), dm) for f in fs], 1)                      # (B, 4·11·1024)
GRP = [(list(range(0, 6)), 1.0), (list(range(6, 10)), 0.25), ([10], 0.05)]; C = 1024
def design(Fm, t):
    Fm = Fm.view(len(Fm), 4, 11, C); K = 0
    for blocks, g in GRP:
        B = Fm[:, :, blocks].reshape(len(Fm), -1); mu, sd = B[t].mean(0).detach(), B[t].std(0).detach(); keep = sd > 1e-6
        Z = ((B[:, keep] - mu[keep]) / sd[keep]).clamp(-6, 6); K = K + g * (Z @ Z.T) / keep.sum()
    return K
def ridge_pred(K, yv, s, q, lam):
    Kss, Kqs = K[s][:, s], K[q][:, s]; cm = Kss.mean(0, keepdim=True); mm = Kss.mean(); Kc = Kss - cm - cm.T + mm; Kq = Kqs - cm - Kqs.mean(1, keepdim=True) + mm
    sc = Kc.diagonal().mean(); al = torch.linalg.solve(Kc + lam * sc * torch.eye(len(s), device=K.device, dtype=K.dtype), yv[s] - yv[s].mean()); return Kq @ al + yv[s].mean()
yt = torch.tensor(y, device=dev, dtype=torch.float64)
@torch.no_grad()
def feats8():
    out = torch.zeros(N, 44 * C, device=dev)
    for k in range(8):
        for i in range(0, N, 16): out[i:i + 16] += fwd(slice(i, i + 16), k, False) / 8
    return out.double()
OOF = {}; t0 = time.time()
for FOLD in a.folds:
    m.load_state_dict({**m.state_dict(), **{n: v for n, v in init_state.items()}}, strict=False)
    tr = np.where(fold != FOLD)[0]; va = np.where(fold == FOLD)[0]; TR, VA = torch.tensor(tr, device=dev), torch.tensor(va, device=dev); n = len(tr)
    opt = torch.optim.AdamW(tun, lr=a.lr, weight_decay=0.0)
    def evaluate(step):
        Fm = feats8(); K = design(Fm, TR); p = ridge_pred(K, yt, TR, VA, a.lam); r = ((p - yt[VA]) ** 2).mean().sqrt().item()
        OOF.setdefault(step, np.full(N, np.nan))[va] = p.cpu().numpy(); print(f"  폴드 {FOLD} 스텝 {step:3d}: 폴드 밖 RMSE {r:.3f} ({time.time() - t0:.0f}s)", flush=True)
    evaluate(0)
    for step in range(1, a.steps + 1):
        k = int(rng.integers(8))
        with torch.no_grad():
            Fm = torch.cat([fwd(TR[i:i + 16], k, False) for i in range(0, n, 16)]).double()
        Fm.requires_grad_(True); K = design(Fm, torch.arange(n, device=dev)); perm = torch.tensor(rng.permutation(n), device=dev); loss = 0; allt = torch.arange(n, device=dev)
        for j in range(5):
            q = perm[j::5]; msk = torch.ones(n, dtype=torch.bool, device=dev); msk[q] = False
            loss = loss + ((ridge_pred(K, yt[TR], allt[msk], q, a.lam) - yt[TR][q]) ** 2).sum() / n
        g, = torch.autograd.grad(loss, Fm); g = g.float(); opt.zero_grad(set_to_none=True)
        for i in range(0, n, a.chunk): (fwd(TR[i:i + a.chunk], k, True) * g[i:i + a.chunk]).sum().backward()
        opt.step(); print(f"    step {step} 내부 CV {loss.item() ** 0.5:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if step % a.eval_every == 0: evaluate(step)
msk = ~np.isnan(OOF[0]); b = OOF[0][msk]
print(f"— 요약 ({msk.sum()}장) 기준 {np.sqrt(((b - y[msk]) ** 2).mean()):.3f}")
for st in sorted(OOF):
    if st == 0: continue
    o = OOF[st][msk]; d = (o - y[msk]) ** 2 - (b - y[msk]) ** 2; print(f"  스텝 {st}: {np.sqrt(((o - y[msk]) ** 2).mean()):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f})")
np.savez(ROOT / f"analysis/dinometa_f{'-'.join(map(str, a.folds))}_lr{a.lr:g}{a.tag}.npz", **{f"s{k}": v for k, v in OOF.items()})
