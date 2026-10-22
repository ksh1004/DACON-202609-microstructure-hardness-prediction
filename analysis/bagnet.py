"""작은 수용 영역의 국소 특징 CNN을 처음부터 학습 (사전학습 없음) — 지금 블렌드와 귀납 편향이 전혀 다른 구성원이 되는지 시험.
경도 = 화소별 국소 응답 φ(주변 r화소)의 [평균 | 평균·표준편차] 풀링 → 선형. 입력은 디노이즈+조명보정(σ48) 이미지 원 해상도(256).
무작위 크롭 + D4 + 밝기·대비 흔들기로 학습, 전체 이미지로 평가, 가중치 EMA. 5폴드 OOF를 analysis/model_oof_bagnet_<tag>.npy 로 저장.
사용: python bagnet.py [--width 48 --depth 7 --pool meanstd --epochs 200 --crop 192]"""
import argparse, sys, time, copy
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("--width", type=int, default=48); ap.add_argument("--depth", type=int, default=7); ap.add_argument("--pool", choices=["mean", "meanstd"], default="meanstd")
ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--crop", type=int, default=192); ap.add_argument("--lr", type=float, default=3e-3); ap.add_argument("--wd", type=float, default=1e-2)
ap.add_argument("--stride", type=int, default=2, help="첫 합성곱 stride (2면 수용 영역이 2배)"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="")
a = ap.parse_args(); sys.argv = sys.argv[:1]
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from train_cnn import d4
from blend_eval import *
torch.manual_seed(a.seed); np.random.seed(a.seed)
X = torch.from_numpy(np.load(ROOT / "outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz")["train"]).unsqueeze(1).float().div(255).cuda()
X = (X - 0.57) / 0.08; ym, ys = y.mean(), y.std(); Y = torch.tensor((y - ym) / ys, dtype=torch.float32).cuda()
class Net(nn.Module):
    def __init__(s, w, depth, pool, stride):
        super().__init__(); L = [nn.Conv2d(1, w, 3, stride, 1), nn.BatchNorm2d(w), nn.ReLU(True)]
        for i in range(depth - 1): L += [nn.Conv2d(w, w, 3, 1, 1), nn.BatchNorm2d(w), nn.ReLU(True)]
        s.f = nn.Sequential(*L); s.pool = pool; s.head = nn.Linear(w * (2 if pool == "meanstd" else 1), 1)
    def forward(s, x):
        z = s.f(x).flatten(2); p = z.mean(2)
        if s.pool == "meanstd": p = torch.cat([p, (z.var(2) + 1e-5).sqrt()], 1)
        return s.head(p).squeeze(1)
def run_fold(k):
    tr = torch.tensor(np.where(fold != k)[0]).cuda(); va = torch.tensor(np.where(fold == k)[0]).cuda()
    m = Net(a.width, a.depth, a.pool, a.stride).cuda(); ema = copy.deepcopy(m)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=a.wd); steps = a.epochs * (len(tr) // 32); sched = torch.optim.lr_scheduler.OneCycleLR(opt, a.lr, total_steps=steps, pct_start=0.1)
    g = torch.Generator(device="cuda").manual_seed(a.seed * 10 + k); it = 0
    for ep in range(a.epochs):
        m.train(); perm = tr[torch.randperm(len(tr), generator=g, device="cuda")]
        for b in range(0, len(perm) - 31, 32):
            idx = perm[b:b + 32]; xb = d4(X[idx], int(torch.randint(0, 8, (1,)).item()))
            i0, j0 = (int(torch.randint(0, 256 - a.crop + 1, (1,)).item()) for _ in range(2)); xb = xb[..., i0:i0 + a.crop, j0:j0 + a.crop]
            xb = xb * (1 + 0.1 * (torch.rand(len(idx), 1, 1, 1, device="cuda") - .5)) + 0.3 * (torch.rand(len(idx), 1, 1, 1, device="cuda") - .5)     # 대비 ±5%, 밝기 흔들기
            loss = F.mse_loss(m(xb), Y[idx]); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step(); it += 1
            with torch.no_grad():
                dcy = min(0.995, (1 + it) / (10 + it))
                for pe, pm in zip(ema.parameters(), m.parameters()): pe.mul_(dcy).add_(pm.detach(), alpha=1 - dcy)
                for be, bm in zip(ema.buffers(), m.buffers()): be.copy_(bm)
    ema.eval()
    with torch.no_grad(): p = sum(ema(d4(X[va], v)) for v in range(8)) / 8
    return va.cpu().numpy(), p.cpu().numpy() * ys + ym
t0 = time.time(); o = np.zeros(len(y))
for k in range(5):
    va, p = run_fold(k); o[va] = p; print(f"  폴드 {k}: RMSE {np.sqrt(((p - y[va]) ** 2).mean()):.3f} ({time.time() - t0:.0f}s)", flush=True)
rf = 1 + 2 * (a.depth - 1) * a.stride + 2
tag = f"w{a.width}d{a.depth}s{a.stride}_{a.pool}_c{a.crop}_e{a.epochs}{a.tag}"; np.save(ROOT / f"analysis/model_oof_bagnet_{tag}.npy", o)
lq = sig >= 10; rm = lambda p, m=slice(None): np.sqrt(((p - y)[m] ** 2).mean())
print(f"[bagnet {tag}] 수용 영역 약 {rf}화소 | OOF {rm(o):.3f} (고품질 {rm(o, ~lq):.3f}, 저품질 {rm(o, lq):.3f})")
