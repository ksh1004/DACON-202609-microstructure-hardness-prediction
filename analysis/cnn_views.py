"""고정 CNN 구성원의 보기 잡음 점검: D4(8) × 격자 이동(원본 2px = 512 입력에서 stride-8 반 칸) 4종 = 32보기 특징을 보기별로 저장,
단일 보기 / D4 8보기(현재 구성) / 16보기 / 32보기 평균의 OOF 비교. train, σ48.
사용: python cnn_views.py <timm 모델> <단계 idx>"""
import sys, time
from pathlib import Path
MODEL, IDX = sys.argv[1], int(sys.argv[2]); sys.argv = sys.argv[:1]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, torch, torch.nn.functional as F, timm
from probe_models import pool, pool_phase, dark_map
from train_cnn import d4
from blend_eval import *
from sklearn.linear_model import RidgeCV
S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"; tag = MODEL.replace(".", "-")
out = ROOT / "outputs/probe2_views" / S48 / f"cnnviews_{tag}_i{IDX}_train.npy"
SH = [(0, 0), (2, 0), (0, 2), (2, 2)]
if not out.exists():
    m = timm.create_model(MODEL, pretrained=True, features_only=True, out_indices=(IDX,)).cuda().eval(); cfg = m.pretrained_cfg
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), device="cuda").view(1, 3, 1, 1); std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), device="cuda").view(1, 3, 1, 1)
    X = torch.from_numpy(np.load(ROOT / f"outputs/denoised/{S48}.npz")["train"]).unsqueeze(1); acc = []; t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(X), 16):
            xb = X[i:i + 16].cuda().float().div_(255); part = []
            for dy, dx in SH:
                for k in range(8):
                    xk = d4(xb, k)
                    if dy or dx: xk = F.pad(xk, (dx, 0, dy, 0), mode="reflect")[..., :xb.shape[-2], :xb.shape[-1]]
                    xi = (F.interpolate(xk, size=(512, 512), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1) - mean) / std
                    with torch.autocast("cuda", dtype=torch.bfloat16): z = m(xi)[0].float()
                    g = int(z.shape[-1]); dm = dark_map(xk, g); wd = dm.flatten(1).unsqueeze(1)
                    part.append(torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.flatten(2) * wd).mean(2)], 1).cpu())
            acc.append(torch.stack(part, 1))
    np.save(out, torch.cat(acc).numpy().astype(np.float32)); print(f"추출 {time.time() - t0:.0f}s", flush=True)
V = np.load(out)   # (500, 32, 11C): 보기 번호 = 이동 s*8 + D4 k
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def design(A, t):
    C = A.shape[1] // 11; A3 = A.reshape(len(A), 11, C); outs, ref = [], None
    for blocks, g in GR:
        B = A3[:, list(blocks)].reshape(len(A), -1).astype(np.float64); mu, sd = B[t].mean(0), B[t].std(0); keep = sd > 1e-10
        B = np.clip((B[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(B * s)
    D_ = np.hstack(outs); return D_ / np.sqrt(D_.shape[1])
def cv(vs):
    A = V[:, list(vs)].mean(1); o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k; D_ = design(A, t)
        o[v] = RidgeCV(alphas=np.logspace(-6, 3, 60)).fit(D_[t], y[t]).predict(D_[v])
    return o
rm = lambda o: np.sqrt(((o - y) ** 2).mean()); rng = np.random.default_rng(0)
print(f"== {MODEL} 단계 {IDX} ==")
s1 = [rm(cv([v])) for v in (0, 3, 5, 9, 14, 20, 27, 30)]; print(f"단일 보기 8개: 평균 {np.mean(s1):.3f} (범위 {min(s1):.3f}~{max(s1):.3f})")
r4 = [rm(cv(rng.choice(32, 4, replace=False))) for _ in range(8)]; print(f"무작위 4보기 8묶음: 평균 {np.mean(r4):.3f} ± {np.std(r4):.3f}")
o8 = cv(range(8)); print(f"D4 8보기(이동 없음, 현재 구성): {rm(o8):.3f}")
for s in (1, 2, 3): print(f"D4 8보기(이동 {SH[s]}): {rm(cv(range(8 * s, 8 * s + 8))):.3f}")
o16 = cv(list(range(8)) + list(range(24, 32))); print(f"16보기(이동 없음 + 대각 이동): {rm(o16):.3f}")
o32 = cv(range(32)); rep("32보기 전체", o32, o8)
np.save(ROOT / f"analysis/cnnviews_oof_{tag}.npy", np.stack([o8, o16, o32]))
