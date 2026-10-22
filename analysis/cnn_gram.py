"""고정 CNN 특징맵에 채널 간 공분산(Gram) 블록 추가: 채널을 train 토큰 PCA 상위 K개로 줄인 뒤 토큰 공분산의 상삼각(K(K+1)/2)을 특징으로.
(PCA는 라벨 없이 train 이미지 토큰으로만 적합.) 기존 phase2m 블록 + Gram 묶음을 한 Ridge로. train, σ48, D4 8보기.
사용: python cnn_gram.py <timm 모델> <단계 idx> [K=64]"""
import sys, time
from pathlib import Path
MODEL, IDX = sys.argv[1], int(sys.argv[2]); K = int(sys.argv[3]) if len(sys.argv) > 3 else 64; sys.argv = sys.argv[:1]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "analysis"))
import numpy as np, torch, torch.nn.functional as F, timm
from probe_models import pool, pool_phase, dark_map
from train_cnn import d4
from blend_eval import *
from sklearn.linear_model import RidgeCV
S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"; tag = MODEL.replace(".", "-")
out = ROOT / "outputs/probe2_views" / S48 / f"cnngram_{tag}_k{K}_train.npz"
if not out.exists():
    m = timm.create_model(MODEL, pretrained=True, features_only=True, out_indices=(IDX,)).cuda().eval(); cfg = m.pretrained_cfg
    mean = torch.tensor(cfg.get("mean", (0.485, 0.456, 0.406)), device="cuda").view(1, 3, 1, 1); std = torch.tensor(cfg.get("std", (0.229, 0.224, 0.225)), device="cuda").view(1, 3, 1, 1)
    X = torch.from_numpy(np.load(ROOT / f"outputs/denoised/{S48}.npz")["train"]).unsqueeze(1)
    def fmap(xk):
        xi = (F.interpolate(xk, size=(512, 512), mode="bilinear", align_corners=False).repeat(1, 3, 1, 1) - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16): return m(xi)[0].float()
    t0 = time.time()
    with torch.no_grad():
        # 1) 채널 공분산(전체 train, 보기 0) → PCA 사영
        S1, S2, n = 0, 0, 0
        for i in range(0, len(X), 16):
            z = fmap(X[i:i + 16].cuda().float().div_(255))
            t = z.flatten(2).transpose(1, 2).reshape(-1, z.shape[1]).double(); S1 = S1 + t.sum(0); S2 = S2 + t.T @ t; n += len(t)
        mu = S1 / n; cov = S2 / n - torch.outer(mu, mu); ev, U = torch.linalg.eigh(cov); P = (U[:, -K:] / ev[-K:].clamp_min(1e-12).sqrt()).float(); mu = mu.float()   # 백색화 사영
        iu = torch.triu_indices(K, K, device="cuda")
        base, gram = [], []
        for i in range(0, len(X), 16):
            xb = X[i:i + 16].cuda().float().div_(255); pb, pg = 0, 0
            for k in range(8):
                xk = d4(xb, k); z = fmap(xk); g = int(z.shape[-1]); dm = dark_map(xk, g); wd = dm.flatten(1).unsqueeze(1)
                pb = pb + torch.cat([pool(z, "hetero3"), pool_phase(z, dm), (z.flatten(2) * wd).mean(2)], 1)
                t = (z.flatten(2).transpose(1, 2) - mu) @ P            # B,N,K (백색화 좌표)
                t = t - t.mean(1, keepdim=True); c = torch.einsum("bnk,bnl->bkl", t, t) / t.shape[1]
                pg = pg + c[:, iu[0], iu[1]]
            base.append((pb / 8).cpu()); gram.append((pg / 8).cpu())
    np.savez(out, base=torch.cat(base).numpy().astype(np.float32), gram=torch.cat(gram).numpy().astype(np.float32)); print(f"추출 {time.time() - t0:.0f}s")
Z = np.load(out); A, Gm = Z["base"], Z["gram"]
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def design(t, g_gram):
    C = A.shape[1] // 11; A3 = A.reshape(len(A), 11, C); outs, ref = [], None
    for blocks, g in GR:
        B = A3[:, list(blocks)].reshape(len(A), -1).astype(np.float64); mu, sd = B[t].mean(0), B[t].std(0); keep = sd > 1e-10
        B = np.clip((B[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(B * s)
    if g_gram > 0:
        B = Gm.astype(np.float64); mu, sd = B[t].mean(0), B[t].std(0) + 1e-12; B = np.clip((B - mu) / sd, -6, 6); outs.append(B * np.sqrt(g_gram * ref / B.shape[1]))
    D_ = np.hstack(outs); return D_ / np.sqrt(outs[0].shape[1])
def cv(g_gram, only=False):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k; D_ = design(t, g_gram)
        if only: D_ = D_[:, -Gm.shape[1]:]
        o[v] = RidgeCV(alphas=np.logspace(-6, 3, 60)).fit(D_[t], y[t]).predict(D_[v])
    return o
L = np.load(ROOT / "analysis/view_oof.npz")["D4_8보기"]; T = np.load(ROOT / "analysis/model_oof_mso400m_patch14_reg1_tipsv2-webli.npy")
rm = lambda o: np.sqrt(((o - y) ** 2).mean()); cr = lambda a, b: np.corrcoef(a - y, b - y)[0, 1]
b0 = cv(0); print(f"== {MODEL} 단계 {IDX}, Gram K={K} ({Gm.shape[1]}차원) ==")
rep("기본 풀링", b0); print(f"   L과 잔차상관 {cr(b0, L):.3f} | L.45+T.25+이것.30 → {rm(.45 * L + .25 * T + .30 * b0):.3f}")
og = cv(1.0, only=True); rep("Gram만", og, b0)
for g in (0.1, 0.25, 0.5, 1.0):
    o = cv(g); rep(f"기본 + Gram (상대 분산 {g})", o, b0); print(f"   L과 잔차상관 {cr(o, L):.3f} | L.45+T.25+이것.30 → {rm(.45 * L + .25 * T + .30 * o):.3f}")
