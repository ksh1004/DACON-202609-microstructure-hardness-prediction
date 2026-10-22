"""증강 행으로 학습: 이미지당 D4 8방향 특징을 평균한 한 행 대신, 방향별 8행(같은 라벨)으로 Ridge 학습 → 예측은 8방향 예측 평균.
U-Net 경로·DRUNet 경로 DINOv2-L(σ48, 층 11~17), λ 고정. 커널은 표준화를 train 평균행 기준으로 하고 행 단위로 계산."""
import sys, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()): import lam_sweep as L
from lam_sweep import prep, fit, rm, rw
from blend_eval import *
import torch
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05)); TP = dict(np.load("analysis/twopath_cache.npz"))
def load_views(stem): return np.stack([np.load(f"outputs/probe2_views/{stem}/train_s0_k{k}.npy").astype(np.float32) for k in range(8)])   # (8, 500, F)
VU, VD = load_views("unet_cm7_am22_l1_i10000_boot_fflowpass48_low"), load_views("drunet_fflowpass48_low")
def zblocks(V, t):
    """V (8,500,F) → 표준화·묶음 배율 적용한 (8,500,F') — 통계는 train 이미지의 평균 특징 기준"""
    A = V.reshape(8, 500, 4, 11, 1024); out = []
    for blocks, g in GR:
        Z = A[:, :, :, list(blocks)].reshape(8, 500, -1); M = Z.mean(0)[t]; mu, sd = M.mean(0), M.std(0); keep = sd > 1e-6
        Z = np.clip((Z[:, :, keep] - mu[keep]) / sd[keep], -6, 6) * np.sqrt(g / keep.sum()); out.append(Z)
    return np.concatenate(out, 2)
def stacked_oof(Vs, lam):
    o = np.zeros(500)
    for k in range(5):
        t, v = fold != k, fold == k
        Z = torch.tensor(np.concatenate([zblocks(V, t) for V in Vs], 2), device="cuda", dtype=torch.float32)      # (8,500,F)
        Zt = Z[:, t].reshape(-1, Z.shape[2]); yt = torch.tensor(np.tile(y[t], 8), device="cuda", dtype=torch.float32); mu = Zt.mean(0); Zt = Zt - mu
        K = Zt @ Zt.T; sc = K.diagonal().mean(); a_ = torch.linalg.solve((K + lam * sc * torch.eye(len(K), device="cuda")).double(), (yt - yt.mean()).double()).float()
        P = (Z[:, v] - mu) @ (Zt.T @ a_) + yt.mean(); o[v] = P.mean(0).cpu().numpy()
        del Z, Zt, K; torch.cuda.empty_cache()
    return o
f = lambda K, l: fit(prep(K), l)[0]; oT, oF = f(TP["T_u"], 0.01), f(TP["F_u"], 0.01)
for lam in (0.01, 0.003):
    for nm, Vs, Kav in (("U-Net 경로", [VU], TP["D_u"]), ("두 경로 결합", [VU, VD], TP["D_u"] + TP["D_d"])):
        oa = f(Kav, lam); os_ = stacked_oof(Vs, lam); b_a, b_s = .45 * oa + .25 * oT + .30 * oF, .45 * os_ + .25 * oT + .30 * oF; d = (b_s - y) ** 2 - (b_a - y) ** 2
        print(f"λ {lam} {nm}: 평균 특징 1행 {rm(oa):.3f} | 방향별 8행 {rm(os_):.3f} | 둘 평균 {rm((oa + os_) / 2):.3f} | 블렌드 {rm(b_a):.3f} → {rm(b_s):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == q].mean() < 0 for q in range(5))}/5)", flush=True)
