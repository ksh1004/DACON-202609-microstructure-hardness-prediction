"""이미 특징이 있는 다른 디노이저 경로들(8보기, σ48)을 DINOv2 결합 Ridge의 세 번째 경로로 (λ 0.01, 3계열 블렌드)."""
import sys, io, contextlib, os; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()): import lam_sweep as L
from lam_sweep import prep, fit, rm, rw
from blend_eval import *
from twopath import kmat
TP = dict(np.load("analysis/twopath_cache.npz")); f = lambda K: fit(prep(K), 0.01)[0]
kv = lambda stem: kmat(np.mean([np.load(f"outputs/probe2_views/{stem}/train_s0_k{k}.npy").astype(np.float32) for k in range(8)], 0), 4, 1024)
oT, oF, ou = f(TP["T_u"]), f(TP["F_u"]), f(TP["D_u"]); o2 = f(TP["D_u"] + TP["D_d"]); b2 = .45 * o2 + .25 * oT + .30 * oF; lq = sig >= 10
print(f"기준: U-Net {rm(ou):.3f}, U-Net+DRUNet {rm(o2):.3f} → 블렌드 {rm(b2):.3f} pw13 {rw(b2, w13):.3f}")
res = {}
for st in ("unet_real_boot_i10000_fflowpass48_low", "unet_kb_boot_i10000_fflowpass48_low", "unet_naf_boot_i10000_fflowpass48_low", "unet_boot2_i10000_fflowpass48_low",
           "unet_cm7_am22_l1_i10000_boot_c256_fflowpass48_low", "unet_cm7_am22_l1_i10000_boot_nb_fflowpass48_low", "eqn12_k0_fflowpass48_low", "eqn12_k1_fflowpass48_low", "unet_cm7_am22_l1_i10000_boot_fflowpass48_low_gn"):
    K = kv(st); ox = f(K); o3 = f(TP["D_u"] + TP["D_d"] + K); oxd = f(TP["D_d"] + K); b3 = .45 * o3 + .25 * oT + .30 * oF; d = (b3 - y) ** 2 - (b2 - y) ** 2; res[st] = K
    print(f"  {st[:44]:44s} 단독 {rm(ox):.3f} (고 {rm(ox, ~lq):.2f} 저 {rm(ox, lq):.2f}) 잔차상관 U {np.corrcoef(ox - y, ou - y)[0, 1]:.3f} | DRUNet+이것 {rm(oxd):.3f} | 세 경로 {rm(o3):.3f} → 블렌드 {rm(b3):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5, pw13 {rw(b3, w13) - rw(b2, w13):+.3f})", flush=True)
