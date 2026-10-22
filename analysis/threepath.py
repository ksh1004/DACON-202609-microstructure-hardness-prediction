"""세 번째 이미지 경로를 DINOv2 결합 Ridge에 더하면? (σ48, train 8보기, λ 0.01, 3계열 블렌드 D .45 T .25 F .30)"""
import sys, io, contextlib, os; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()): import lam_sweep as L
from lam_sweep import *
from twopath import kmat
TP = dict(np.load("analysis/twopath_cache.npz")); lam = 0.01; f = lambda K: fit(prep(K), lam)[0]
kv = lambda stem: kmat(np.mean([np.load(f"outputs/probe2_views/{stem}/train_s0_k{k}.npy").astype(np.float32) for k in range(8)], 0), 4, 1024)
X = {nm: kv(st) for nm, st in (("DRUNet×0.5", "drunet_s0.5_fflowpass48_low"), ("DRUNet×1.6", "drunet_s1.6_fflowpass48_low"), ("가우시안 σ1", "rawg1_fflowpass48_low")) if os.path.exists(f"outputs/probe2_views/{st}/train_s0_k7.npy")}
oT, oF = f(TP["T_u"]), f(TP["F_u"]); lq = sig >= 10
o2 = f(TP["D_u"] + TP["D_d"]); b2 = .45 * o2 + .25 * oT + .30 * oF; ou = f(TP["D_u"])
print(f"D: U-Net {rm(ou):.3f} | U-Net+DRUNet {rm(o2):.3f} → 블렌드 {rm(b2):.3f} pw13 {rw(b2, w13):.3f}")
for nm, K in X.items():
    ox = f(K); o3 = f(TP["D_u"] + TP["D_d"] + K); ox2 = f(TP["D_u"] + K); b3 = .45 * o3 + .25 * oT + .30 * oF; d = (b3 - y) ** 2 - (b2 - y) ** 2
    print(f"  {nm:12s} 단독 {rm(ox):.3f} (고 {rm(ox, ~lq):.2f} 저 {rm(ox, lq):.2f}) | 잔차상관 U {np.corrcoef(ox - y, ou - y)[0, 1]:.3f} | U-Net+이것 {rm(ox2):.3f} | 세 경로 {rm(o3):.3f} → 블렌드 {rm(b3):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5, pw13 {rw(b3, w13) - rw(b2, w13):+.3f})")
if len(X) >= 2:
    o4 = f(TP["D_u"] + TP["D_d"] + sum(X.values())); b4 = .45 * o4 + .25 * oT + .30 * oF; print(f"  전부 결합 ({2 + len(X)}경로) {rm(o4):.3f} → 블렌드 {rm(b4):.3f} pw13 {rw(b4, w13):.3f}")
