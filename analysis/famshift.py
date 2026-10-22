"""DINOv2를 ft1b 경로로 둔 상태에서 TIPSv2·CNN의 경로(U-Net/DRUNet/평균)를 바꿨을 때 3계열 블렌드 (σ48, 8보기, λ 0.01)."""
import sys, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
    from twopath import kmat
from lam_sweep import prep, fit, rm, rw
from blend_eval import *
TP = dict(np.load("analysis/twopath_cache.npz")); f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
kv = lambda stem: kmat(np.mean([np.load(f"outputs/probe2_views/{stem}/train_s0_k{k}.npy").astype(np.float32) for k in range(8)], 0), 4, 1024)
oFb = f(kv("drunetft_ft1b_fflowpass48_low")); oDj = f(TP["D_u"] + TP["D_d"] + kv("unet_drt10b_i10000_fflowpass48_low"))
O = {n: f(TP[n]) for n in ("T_u", "T_d", "F_u", "F_d")}
for n in O: print(f"  {n} {rm(O[n]):.3f} (고 {rm(O[n], ~lq):.2f} 저 {rm(O[n], lq):.2f})")
O["T_a"], O["F_a"] = (O["T_u"] + O["T_d"]) / 2, (O["F_u"] + O["F_d"]) / 2
base = .45 * oDj + .25 * O["T_u"] + .30 * O["F_u"]
for dn, oD in (("U+D+X", oDj), ("Fb", oFb)):
    for t in ("T_u", "T_d", "T_a"):
        for c in ("F_u", "F_d", "F_a"):
            b = .45 * oD + .25 * O[t] + .30 * O[c]; d = (b - y) ** 2 - (base - y) ** 2
            print(f"DINO {dn:6s} TIPS {t} CNN {c}: 블렌드 {rm(b):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5, pw13 {rw(b, w13):.3f})")
