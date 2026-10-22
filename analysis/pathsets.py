"""경로 집합 비교: U(부트스트랩) D(DRUNet) X(DRT drt10b) F(DRUNet 미세조정 ft1) Fb(ft1b) — 결합 Ridge (σ48, 8보기, λ 0.01), 3계열 블렌드."""
import sys, io, contextlib, os; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
    from kmat_util import kmat
from lam_sweep import prep, fit, rm, rw
from blend_eval import *
TP = dict(np.load("analysis/twopath_cache.npz")); f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
kv = lambda stem: kmat(np.mean([np.load(f"outputs/probe2_views/{stem}/train_s0_k{k}.npy").astype(np.float32) for k in range(8)], 0), 4, 1024)
K = dict(U=TP["D_u"], D=TP["D_d"])
for n, st in (("X", "unet_drt10b_i10000_fflowpass48_low"), ("F", "drunetft_ft1_fflowpass48_low"), ("Fb", "drunetft_ft1b_fflowpass48_low"), ("Fc", "drunetft_ft1c_fflowpass48_low"), ("Fs", "drunetft_ft1s1_fflowpass48_low"), ("Fsb", "drunetft_ft1s1b_fflowpass48_low"), ("S", "scunetft_sc1_fflowpass48_low"), ("Sb", "scunetft_sc1b_fflowpass48_low"), ("Ft4", "drunetft_ft1b_s4_fflowpass48_low")):
    if os.path.exists(f"outputs/probe2_views/{st}/train_s0_k7.npy"): K[n] = kv(st)
oT, oF = f(TP["T_u"]), f(TP["F_u"]); bl = lambda oD: .45 * oD + .25 * oT + .30 * oF; base = bl(f(K["U"] + K["D"] + K["X"]))
print("기준 = U+D+X (v25a의 DINOv2 구성)")
for combo in sys.argv[1:]:
    cs = combo.split("+")
    if not all(c in K for c in cs): print("  없음:", combo); continue
    o = f(sum(K[c] for c in cs)); b = bl(o); d = (b - y) ** 2 - (base - y) ** 2
    print(f"  {combo:14s} 결합 {rm(o):.3f} (고 {rm(o, ~lq):.2f} 저 {rm(o, lq):.2f}) → 블렌드 {rm(b):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5, 가운데 {d[mid].sum() / 500:+.2f}, pw13 {rw(b, w13):.3f})", flush=True)
