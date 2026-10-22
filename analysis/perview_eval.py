"""방향별 디노이즈 특징(_pv) vs 평균 이미지 특징: 세 경로(U-Net 부트스트랩·DRUNet·DRT) 각각 바꿔 넣은 결합 Ridge (σ48, 8보기, λ 0.01)."""
import sys, io, contextlib, os; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
    from twopath import kmat
from lam_sweep import prep, fit, rm, rw
from blend_eval import *
TP = dict(np.load("analysis/twopath_cache.npz")); f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
kv = lambda stem: kmat(np.mean([np.load(f"outputs/probe2_views/{stem}/train_s0_k{k}.npy").astype(np.float32) for k in range(8)], 0), 4, 1024)
K = dict(U=TP["D_u"], D=TP["D_d"], X=kv("unet_drt10b_i10000_fflowpass48_low"))
for n, st in (("Up", "unet_boot_pv_fflowpass48_low"), ("Dp", "drunet_pv_fflowpass48_low"), ("Xp", "unet_drt10b_i10000_pv_fflowpass48_low")):
    if os.path.exists(f"outputs/probe2_views/{st}/train_s0_k7.npy"): K[n] = kv(st)
oT, oF = f(TP["T_u"]), f(TP["F_u"]); bl = lambda oD: .45 * oD + .25 * oT + .30 * oF; base = bl(f(K["U"] + K["D"] + K["X"]))
for n in ("U", "D", "X"):
    if n + "p" in K:
        a, b = f(K[n]), f(K[n + "p"]); print(f"{n}: 평균 이미지 {rm(a):.3f} (고 {rm(a, ~lq):.2f} 저 {rm(a, lq):.2f}) | 방향별 {rm(b):.3f} (고 {rm(b, ~lq):.2f} 저 {rm(b, lq):.2f}) | 둘 결합 {rm(f(K[n] + K[n + 'p'])):.3f}")
for combo in (("U", "D", "X"), ("Up", "Dp", "Xp"), ("U", "D", "Xp"), ("U", "Dp", "X"), ("Up", "D", "X"), ("U", "D", "X", "Up", "Dp", "Xp")):
    if not all(c in K for c in combo): continue
    o = f(sum(K[c] for c in combo)); b = bl(o); d = (b - y) ** 2 - (base - y) ** 2
    print(f"  {'+'.join(combo):18s} 결합 {rm(o):.3f} (고 {rm(o, ~lq):.2f} 저 {rm(o, lq):.2f}) → 블렌드 {rm(b):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5, 가운데 {d[mid].sum() / 500:+.2f}, pw13 {rw(b, w13):.3f})", flush=True)
