"""새 이미지 경로 X를 DINOv2 결합 Ridge에 넣었을 때 (σ48, 8보기, λ 0.01, 3계열 블렌드 D .45 T .25 F .30). 사용: python pathx.py <probe2_views 폴더명> ..."""
import sys, io, contextlib; sys.path.insert(0, "analysis")
args = sys.argv[1:]; sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
    from twopath import kmat
from lam_sweep import prep, fit, rm, rw
from blend_eval import *
TP = dict(np.load("analysis/twopath_cache.npz")); f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
kv = lambda stem: kmat(np.mean([np.load(f"outputs/probe2_views/{stem}/train_s0_k{k}.npy").astype(np.float32) for k in range(8)], 0), 4, 1024)
oT, oF, ou, od = f(TP["T_u"]), f(TP["F_u"]), f(TP["D_u"]), f(TP["D_d"]); o2 = f(TP["D_u"] + TP["D_d"]); bl = lambda oD: .45 * oD + .25 * oT + .30 * oF; b2 = bl(o2)
print(f"기준: U-Net {rm(ou):.3f} DRUNet {rm(od):.3f} | U+D 결합 {rm(o2):.3f} (고 {rm(o2, ~lq):.2f} 저 {rm(o2, lq):.2f}) → 블렌드 {rm(b2):.3f} pw13 {rw(b2, w13):.3f}")
for st in args:
    K = kv(st); ox = f(K); oux, odx, o3 = f(TP["D_u"] + K), f(TP["D_d"] + K), f(TP["D_u"] + TP["D_d"] + K)
    print(f"  [{st}] 단독 {rm(ox):.3f} (고 {rm(ox, ~lq):.2f} 저 {rm(ox, lq):.2f}) 잔차상관 U {np.corrcoef(ox - y, ou - y)[0, 1]:.3f} D {np.corrcoef(ox - y, od - y)[0, 1]:.3f}")
    for nm, o in (("U+X", oux), ("D+X", odx), ("U+D+X", o3)):
        b = bl(o); d = (b - y) ** 2 - (b2 - y) ** 2
        print(f"     {nm:6s} 결합 {rm(o):.3f} (고 {rm(o, ~lq):.2f} 저 {rm(o, lq):.2f}) → 블렌드 {rm(b):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5, 가운데 {d[mid].sum() / 500:+.2f}, pw13 {rw(b, w13) - rw(b2, w13):+.3f})")
