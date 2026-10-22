"""pathx 확장: 인자 하나 = 경로 사양. '+'로 나눈 성분은 각각 커널로 더하고(결합 Ridge), ','로 나눈 이름은 특징을 평균(디노이저 TTA).
짧은 이름(drt10b 등)은 unet_<이름>_i10000_fflowpass48_low 로 확장. 기준 U+D 결합에 더해 평가 (σ48, 8보기, λ 0.01)."""
import sys, io, contextlib; sys.path.insert(0, "analysis")
args = sys.argv[1:]; sys.argv = sys.argv[:1]
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
    from twopath import kmat
from lam_sweep import prep, fit, rm, rw
from blend_eval import *
TP = dict(np.load("analysis/twopath_cache.npz")); f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
full = lambda s: s if s.startswith(("unet_", "drunet")) else f"unet_{s}_i10000_fflowpass48_low"
cache = {}
def feat(stem):
    if stem not in cache: cache[stem] = np.mean([np.load(f"outputs/probe2_views/{full(stem)}/train_s0_k{k}.npy").astype(np.float32) for k in range(8)], 0)
    return cache[stem]
kspec = lambda comp: kmat(np.mean([feat(s) for s in comp.split(",")], 0), 4, 1024)
oT, oF, ou, od = f(TP["T_u"]), f(TP["F_u"]), f(TP["D_u"]), f(TP["D_d"]); o2 = f(TP["D_u"] + TP["D_d"]); bl = lambda oD: .45 * oD + .25 * oT + .30 * oF; b2 = bl(o2)
print(f"기준: U+D 결합 {rm(o2):.3f} (고 {rm(o2, ~lq):.2f} 저 {rm(o2, lq):.2f}) → 블렌드 {rm(b2):.3f} pw13 {rw(b2, w13):.3f}")
for spec in args:
    Ks = [kspec(c) for c in spec.split("+")]; KX = sum(Ks)
    ox = f(KX); o3 = f(TP["D_u"] + TP["D_d"] + KX); b = bl(o3); d = (b - y) ** 2 - (b2 - y) ** 2
    print(f"  [{spec}] 단독 {rm(ox):.3f} (고 {rm(ox, ~lq):.2f} 저 {rm(ox, lq):.2f}) | U+D+X 결합 {rm(o3):.3f} (고 {rm(o3, ~lq):.2f} 저 {rm(o3, lq):.2f}) → 블렌드 {rm(b):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5, 가운데 {d[mid].sum() / 500:+.2f}, pw13 {rw(b, w13) - rw(b2, w13):+.3f})", flush=True)
