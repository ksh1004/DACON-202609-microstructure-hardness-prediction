"""F 경로(ft1b) σ48·8보기 TIPSv2-So400m 특징의 층 부분집합 비교 (층 2~14 추출본, λ 0.01 Ridge 단독 + 3계열 블렌드에서 T 자리 교체)."""
import sys, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
from lam_sweep import prep, fit, rm
from kmat_util import kmat
from blend_eval import *
LAY = [2, 4, 6, 8, 10, 12, 14]; st = "drunetft_ft1b_fflowpass48_low"
X = np.mean([np.load(f"outputs/probe2_views/{st}/train_s0_k{k}_mso400m_patch14_reg1_tipsv2-webli_L2to14.npy").astype(np.float32) for k in range(8)], 0).reshape(500, len(LAY), 11, 1152)
TP = dict(np.load("analysis/twopath_cache.npz")); f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10
oD, oF = f(TP["D_u"]), f(TP["F_u"]); bl = lambda oT: .45 * oD + .25 * oT + .30 * oF
for sub in ([8, 10, 12, 14], [6, 8, 10, 12, 14], [4, 6, 8, 10, 12, 14], [2, 4, 6, 8, 10, 12, 14], [4, 8, 12], [6, 10, 14]):
    ix = [LAY.index(l) for l in sub]; o = f(kmat(X[:, ix].reshape(500, -1), len(ix), 1152))
    print(f"  TIPS 층 {'-'.join(map(str, sub)):20s} 단독 {rm(o):.3f} (고 {rm(o, ~lq):.2f} 저 {rm(o, lq):.2f}) → 블렌드 {rm(bl(o)):.3f}", flush=True)
