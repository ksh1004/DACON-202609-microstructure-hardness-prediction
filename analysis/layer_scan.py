"""F 경로(ft1b) σ48·8보기 DINOv2 특징의 층 부분집합 비교 (층 3~17 추출본, λ 0.01 Ridge 단독 + 3계열 블렌드)."""
import sys, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
from lam_sweep import prep, fit, rm
from kmat_util import kmat
from blend_eval import *
LAY = [3, 5, 7, 9, 11, 13, 15, 17]; st = "drunetft_ft1b_fflowpass48_low"
X = np.mean([np.load(f"outputs/probe2_views/{st}/train_s0_k{k}_L3to17.npy").astype(np.float32) for k in range(8)], 0).reshape(500, len(LAY), 11, 1024)
TP = dict(np.load("analysis/twopath_cache.npz")); f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10
oT, oF = f(TP["T_u"]), f(TP["F_u"]); bl = lambda oD: .45 * oD + .25 * oT + .30 * oF
for sub in ([11, 13, 15, 17], [9, 11, 13, 15, 17], [7, 9, 11, 13, 15, 17], [5, 7, 9, 11, 13, 15, 17], [3, 5, 7, 9, 11, 13, 15, 17], [7, 9, 11, 13, 15], [5, 9, 13, 17], [7, 11, 15]):
    ix = [LAY.index(l) for l in sub]; o = f(kmat(X[:, ix].reshape(500, -1), len(ix), 1024))
    print(f"  층 {'-'.join(map(str, sub)):22s} 단독 {rm(o):.3f} (고 {rm(o, ~lq):.2f} 저 {rm(o, lq):.2f}) → 블렌드 {rm(bl(o)):.3f}", flush=True)
