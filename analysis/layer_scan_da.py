"""Depth-Anything-V2-L(DINOv2 백본을 깊이 추정으로 재학습) F 경로 σ48·8보기 특징: 층 부분집합 단독 성능 + 현재 F 경로 DINOv2(층 9~17)와의 상보성."""
import sys, io, contextlib, glob; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
from lam_sweep import prep, fit, rm
from kmat_util import kmat
from blend_eval import *
st = "drunetft_ft1b_fflowpass48_low"; f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10
LA = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]
fs = sorted(glob.glob(f"outputs/probe2_views/{st}/train_s0_k*_mdepth*_L3to21.npy")); assert len(fs) == 8, fs
X = np.mean([np.load(p).astype(np.float32) for p in fs], 0).reshape(500, len(LA), 11, 1024)
LD = [3, 5, 7, 9, 11, 13, 15, 17]
XD = np.mean([np.load(f"outputs/probe2_views/{st}/train_s0_k{k}_L3to17.npy").astype(np.float32) for k in range(8)], 0).reshape(500, len(LD), 11, 1024)
oD = f(kmat(XD[:, [LD.index(l) for l in (9, 11, 13, 15, 17)]].reshape(500, -1), 5, 1024))
print(f"  기준: DINOv2 F 경로 층 9~17 단독 {rm(oD):.3f} (고 {rm(oD, ~lq):.2f} 저 {rm(oD, lq):.2f})")
for sub in ([9, 11, 13, 15, 17], [11, 13, 15, 17], [5, 7, 9, 11, 13], [13, 15, 17, 19, 21], [7, 9, 11, 13, 15, 17, 19], [3, 5, 7, 9]):
    o = f(kmat(X[:, [LA.index(l) for l in sub]].reshape(500, -1), len(sub), 1024))
    rc = np.corrcoef(o - y, oD - y)[0, 1]
    print(f"  DA 층 {'-'.join(map(str, sub)):22s} 단독 {rm(o):.3f} (고 {rm(o, ~lq):.2f} 저 {rm(o, lq):.2f}) | 잔차상관 {rc:.3f} | DINO와 평균 0.5 {rm(.5 * o + .5 * oD):.3f} · 0.3 {rm(.3 * o + .7 * oD):.3f}", flush=True)
