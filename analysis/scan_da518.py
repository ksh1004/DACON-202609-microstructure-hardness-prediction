"""Depth-Anything-V2-L F 경로 σ48·8보기 층 9~17: 입력 448 vs 518, 그리고 두 해상도 예측 평균."""
import sys, io, contextlib, glob; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
from lam_sweep import prep, fit, rm
from kmat_util import kmat
from blend_eval import *
f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10; D = "outputs/probe2_views/drunetft_ft1b_fflowpass48_low"
def member(pattern, lays, pick=(9, 11, 13, 15, 17)):
    fs = sorted(glob.glob(f"{D}/{pattern}")); assert len(fs) == 8, (pattern, len(fs))
    X = np.mean([np.load(p).astype(np.float32) for p in fs], 0).reshape(500, len(lays), 11, 1024)
    return f(kmat(X[:, [lays.index(l) for l in pick]].reshape(500, -1), len(pick), 1024))
a = member("train_s0_k?_mdepth-anything__Depth-Anything-V2-Large-hf_L3to21.npy", [3, 5, 7, 9, 11, 13, 15, 17, 19, 21])
b = member("train_s0_k?_r518_mdepth-anything__Depth-Anything-V2-Large-hf_L9to17.npy", [9, 11, 13, 15, 17])
d = member("train_s0_k?_L3to17.npy", [3, 5, 7, 9, 11, 13, 15, 17])
for n, o in (("DA 448", a), ("DA 518", b), ("DA 448+518 평균", (a + b) / 2), ("DINOv2+DA448", (d + a) / 2), ("DINOv2+DA518", (d + b) / 2), ("DINOv2+DA448+DA518", (d + a + b) / 3)):
    print(f"  {n:20s} {rm(o):.3f} (고 {rm(o, ~lq):.2f} 저 {rm(o, lq):.2f})")
print(f"  잔차상관 DA448-DA518 {np.corrcoef(a - y, b - y)[0, 1]:.3f}")
