"""Depth-Anything-V2-L σ48·8보기(층 9~17): F·DRT(X)·U 경로 단독, 경로 예측 평균, DINOv2 F와의 결합."""
import sys, io, contextlib, glob; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
from lam_sweep import prep, fit, rm
from kmat_util import kmat
from blend_eval import *
f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10
def member(st, pattern, lays, pick=(9, 11, 13, 15, 17)):
    fs = sorted(glob.glob(f"outputs/probe2_views/{st}/{pattern}")); assert len(fs) == 8, (st, pattern, len(fs))
    X = np.mean([np.load(p).astype(np.float32) for p in fs], 0).reshape(500, len(lays), 11, 1024)
    return f(kmat(X[:, [lays.index(l) for l in pick]].reshape(500, -1), len(pick), 1024))
FS = "drunetft_ft1b_fflowpass48_low"
O = {"DINO_F": member(FS, "train_s0_k?_L3to17.npy", [3, 5, 7, 9, 11, 13, 15, 17]),
     "DA_F": member(FS, "train_s0_k?_mdepth-anything__Depth-Anything-V2-Large-hf_L3to21.npy", [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]),
     "DA_X": member("unet_drt10b_i10000_fflowpass48_low", "train_s0_k?_mdepth-anything__Depth-Anything-V2-Large-hf_L9to17.npy", [9, 11, 13, 15, 17]),
     "DA_U": member("unet_cm7_am22_l1_i10000_boot_fflowpass48_low", "train_s0_k?_mdepth-anything__Depth-Anything-V2-Large-hf_L9to17.npy", [9, 11, 13, 15, 17])}
for n, o in O.items(): print(f"  {n:7s} 단독 {rm(o):.3f} (고 {rm(o, ~lq):.2f} 저 {rm(o, lq):.2f}) | DA_F와 잔차상관 {np.corrcoef(o - y, O['DA_F'] - y)[0, 1]:.3f}")
m = lambda *ks: np.mean([O[k] for k in ks], 0)
for combo in (("DINO_F", "DA_F"), ("DA_F", "DA_X"), ("DA_F", "DA_X", "DA_U"), ("DINO_F", "DA_F", "DA_X"), ("DINO_F", "DA_F", "DA_X", "DA_U")):
    print(f"  평균 {'+'.join(combo):26s} {rm(m(*combo)):.3f}")
print(f"  DINO_F 0.5 + (DA_F, DA_X 평균) 0.5 = {rm(.5 * O['DINO_F'] + .25 * O['DA_F'] + .25 * O['DA_X']):.3f}")
