"""DINOv2 재학습 계열 후보(F 경로 σ48·8보기, 층 9~17) 비교: 단독, 기존 두 모델(DINOv2·DA-V2-L)과의 잔차상관, 평균 조합."""
import sys, io, contextlib, glob; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
from lam_sweep import prep, fit, rm
from kmat_util import kmat
from blend_eval import *
st = "drunetft_ft1b_fflowpass48_low"; f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10; D = f"outputs/probe2_views/{st}"
def member(pattern, lays, pick=(9, 11, 13, 15, 17)):
    fs = sorted(glob.glob(f"{D}/{pattern}")); assert len(fs) == 8, (pattern, len(fs))
    X = np.mean([np.load(p).astype(np.float32) for p in fs], 0).reshape(500, len(lays), 11, 1024)
    return f(kmat(X[:, [lays.index(l) for l in pick]].reshape(500, -1), len(pick), 1024))
O = {"DINOv2": member("train_s0_k?_L3to17.npy", [3, 5, 7, 9, 11, 13, 15, 17]),
     "DA-V2-L": member("train_s0_k?_mdepth-anything__Depth-Anything-V2-Large-hf_L3to21.npy", [3, 5, 7, 9, 11, 13, 15, 17, 19, 21])}
for nm, pat in (("DA-V1-L", "train_s0_k?_mLiheYoung__depth-anything-large-hf_L9to17.npy"),
                ("DA-V2-MetIn", "train_s0_k?_mdepth-anything__Depth-Anything-V2-Metric-Indoor-Large-hf_L9to17.npy"),
                ("DA-V2-MetOut", "train_s0_k?_mdepth-anything__Depth-Anything-V2-Metric-Outdoor-Large-hf_L9to17.npy"), ("MiDaS-DPT-L", "train_s0_k?_mIntel__dpt-large_L9to17.npy"), ("MoGe-1", "train_s0_k?_mmoge_Ruicheng__moge-vitl_L9to17.npy"), ("MoGe-2", "train_s0_k?_mmoge_Ruicheng__moge-2-vitl_L9to17.npy")):
    try: O[nm] = member(pat, [9, 11, 13, 15, 17])
    except AssertionError as e: print("  없음:", nm)
base = (O["DINOv2"] + O["DA-V2-L"]) / 2
print(f"  기준 (DINOv2 + DA-V2-L)/2 = {rm(base):.3f}")
for n, o in O.items():
    print(f"  {n:13s} 단독 {rm(o):.3f} (고 {rm(o, ~lq):.2f} 저 {rm(o, lq):.2f}) | 잔차상관 DINOv2 {np.corrcoef(o - y, O['DINOv2'] - y)[0, 1]:.3f} · DA-V2-L {np.corrcoef(o - y, O['DA-V2-L'] - y)[0, 1]:.3f}"
          + ("" if n in ("DINOv2", "DA-V2-L") else f" | 기준과 3자 평균 {rm((O['DINOv2'] + O['DA-V2-L'] + o) / 3):.3f} · DA-V2-L 대신 {rm((O['DINOv2'] + o) / 2):.3f}"), flush=True)
