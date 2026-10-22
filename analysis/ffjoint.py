"""조명보정 σ 4종(32/48/64/96)을 지금처럼 예측 평균하는 대신 한 Ridge에 결합하면? DINOv2 16보기(U-Net 경로, DRUNet 경로), λ 0.01. 전체 블렌드(v24a 가중)로 평가."""
import sys, os, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()): import lam_sweep as L
from lam_sweep import prep, fit, rm, rw
from blend_eval import *
from twopath import kmat
M = "vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2m_T8_S"; cache = "analysis/ffjoint_cache.npz"
KC = dict(np.load(cache)) if os.path.exists(cache) else {}
for path, stem in (("u", "unet_cm7_am22_l1_i10000_boot_fflowpass{}_low"), ("d", "drunet_fflowpass{}_low")):
    for s in (32, 48, 64, 96):
        key = f"{path}{s}"
        if key not in KC:
            A = np.load(f"outputs/probe2/{M}{stem.format(s)}_sh2/emb.npz")["train"].astype(np.float32).reshape(500, 7, 11, 1024)[:, 1:5]
            KC[key] = kmat(A.reshape(500, -1), 4, 1024); print("커널", key, flush=True)
np.savez(cache, **KC)
f = lambda K: fit(prep(K), 0.01)[0]; FFs = (32, 48, 64, 96)
avg_u = np.mean([f(KC[f"u{s}"]) for s in FFs], 0); avg_j = np.mean([f(KC[f"u{s}"] + KC[f"d{s}"]) for s in FFs], 0)
jt_u = f(sum(KC[f"u{s}"] for s in FFs) / 4); jt_j = f(sum(KC[f"u{s}"] + KC[f"d{s}"] for s in FFs) / 4)
print(f"U-Net 경로: σ별 예측 평균 {rm(avg_u):.3f} | σ 결합 {rm(jt_u):.3f}")
print(f"두 경로: σ별 (경로 결합) 예측 평균 {rm(avg_j):.3f} (= 지금 Dj) | σ·경로 모두 결합 {rm(jt_j):.3f}")
s = "_a0.01"; c = y.mean()
G = dict(P=oof(E + f"mr_v14_pp_sc{s}"), A=grp([E + f"dnpp0.25_ff{q}{s}" for q in FF])[0], T=grp([E + f"tips16_dnppm0.05_ff{q}{s}" for q in FF])[0], J=grp([E + f"cnnfj4_ff{q}{s}" for q in FF])[0])
W = dict(P=.1765, A=.1456, T=.2059, J=.2471, Dj=.225)
for nm, oDj in (("Dj = σ별 예측 평균 (v24a)", avg_j), ("Dj = σ·경로 결합", jt_j), ("Dj = 둘의 평균", (avg_j + jt_j) / 2)):
    o = c + 1.03 * (sum(v * G[k] for k, v in W.items() if k != "Dj") + W["Dj"] * oDj - c); print(f"  블렌드 {nm:28s} OOF {rm(o):.3f} pw13 {rw(o, w13):.3f}")
