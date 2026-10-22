"""σ48·8보기 F 경로 변형(ft1b·ft1s1b·ft1s1·ft1)의 '예측 평균' vs 단독 (층 11~17 추출본, λ 0.01) — 3계열 블렌드 포함."""
import sys, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
from lam_sweep import prep, fit, rm
from kmat_util import kmat
from blend_eval import *
TP = dict(np.load("analysis/twopath_cache.npz")); f = lambda K: fit(prep(K), 0.01)[0]; lq = sig >= 10
oT, oF = f(TP["T_u"]), f(TP["F_u"]); bl = lambda oD: .45 * oD + .25 * oT + .30 * oF
kv = lambda st: kmat(np.mean([np.load(f"outputs/probe2_views/{st}/train_s0_k{k}.npy").astype(np.float32) for k in range(8)], 0), 4, 1024)
O = {n: f(kv(st)) for n, st in (("Fb", "drunetft_ft1b_fflowpass48_low"), ("Fsb", "drunetft_ft1s1b_fflowpass48_low"), ("Fs", "drunetft_ft1s1_fflowpass48_low"), ("F", "drunetft_ft1_fflowpass48_low"))}
for combo in (["Fb"], ["Fsb"], ["Fb", "Fsb"], ["Fb", "Fsb", "Fs"], ["Fb", "Fsb", "Fs", "F"]):
    o = np.mean([O[c] for c in combo], 0)
    print(f"  {'+'.join(combo):14s} 예측 평균 {rm(o):.3f} (고 {rm(o, ~lq):.2f} 저 {rm(o, lq):.2f}) → 블렌드 {rm(bl(o)):.3f}", flush=True)
print("  잔차상관 Fb-Fsb %.3f, Fb-Fs %.3f" % (np.corrcoef(O["Fb"] - y, O["Fsb"] - y)[0, 1], np.corrcoef(O["Fb"] - y, O["Fs"] - y)[0, 1]))
