"""모드별 특징 표준화: DINOv2 특징을 고품질(σ<10)/저품질(σ≥10) 각각의 train 폴드 통계로 따로 표준화 (σ48, 8보기, 세 경로 결합, λ 0.01)."""
import sys, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    import lam_sweep as L
from lam_sweep import prep, fit, rm, rw
from blend_eval import *
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05)); lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
TP = dict(np.load("analysis/twopath_cache.npz")); f = lambda K: fit(prep(K), 0.01)[0]
STEMS = ["unet_cm7_am22_l1_i10000_boot_fflowpass48_low", "drunet_fflowpass48_low", "unet_drt10b_i10000_fflowpass48_low"]
FE = [np.mean([np.load(f"outputs/probe2_views/{s}/train_s0_k{k}.npy").astype(np.float32) for k in range(8)], 0).astype(np.float64).reshape(500, 4, 11, 1024) for s in STEMS]
def kmat(X, mode, shrink=1.0):
    Ks = []
    for k in range(5):
        t = fold != k; K = np.zeros((500, 500))
        for blocks, g in GR:
            Z = X[:, :, list(blocks)].reshape(500, -1).copy(); mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10
            if mode == "none": Z = (Z - mu) / np.where(keep, sd, 1)
            else:
                out = np.empty_like(Z)
                for m in (lq, ~lq):
                    mm, sm = Z[t & m].mean(0), Z[t & m].std(0)
                    mm = shrink * mm + (1 - shrink) * mu
                    out[m] = (Z[m] - mm) / (np.where(keep, sd, 1) if mode == "mean" else np.where(sm > 1e-10, sm, 1))
                Z = out
            Z = np.clip(Z[:, keep], -6, 6); K += g * (Z @ Z.T) / keep.sum()
        Ks.append(K)
    return np.array(Ks)
oT, oF = f(TP["T_u"]), f(TP["F_u"]); bl = lambda oD: .45 * oD + .25 * oT + .30 * oF
base = None
for mode, sh in (("none", 1), ("mean", 1), ("mean", .5), ("both", 1)):
    o = f(sum(kmat(X, mode, sh) for X in FE)); b = bl(o)
    if base is None: base = b
    d = (b - y) ** 2 - (base - y) ** 2; s_lq = np.polyfit(y[lq], o[lq], 1)[0]; s_hq = np.polyfit(y[~lq], o[~lq], 1)[0]
    print(f"{mode}×{sh}: 세 경로 결합 {rm(o):.3f} (고 {rm(o, ~lq):.2f} 저 {rm(o, lq):.2f}, 기울기 고 {s_hq:.3f} 저 {s_lq:.3f}) → 블렌드 {rm(b):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5, 가운데 {d[mid].sum() / 500:+.2f}, pw13 {rw(b, w13):.3f})", flush=True)
