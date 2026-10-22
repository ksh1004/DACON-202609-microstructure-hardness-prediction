"""목표값 변환 시험: g(y)로 Ridge(커널) 적합 → 역변환. 블록 = DINO hetero3 + 상풀링 (σ48)."""
import sys; sys.path.insert(0, "analysis"); sys.path.insert(0, "src")
from blend_eval import *
from multi_ridge import load_group
from scipy import stats
te_df = pd.read_csv("data/sample_submission.csv")
S = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"; Em = f"outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2_S{S}"
groups = [load_group(s, tr, te_df) for s in (f"emb:{Em}:1:1,2,3,4:1024:10:0,1,2,3,4,5", f"emb:{Em}:0.25:1,2,3,4:1024:10:6,7,8,9")]
def design(t):
    out, ref = [], None
    for g, A, B, _ in groups:
        mu, sd = A[t].mean(0), A[t].std(0); keep = sd > 1e-10
        Z = np.clip((A[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        out.append(Z * s)
    return np.hstack(out)
def krr(K, Kv, yy, alphas=np.logspace(0, 6, 25)):
    best = None
    for al in alphas:
        Mi = np.linalg.inv(K + al * np.eye(len(yy))); a = Mi @ yy; loo = a / np.diag(Mi)
        if best is None or (loo ** 2).mean() < best[0]: best = ((loo ** 2).mean(), a)
    return Kv @ best[1]
T = {"없음": (lambda v: v, lambda z: z)}
for c in (0, 100, 130, 140):
    T[f"log(y-{c})"] = ((lambda v, c=c: np.log(v - c)), (lambda z, c=c: np.exp(z) + c))
T["-1/(y-100)"] = (lambda v: -1 / (v - 100), lambda z: 100 - 1 / np.minimum(z, -1e-4))
res = {k: np.zeros(len(y)) for k in T}; res["분위-정규"] = np.zeros(len(y))
for k in range(5):
    t = fold != k; X = design(t); K = X[t] @ X[t].T; Kv = X[~t] @ X[t].T
    for nm, (g, gi) in T.items():
        z = g(y[t]); res[nm][~t] = gi(krr(K, Kv, z - z.mean()) + z.mean())
    ys = np.sort(y[t]); q = stats.norm.ppf((np.arange(len(ys)) + .5) / len(ys)); z = np.interp(y[t], ys, q)
    res["분위-정규"][~t] = np.interp(krr(K, Kv, z), q, ys)
for nm, o in res.items():
    rep(nm, o, res["없음"] if nm != "없음" else None)
    print(f"{'':40s} 예측 왜도 {stats.skew(o):.2f}, 상위5% 예측 구간 실제−예측 {np.mean((y - o)[o >= np.quantile(o, .95)]):+.1f}")
