"""견고 회귀 시험: 커널 Ridge + LOO 잔차 기반 Huber 가중 (IRLS). 라벨 이상치가 최소노름 보간을 해치는지 확인."""
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
def krr_loo(K, yy, alphas, w=None):
    """가중 KRR: (K + α W^-1) a = y. LOO 잔차 = a_i / (inv)_ii. 최적 α는 가중 LOO로 선택"""
    best = None
    for al in alphas:
        M = K + al * np.diag(1 / w if w is not None else np.ones(len(yy)))
        Mi = np.linalg.inv(M); a = Mi @ yy; loo = a / np.diag(Mi)
        sc = np.average(loo ** 2, weights=w)
        if best is None or sc < best[0]: best = (sc, al, a, loo)
    return best
res = {c: np.zeros(len(y)) for c in ("ridge", "h2.0", "h1.5", "h1.0")}
for k in range(5):
    t = fold != k; X = design(t); ym = y[t].mean()
    K = X[t] @ X[t].T; Kv = X[~t] @ X[t].T; al = np.logspace(1, 6, 26)
    _, a0, coef, loo = krr_loo(K, y[t] - ym, al)
    res["ridge"][~t] = Kv @ coef + ym
    for c in (2.0, 1.5, 1.0):
        lo, co_ = loo, coef
        for it in range(4):
            s = 1.4826 * np.median(np.abs(lo)); w = np.minimum(1, c * s / np.abs(lo))
            _, a1, co_, lo = krr_loo(K, y[t] - ym, al, w)
        res[f"h{c}"][~t] = Kv @ co_ + ym
        if k == 0: print(f"  fold0 c={c}: 가중<1 비율 {np.mean(w < 1):.2f}, 최소 가중 {w.min():.2f}, α {a0:.0f}→{a1:.0f}")
r = res["ridge"] - y
print("잔차 첨도(정규=0) %.2f, |r|>2.5σ 비율 %.3f (정규 0.012), 최대 |r| %.1f, 정규성 p=%.3f" % (stats.kurtosis(r), np.mean(np.abs(r) > 2.5 * r.std()), np.abs(r).max(), stats.normaltest(r).pvalue))
for c, o in res.items(): rep(c, o, res["ridge"] if c != "ridge" else None)
