"""DA 깊이 디코더 특징 구성원(σ48)·깊이 지도 통계가 v29b에 정보를 더하는지: 중첩 가중(P 고정)과 잔차 설명력."""
import sys, io, contextlib; sys.path.insert(0, "analysis"); sys.argv = ["b39_da.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b39_da.py", encoding="utf-8").read())
from sklearn.linear_model import RidgeCV
W = P3["v29b: FoL→(FoL+DAo)/2 + 재조정 0.5"]; o29 = bl(W, 0, 1.03); res = y - o29
ob, _ = nested_fixP(W); print(f"v29b OOF {rm(o29):.3f} | 구성원만 중첩 가중 {rm(ob):.3f}")
for n in ("daneck_all_ff48_a0.01", "daneck_hi_ff48_a0.01"):
    G[n] = (oof(E + n), np.zeros(1000)); b30.G[n] = G[n]
    o_, Wf_ = nested_fixP({**W, n: 0.05})
    print(f"  + {n:24s} 잔차상관 {np.corrcoef(G[n][0] - y, res * -1)[0, 1]:+.3f} | 중첩 가중 {rm(o_):.3f} | 전체 적합 비중 {Wf_[n]:.3f}")
ids = pd.read_csv("data/train.csv").ID
A = pd.read_csv("outputs/features_cache/dadepth_drunetft_ft1b_fflowpass48_low_train.csv").set_index("ID").loc[ids]
X = np.column_stack([A.values.astype(float), sig, Lq]); pr = np.zeros(len(y))
for k in range(5):
    t, v = fold != k, fold == k; mu, sd = X[t].mean(0), X[t].std(0) + 1e-9
    r = RidgeCV(alphas=np.logspace(-2, 6, 40)).fit((X[t] - mu) / sd, res[t]); pr[v] = r.predict((X[v] - mu) / sd)
print(f"  깊이 지도 통계 {A.shape[1]}개: v29b 잔차 설명력 R² {1 - np.mean((res - pr) ** 2) / np.mean(res ** 2):+.3f} → 보정 OOF {rm(o29 + pr):.3f}")
d2 = (o29 + pr - y) ** 2 - (o29 - y) ** 2
print(f"  폴드별 개선 {[round(float(d2[fold == k].mean()), 2) for k in range(5)]} ({sum(d2[fold == k].mean() < 0 for k in range(5))}/5) | 고품질 {rm(o29 + pr, ~Lq):.3f} (원래 {rm(o29, ~Lq):.3f}) 저품질 {rm(o29 + pr, Lq):.3f} (원래 {rm(o29, Lq):.3f}) | 가운데 {rm(o29 + pr, mid):.3f} (원래 {rm(o29, mid):.3f})")
mu, sd = X.mean(0), X.std(0) + 1e-9; r = RidgeCV(alphas=np.logspace(-2, 6, 40)).fit((X - mu) / sd, res)
names = list(A.columns) + ["sig", "lq"]; cf = sorted(zip(names, r.coef_), key=lambda t: -abs(t[1]))
print(f"  전체 적합 α {r.alpha_:.3g}, 계수 상위: " + ", ".join(f"{n} {c:+.2f}" for n, c in cf[:6]))
for n in A.columns:
    print(f"    {n:14s} 잔차상관 {np.corrcoef(A[n].values, res)[0, 1]:+.3f} | 경도상관 {np.corrcoef(A[n].values, y)[0, 1]:+.3f}")
