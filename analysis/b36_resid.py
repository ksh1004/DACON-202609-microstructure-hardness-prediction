"""v28a OOF 잔차에 남은 체계적 패턴: 측정 특징(σ48 U 경로)·노이즈 σ로 잔차를 중첩 교차검증 Ridge로 예측 → 설명력·보정 효과."""
import sys, io, contextlib; sys.path.insert(0, "analysis"); sys.argv = ["b34.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b34.py", encoding="utf-8").read())
from sklearn.linear_model import RidgeCV
W = PRE2["v28 + 재조정 0.5"]; o = bl(W, 0, 1.03); res = y - o
print(f"v28a OOF {rm(o):.3f} | 잔차 평균: 고품질 {res[~Lq].mean():+.2f} 저품질 {res[Lq].mean():+.2f}")
S = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"; ids = pd.read_csv("data/train.csv").ID
def load(pre, rx):
    A = pd.read_csv(f"outputs/features_cache/{pre}_train.csv").set_index("ID").loc[ids]; return A[[c for c in A.columns if pd.Series([c]).str.match(rx)[0]]]
sets = {"σ만": pd.DataFrame({"sig": sig, "lq": Lq.astype(float)}, index=ids),
        "상분율": load(f"phase_{S}", "pf_"), "수작업": load(S, "(ph|pore|gr|sp)_"), "평활CDF": load(f"smcdf3_{S}", "sc")}
sets["전부"] = pd.concat(list(sets.values()), axis=1)
for nm, F in sets.items():
    X = F.replace([np.inf, -np.inf], np.nan).fillna(F.median()).values.astype(float); X = np.column_stack([X, sig, Lq])
    pr = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k; mu, sd = X[t].mean(0), X[t].std(0) + 1e-9
        r = RidgeCV(alphas=np.logspace(-2, 6, 40)).fit((X[t] - mu) / sd, res[t]); pr[v] = r.predict((X[v] - mu) / sd)
    r2 = 1 - np.mean((res - pr) ** 2) / np.mean(res ** 2)
    print(f"  {nm:6s} 특징 {X.shape[1]:4d}개 | 잔차 설명력 R² {r2:+.3f} | 보정 OOF {rm(o + pr):.3f} (고 {rm(o + pr, ~Lq):.3f} 저 {rm(o + pr, Lq):.3f})")
