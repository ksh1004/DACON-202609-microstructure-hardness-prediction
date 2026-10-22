"""위상 특징(src/topo_features.py, F 경로 σ48) 평가: 단독 Ridge, v28a 잔차 설명력(중첩), 비교용 기존 측정 특징."""
import sys, io, contextlib; sys.path.insert(0, "analysis"); sys.argv = ["b34.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b34.py", encoding="utf-8").read())
from sklearn.linear_model import RidgeCV
ids = pd.read_csv("data/train.csv").ID
W = PRE2["v28 + 재조정 0.5"]; o28 = bl(W, 0, 1.03); res = y - o28
def X_of(pre, rx=None):
    A = pd.read_csv(f"outputs/features_cache/{pre}_train.csv").set_index("ID").loc[ids]
    if rx: A = A[[c for c in A.columns if pd.Series([c]).str.match(rx)[0]]]
    return A.replace([np.inf, -np.inf], np.nan).fillna(A.median()).values.astype(float)
def cv(X, target):
    pr = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k; mu, sd = X[t].mean(0), X[t].std(0) + 1e-9
        Z = np.clip((X - mu) / sd, -6, 6); r = RidgeCV(alphas=np.logspace(-2, 6, 40)).fit(Z[t], target[t]); pr[v] = r.predict(Z[v])
    return pr
T = X_of("topo_drunetft_ft1b_fflowpass48_low")
S = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
for nm, X in (("위상(F 경로)", T), ("위상 s0만", X_of("topo_drunetft_ft1b_fflowpass48_low", "tp_.*_s0_")), ("상분율(U 경로)", X_of(f"phase_{S}", "pf_")),
              ("수작업(U 경로)", X_of(S, "(ph|pore|gr|sp)_")), ("위상+상분율", np.hstack([T, X_of(f"phase_{S}", "pf_")]))):
    p = cv(X, y); pr = cv(X, res); r2 = 1 - np.mean((res - pr) ** 2) / np.mean(res ** 2)
    print(f"  {nm:12s} {X.shape[1]:4d}개 | 단독 경도 OOF {rm(p):.3f} (고 {rm(p, ~Lq):.2f} 저 {rm(p, Lq):.2f}) | v28a 잔차 설명력 R² {r2:+.3f} → 보정 OOF {rm(o28 + pr):.3f}", flush=True)
