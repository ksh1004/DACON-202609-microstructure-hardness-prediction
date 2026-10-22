"""v16 계열 Public 결과 해석 (10/13): (1) 300장 짝비교의 잡음 크기를 OOF 부트스트랩으로 추정,
(2) 채점된 제출들의 블렌드 Public 예상 (항등식: MSE(Σw p) = Σw M − ½ ΣΣ w_i w_j D_ij, D는 test 전체로 근사)."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
SC = {"sub_v15a_dinopp.csv": 10.2913472134, "sub_v16a_mv16.csv": 10.3467169407, "sub_v16_noP.csv": 10.3171457020, "sub_v16_noC.csv": 10.3509407312,
      "sub_v14c_blend_v12b_v14b.csv": 10.3082131304, "sub_v12b_fullfit.csv": 10.3926576961, "sub_v13c_blend_v12a_v13a.csv": 10.3465580153}
T = {f: pd.read_csv(f"submissions/{f}").set_index("ID").loc[te, "hardness"].values for f in SC}
nm = lambda f: f.replace("sub_", "").replace(".csv", "")[:16]
# (1) OOF 부트스트랩: 300장을 뽑았을 때 두 모델 RMSE 차의 표준편차
co, _ = grp(CNN); do, _ = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF]); po = oof(E + "mv16_mr_pp_sc")
O = {"v15a": np.load("analysis/v15a_oof.npy"), "v16a": .15 * co + .55 * do + .3 * po, "noP": .15 * co + .85 * do, "noC": (.55 * do + .3 * po) / .85}
rng = np.random.default_rng(0)
def boot(a, b, n=300, B=4000):
    ea, eb = (O[a] - y) ** 2, (O[b] - y) ** 2; d = []
    for _ in range(B):
        i = rng.integers(0, len(y), n); d.append(np.sqrt(eb[i].mean()) - np.sqrt(ea[i].mean()))
    return np.std(d)
print("== 300장 표본에서 두 모델 RMSE 차이의 표준편차 (OOF 부트스트랩) ==")
for a, b, obs in (("v15a", "v16a", SC["sub_v16a_mv16.csv"] - SC["sub_v15a_dinopp.csv"]), ("v16a", "noP", SC["sub_v16_noP.csv"] - SC["sub_v16a_mv16.csv"]), ("v16a", "noC", SC["sub_v16_noC.csv"] - SC["sub_v16a_mv16.csv"])):
    s = boot(a, b); print(f"  {b} − {a}: 예측 RMS차 {np.sqrt(((O[a]-O[b])**2).mean()):.2f} | 잡음 표준편차 ±{s:.3f} | Public 관측 {obs:+.4f} ({obs/s:+.1f}σ) | OOF에서는 {np.sqrt(((O[b]-y)**2).mean()) - np.sqrt(((O[a]-y)**2).mean()):+.3f}")
# (2) 블렌드 항등식
F = list(SC); M = np.array([SC[f] ** 2 for f in F]); D = np.array([[((T[a] - T[b]) ** 2).mean() for b in F] for a in F])
def pub(w): w = np.asarray(w, float); return np.sqrt(w @ M - 0.5 * w @ D @ w)
def W(**kw): return [kw.get(nm(f), 0.0) for f in F]
print("\n== 블렌드의 Public 예상 (항등식) ==")
i15, i16, iP, iC, i14 = (F.index(f) for f in ("sub_v15a_dinopp.csv", "sub_v16a_mv16.csv", "sub_v16_noP.csv", "sub_v16_noC.csv", "sub_v14c_blend_v12b_v14b.csv"))
def w_of(d):
    w = np.zeros(len(F))
    for k, v in d.items(): w[k] = v
    return w
for label, d in (("v15a 단독", {i15: 1}), ("v16a 단독", {i16: 1}),
                 ("v15a : v16a = 0.7 : 0.3", {i15: .7, i16: .3}), ("v15a : v16a = 0.6 : 0.4", {i15: .6, i16: .4}), ("v15a : v16a = 0.5 : 0.5", {i15: .5, i16: .5}),
                 ("v15a : v16a = 0.4 : 0.6", {i15: .4, i16: .6}), ("v15a : v16a = 0.3 : 0.7", {i15: .3, i16: .7}),
                 ("v15a : noP = 0.5 : 0.5", {i15: .5, iP: .5}), ("v15a : v16a : noP = 1/3씩", {i15: 1/3, i16: 1/3, iP: 1/3}),
                 ("v15a : v14c : v16a = 1/3씩", {i15: 1/3, i14: 1/3, i16: 1/3})):
    print(f"  {label:32s} {pub(w_of(d)):.4f}")
print("  D(v15a, v16a) = %.2f (RMS %.2f)" % (D[i15, i16], np.sqrt(D[i15, i16])))
if "--save" in sys.argv:
    t = .5 * T["sub_v15a_dinopp.csv"] + .5 * T["sub_v16a_mv16.csv"]
    pd.DataFrame({"ID": te, "hardness": t}).to_csv("submissions/sub_v17a_blend_v15a_v16a.csv", index=False)
    o = .5 * O["v15a"] + .5 * O["v16a"]; rep("v17a = v15a : v16a = 1 : 1 (OOF)", o, O["v15a"]); print("→ submissions/sub_v17a_blend_v15a_v16a.csv")
