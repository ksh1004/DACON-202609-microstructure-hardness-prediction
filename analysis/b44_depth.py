"""v29b + Depth-Anything 깊이 디코더 구성원(Nk, 4보정 평균) / + 깊이 지도 통계(4보정 평균) 잔차 보정 — 모두 중첩 교차검증.
사용: python b44_depth.py [--save 이름 배율 프리셋]"""
import sys, io, contextlib; sys.path.insert(0, "analysis"); _argv44 = sys.argv; sys.argv = ["b39_da.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b39_da.py", encoding="utf-8").read())
sys.argv = _argv44
from sklearn.linear_model import RidgeCV
ids = pd.read_csv("data/train.csv").ID
G["Nk"] = grp([E + f"daneck_all_ff{f}{s}" for f in FF]); b30.G["Nk"] = G["Nk"]
W29 = P3["v29b: FoL→(FoL+DAo)/2 + 재조정 0.5"]; b30.base, b30.bt = b30.bl(W29, 0, 1.03), b30.bl(W29, 1, 1.03)
print(f"구성원 Nk(디코더, 4보정) OOF {rm(G['Nk'][0]):.3f} | FDm과 잔차상관 {np.corrcoef(G['Nk'][0] - y, G['FDm'][0] - y)[0, 1]:.3f}")
o_b, _ = nested_fixP(W29); o_n, Wn = nested_fixP({**W29, "Nk": 0.05})
print(f"  중첩·P고정: v29b 구성원 {rm(o_b):.3f} → +Nk {rm(o_n):.3f} | 전체 적합 {dict((k, round(float(v), 3)) for k, v in Wn.items())}")
P5 = {"v29b (기준)": W29, "v30n: v29b + Nk 재조정 0.5": mid_w({**W29, "Nk": 0.0}, Wn, 0.5)}
# 깊이 지도 통계 (4보정 평균) → v29b 잔차 보정 Ridge (중첩)
def dd(split):
    ii = ids if split == "train" else te
    return np.mean([pd.read_csv(f"outputs/features_cache/dadepth_drunetft_ft1b_fflowpass{f}_low_{split}.csv").set_index("ID").loc[ii].values for f in FF], 0)
Dtr, Dte = dd("train"), dd("test")
def corr_model(base_o, base_t):
    res = y - base_o; X = np.column_stack([Dtr, sig, Lq]); Xt = np.column_stack([Dte, sig_te, Lq_te]); pr = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k; mu, sd = X[t].mean(0), X[t].std(0) + 1e-9
        r = RidgeCV(alphas=np.logspace(-2, 6, 40)).fit((X[t] - mu) / sd, res[t]); pr[v] = r.predict((X[v] - mu) / sd)
    mu, sd = X.mean(0), X.std(0) + 1e-9; r = RidgeCV(alphas=np.logspace(-2, 6, 40)).fit((X - mu) / sd, res)
    return base_o + pr, base_t + r.predict((Xt - mu) / sd), r.alpha_
sig_te = pd.read_csv("outputs/features/test_features.csv").set_index("ID").loc[te, "q_noise_sigma"].values; Lq_te = (sig_te >= 10).astype(float)
for nm in list(P5):
    o, t = bl(P5[nm], 0, 1.03), bl(P5[nm], 1, 1.03); oc, tc, al = corr_model(o, t)
    d2 = (oc - y) ** 2 - (o - y) ** 2
    print(f"  [{nm}] 깊이 통계 보정: OOF {rm(o):.3f} → {rm(oc):.3f} (α {al:.3g}, 폴드 {sum(d2[fold == k].mean() < 0 for k in range(5))}/5, 고 {rm(oc, ~Lq):.3f} 저 {rm(oc, Lq):.3f}, 가운데 {rm(oc, mid):.3f}→원래 {rm(o, mid):.3f})")
if "--save" not in sys.argv:
    for nm, w in P5.items(): show(nm, w)
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", P5[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, {a: round(float(b), 4) for a, b in P5[pre].items()})
