"""v35 후보: J3에 측정 특징(상분율·수작업·평활 CDF, σ48 U 경로, 상대 비중 0.5)을 같은 Ridge로 결합한 J3m.
(a) P 유지(0.1765 고정)  (b) P 제거(나머지 중첩 비음수 적합). 사용: python b51_j3m.py [--save 이름 배율 프리셋]"""
import sys, io, contextlib; sys.path.insert(0, "analysis"); _argv51 = sys.argv; sys.argv = ["b48_joint3.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b48_joint3.py", encoding="utf-8").read())
sys.argv = _argv51
from scipy.optimize import nnls
G["J3m"] = grp([E + f"joint3m0.5_ff{f}{s}" for f in FF]); b30.G["J3m"] = G["J3m"]
print(f"구성원 OOF: J3 {rm(G['J3'][0]):.3f} | J3m {rm(G['J3m'][0]):.3f} | P {rm(G['P'][0]):.3f} | J3m-P 잔차상관 {np.corrcoef(G['J3m'][0] - y, G['P'][0] - y)[0, 1]:.3f}")
W32 = P8["v32b: v29b에서 FDm→J3 (Nk 없음) + 재조정 0.5"]; b30.base, b30.bt = b30.bl(W32, 0, 1.03), b30.bl(W32, 1, 1.03)
def nested_free(W, k=1.03):
    names = list(W); X = np.stack([G[n][0] for n in names], 1)
    fitw = lambda Xs, ys: (lambda w: w / w.sum())(nnls(Xs - c, ys - c)[0])
    o = np.zeros(len(y))
    for f in range(5):
        t, v = fold != f, fold == f; o[v] = c + k * (X[v] @ fitw(X[t], y[t]) - c)
    return o, dict(zip(names, np.round(fitw(X, y), 4)))
o0, _ = nested_fixP(W32); print(f"  [v32b 중첩·P고정] {rm(o0):.3f}")
Wa = sw(W32, "J3", "J3m"); oa, Wfa = nested_fixP(Wa); print(f"  [(a) J3→J3m, P 유지 중첩] {rm(oa):.3f} | 전체 적합 {dict((k, round(float(v), 3)) for k, v in Wfa.items())}")
Wb0 = {k: v for k, v in Wa.items() if k != "P"}; ob, Wfb = nested_free(Wb0); print(f"  [(b) J3→J3m, P 제거 중첩] {rm(ob):.3f} | 전체 적합 {Wfb}")
oc, Wfc = nested_free({k: v for k, v in W32.items() if k != "P"}); print(f"  [(참고) v32b에서 P만 제거 중첩] {rm(oc):.3f}")
P11 = {"v32b (기준)": W32, "v35a: J3→J3m (P 유지)": Wa, "v35a + 재조정 0.5": mid_w(Wa, Wfa, 0.5), "v35b: J3→J3m, P 제거 (중첩 적합 가중)": {k: float(v) for k, v in Wfb.items()}}
w18 = wg(18.0)
if "--save" not in sys.argv:
    for nm, w in P11.items():
        o, _ = show(nm, w); print(f"      Private형 {rm(o, w=w18):.3f} | 가운데 {rm(o, mid):.3f} 양끝 {rm(o, ~mid):.3f} | 가중 {dict((a, round(float(b), 3)) for a, b in w.items())}")
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", P11[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, {a: round(float(b), 4) for a, b in P11[pre].items()})
