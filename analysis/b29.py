"""v25 후보: v24a에 세 번째 이미지 경로(증류 디노이저 DRT2)를 반영 — DINOv2는 세 경로 결합(mv16j3), TIPSv2·고정 CNN은 기존 경로와 DRT2 경로 예측 평균.
사용: python b29.py [--tag drt2_i10000] [--save 이름 배율]"""
import sys, os; sys.path.insert(0, "analysis")
from blend_eval import *
TAG = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "drt2_i10000"; s = "_a0.01"; c = y.mean()
ok = lambda n: all(os.path.exists(E + n.format(f) + "/oof.csv") for f in FF)
G = dict(P=(oof(E + f"mr_v14_pp_sc{s}"), tp(E + f"mr_v14_pp_sc{s}")), A=grp([E + f"dnpp0.25_ff{f}{s}" for f in FF]), Dj=grp([E + f"mv16j_ff{f}{s}" for f in FF]),
         T=grp([E + f"tips16_dnppm0.05_ff{f}{s}" for f in FF]), J=grp([E + f"cnnfj4_ff{f}{s}" for f in FF]))
for k, n in (("Dj3", f"mv16j3_{TAG}_ff{{}}{s}"), ("Tx", f"tips16_{TAG}_ff{{}}{s}"), ("Jx", f"cnnfj4_{TAG}_ff{{}}{s}")):
    if ok(n): G[k] = grp([E + n.format(f) for f in FF])
    else: print("미완료:", k)
if "Tx" in G: G["Tm"] = tuple((a_ + b_) / 2 for a_, b_ in zip(G["T"], G["Tx"]))
if "Jx" in G: G["Jm"] = tuple((a_ + b_) / 2 for a_, b_ in zip(G["J"], G["Jx"]))
W24 = dict(P=.1765, A=.1456, Dj=.2250, T=.2059, J=.2471)
wg = lambda sd: np.exp(-0.5 * ((y - y.mean()) / sd) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166 = wg(16.6)
rm = lambda o, m=None, w=None: np.sqrt(np.average(((o - y) ** 2) if m is None else ((o - y)[m] ** 2), weights=w)); Lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
bl = lambda w, i, k: c + k * (sum(v * G[n][i] for n, v in w.items()) - c); base, bt = bl(W24, 0, 1.03), bl(W24, 1, 1.03)
print("구성원 OOF: " + " | ".join(f"{n} {rm(G[n][0]):.3f}" for n in G))
def show(nm, w, k=1.03):
    o, t = bl(w, 0, k), bl(w, 1, k); d = (o - y) ** 2 - (base - y) ** 2; cv = np.cov(y, o)
    print(f"{nm:40s} OOF {rm(o):.3f} pw13 {rm(o, w=w13):.3f} Public꼴 {rm(o, w=w166):.3f} | 고 {rm(o, ~Lq):.3f} 저 {rm(o, Lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} {sum(d[fold == f].mean() < 0 for f in range(5))}/5 | 가운데 {d[mid].sum() / 500:+.2f} 양끝 {d[~mid].sum() / 500:+.2f} | 기울기 {cv[0, 1] / cv[1, 1]:.3f} test std {t.std():.2f} v24a와 RMS {np.sqrt(((t - bt) ** 2).mean()):.2f}")
    return o, t
show("v24a", W24)
PRE = {}
if "Dj3" in G: PRE["Dj → Dj3"] = {**W24, "Dj": 0, "Dj3": W24["Dj"]}
if "Tm" in G: PRE["T → 두 경로 평균"] = {**W24, "T": 0, "Tm": W24["T"]}
if "Jm" in G: PRE["J → 두 경로 평균"] = {**W24, "J": 0, "Jm": W24["J"]}
if all(k in G for k in ("Dj3", "Tm", "Jm")): PRE["세 가지 모두"] = {"P": W24["P"], "A": W24["A"], "Dj3": W24["Dj"], "Tm": W24["T"], "Jm": W24["J"]}
if all(k in G for k in ("Dj3", "Tx")):
    PRE["Dj3 + T→DRT만, J 그대로"] = {"P": W24["P"], "A": W24["A"], "Dj3": W24["Dj"], "Tx": W24["T"], "J": W24["J"]}
    PRE["Dj3 + T→DRT만, J 평균"] = {"P": W24["P"], "A": W24["A"], "Dj3": W24["Dj"], "Tx": W24["T"], "Jm": W24["J"]}
    PRE["Dj3 + T 평균, J 그대로"] = {"P": W24["P"], "A": W24["A"], "Dj3": W24["Dj"], "Tm": W24["T"], "J": W24["J"]}
    PRE["위 + A를 Dj3로"] = {"P": W24["P"], "Dj3": W24["Dj"] + W24["A"], "Tm": W24["T"], "J": W24["J"]}
for nm, w in PRE.items(): show(nm, {k: v for k, v in w.items() if v})
if "세 가지 모두" in PRE:
    for k in (1.0, 1.05): show(f"세 가지 모두 ×{k}", {k_: v for k_, v in PRE["세 가지 모두"].items() if v}, k)
if "--save" in sys.argv:
    i = sys.argv.index("--save"); name, k = sys.argv[i + 1], float(sys.argv[i + 2]); pre = sys.argv[i + 3] if len(sys.argv) > i + 3 else "세 가지 모두"; w = {k_: v for k_, v in PRE[pre].items() if v}
    o, t = show(f"저장: {name}", w, k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, w)
