"""v21 후보: v19a의 성분을 바꾸거나 더한 블렌드를 v19a와 비교 (OOF, pw13, Public 꼴, 경도 구간 분해, test 예측 차이).
성분: C 미세조정 CNN, P 측정 블록, A DINOv2 4보기, D DINOv2 16보기, T TIPSv2, F 고정 CNN(3개 예측 평균), Fj3/Fj4 고정 CNN 결합 Ridge, X = 새 계열(있으면).
사용: python b25.py [--x <emb_reg 이름 접두어, 예: capi16_dnppm0.05>] [--save 이름 wC wP wA wD wT wF wX [F종류=Fj4]]"""
import sys, os; sys.path.insert(0, "analysis")
from blend_eval import *
xn = sys.argv[sys.argv.index("--x") + 1] if "--x" in sys.argv else None
ex = lambda n: all(os.path.exists(E + f"{n}_ff{f}/oof.csv") for f in FF)
G = dict(C=grp(CNN), P=(oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")), A=grp([E + f"dnpp0.25_ff{f}" for f in FF]), D=grp([E + f"mv16_dnppm0.05_ff{f}" for f in FF]),
         T=grp([E + f"tips16_dnppm0.05_ff{f}" for f in FF]), F=grp([E + f"cnnf_{n}_ff{f}" for n in ("dn121", "r50a1", "r50a2") for f in FF]))
for k, n in (("Fj3", "cnnfj3"), ("Fj4", "cnnfj4"), ("X", xn)):
    if n and ex(n): G[k] = grp([E + f"{n}_ff{f}" for f in FF])
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166, w18 = wg(16.6), wg(18.0)
rm = lambda o, m=None, w=None: np.sqrt(np.average(((o - y) ** 2) if m is None else ((o - y)[m] ** 2), weights=w))
Lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
bl = lambda w, i: sum(v * G[k][i] for k, v in w.items() if v)
W19 = dict(C=.15, P=.15, A=.12375, D=.19125, T=.175, F=.21); base, bt = bl(W19, 0), bl(W19, 1)
print("성분 단독 OOF: " + " ".join(f"{k} {rm(v[0]):.3f}" for k, v in G.items()))
def show(nm, w):
    assert abs(sum(w.values()) - 1) < 1e-9, (nm, sum(w.values())); o, t = bl(w, 0), bl(w, 1); d = (o - y) ** 2 - (base - y) ** 2
    print(f"{nm:44s} OOF {rm(o):.3f} pw13 {rm(o, w=w13):.3f} Public꼴 {rm(o, w=w166):.3f} Private꼴 {rm(o, w=w18):.3f} | 고 {rm(o, ~Lq):.3f} 저 {rm(o, Lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | 가운데 {d[mid].sum() / 500:+.2f} 양끝 {d[~mid].sum() / 500:+.2f} | test RMS차 {np.sqrt(((t - bt) ** 2).mean()):.2f}")
    return o, t
show("v19a (기준)", W19)
sw = lambda **kw: {**{k: v for k, v in W19.items()}, **kw}
if "Fj3" in G: show("F → Fj3 (3개 결합 Ridge)", sw(F=0, Fj3=.21))
if "Fj4" in G:
    show("F → Fj4 (4개 결합 Ridge)", sw(F=0, Fj4=.21))
    show("F → Fj4, 몫 0.21→0.25 (A·D·T에서)", dict(C=.15, P=.15, A=.1166, D=.1802, T=.1632, Fj4=.24))
if "X" in G:
    FK = "Fj4" if "Fj4" in G else ("Fj3" if "Fj3" in G else "F")
    for wx in (.07, .105, .14, .175):       # 특징 몫 0.70 안에서 X의 몫 10~25%, 나머지는 비례 축소
        r = (.70 - wx) / .70; show(f"+ X {wx / .70:.0%} (특징 몫 안, F 그대로)", dict(C=.15, P=.15, A=.12375 * r, D=.19125 * r, T=.175 * r, F=.21 * r, X=wx))
    for wx in (.105, .14):
        r = (.70 - wx) / .70; show(f"+ X {wx / .70:.0%}, F → {FK}", {"C": .15, "P": .15, "A": .12375 * r, "D": .19125 * r, "T": .175 * r, FK: .21 * r, "X": wx})
    show("X가 D의 절반을 대신", dict(C=.15, P=.15, A=.12375, D=.095, T=.175, F=.21, X=.09625))
    print("X와 다른 성분의 잔차상관:", " ".join(f"{k} {np.corrcoef(G['X'][0] - y, v[0] - y)[0, 1]:.3f}" for k, v in G.items() if k != "X"))
if "--save" in sys.argv:
    i = sys.argv.index("--save"); name = sys.argv[i + 1]; ws = [float(v) for v in sys.argv[i + 2:i + 9]]; fk = sys.argv[i + 9] if len(sys.argv) > i + 9 else "Fj4"
    w = {k: v for k, v in zip(("C", "P", "A", "D", "T", fk, "X"), ws) if v}; o, t = show(f"저장: {name}", w)
    pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print(f"→ submissions/{name} {w} | test 평균 {t.mean():.2f} std {t.std():.2f}")
