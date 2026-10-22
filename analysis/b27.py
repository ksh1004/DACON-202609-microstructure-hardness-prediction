"""v23 후보: v22a(미세조정 CNN 없음)에서 모든 Ridge 구성원을 고정 벌점(--alpha-rel)으로 다시 맞춘 블렌드 평가·저장.
사용: python b27.py [접미사=_a0.01] [--save 이름 배율 [P도새로=1]]"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
SUF = next((a for a in sys.argv[1:] if a.startswith("_a")), "_a0.01"); c = y.mean()
def G(s):
    return dict(P=(oof(E + f"mr_v14_pp_sc{s}"), tp(E + f"mr_v14_pp_sc{s}")), A=grp([E + f"dnpp0.25_ff{f}{s}" for f in FF]), D=grp([E + f"mv16_dnppm0.05_ff{f}{s}" for f in FF]),
                T=grp([E + f"tips16_dnppm0.05_ff{f}{s}" for f in FF]), J=grp([E + f"cnnfj4_ff{f}{s}" for f in FF]))
O, Nw = G(""), G(SUF); W = dict(P=.15 / .85, A=.12375 / .85, D=.19125 / .85, T=.175 / .85, J=.21 / .85)
bl = lambda g, i, w=W: sum(v * g[k][i] for k, v in w.items())
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166, w18 = wg(16.6), wg(18.0)
rm = lambda o, m=None, w=None: np.sqrt(np.average(((o - y) ** 2) if m is None else ((o - y)[m] ** 2), weights=w))
Lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
base = c + 1.05 * (bl(O, 0) - c); bt = c + 1.05 * (bl(O, 1) - c)          # v22a
print("구성원 OOF (기존 → 새 벌점): " + " | ".join(f"{k} {rm(O[k][0]):.3f}→{rm(Nw[k][0]):.3f}" for k in W))
R = np.corrcoef([Nw[k][0] - y for k in W]); R0 = np.corrcoef([O[k][0] - y for k in W]); ks = list(W)
print("구성원 간 잔차상관 평균: 기존 %.3f → 새 %.3f" % (R0[np.triu_indices(5, 1)].mean(), R[np.triu_indices(5, 1)].mean()))
def show(nm, o, t):
    d = (o - y) ** 2 - (base - y) ** 2; cv = np.cov(y, o)
    print(f"{nm:40s} OOF {rm(o):.3f} pw13 {rm(o, w=w13):.3f} Public꼴 {rm(o, w=w166):.3f} Private꼴 {rm(o, w=w18):.3f} | 고 {rm(o, ~Lq):.3f} 저 {rm(o, Lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | 가운데 {d[mid].sum() / 500:+.2f} 양끝 {d[~mid].sum() / 500:+.2f} | 기울기 {cv[0, 1] / cv[1, 1]:.3f} | test std {t.std():.2f} RMS차 {np.sqrt(((t - bt) ** 2).mean()):.2f}")
show("v22a (기준, ×1.05)", base, bt)
show("v22a 구성원 그대로 ×1.0", bl(O, 0), bl(O, 1))
for k in (1.0, 1.02, 1.03, 1.05): show(f"새 벌점 전부 ×{k}", c + k * (bl(Nw, 0) - c), c + k * (bl(Nw, 1) - c))
Mx = {**Nw, "P": O["P"]}
for k in (1.0, 1.03): show(f"새 벌점(측정 블록 P만 기존) ×{k}", c + k * (bl(Mx, 0) - c), c + k * (bl(Mx, 1) - c))
for nm, w in (("P 0.10로 줄임", dict(P=.10)), ("P 0", dict(P=0.0)), ("J ×1.3", dict(J=W["J"] * 1.3)), ("T ×1.3", dict(T=W["T"] * 1.3))):
    w2 = {**W, **w}; s_ = sum(v for k_, v in w2.items() if k_ not in w); r = (1 - sum(w.values())) / s_; w2 = {k_: (v if k_ in w else v * r) for k_, v in w2.items()}
    show(f"새 벌점 ×1.03, {nm}", c + 1.03 * (bl(Nw, 0, w2) - c), c + 1.03 * (bl(Nw, 1, w2) - c))
if "--save" in sys.argv:
    i = sys.argv.index("--save"); name, k = sys.argv[i + 1], float(sys.argv[i + 2]); g = Nw if (len(sys.argv) <= i + 3 or sys.argv[i + 3] == "1") else Mx
    t = c + k * (bl(g, 1) - c); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); show(f"저장: {name}", c + k * (bl(g, 0) - c), t)
