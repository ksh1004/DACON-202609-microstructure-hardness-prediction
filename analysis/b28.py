"""v24 후보: v23a(약한 정규화)에서 DINOv2 16보기 구성원 D를 두 디노이즈 경로 결합(Dj = mv16j_ff*_a0.01)으로 교체. 평가·저장.
사용: python b28.py [--save 이름 배율 [가중 프리셋]]"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
s = "_a0.01"; c = y.mean()
G = dict(P=(oof(E + f"mr_v14_pp_sc{s}"), tp(E + f"mr_v14_pp_sc{s}")), A=grp([E + f"dnpp0.25_ff{f}{s}" for f in FF]), D=grp([E + f"mv16_dnppm0.05_ff{f}{s}" for f in FF]),
         T=grp([E + f"tips16_dnppm0.05_ff{f}{s}" for f in FF]), J=grp([E + f"cnnfj4_ff{f}{s}" for f in FF]), Dj=grp([E + f"mv16j_ff{f}{s}" for f in FF]), Dd=grp([E + f"mv16dr_ff{f}{s}" for f in FF]),
         K=grp([E + f"dpnj_ff{f}{s}" for f in FF]))
W23 = dict(P=.15 / .85, A=.12375 / .85, D=.19125 / .85, T=.175 / .85, J=.21 / .85)
wg = lambda sd: np.exp(-0.5 * ((y - y.mean()) / sd) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166 = wg(16.6); w18 = wg(18.0)
rm = lambda o, m=None, w=None: np.sqrt(np.average(((o - y) ** 2) if m is None else ((o - y)[m] ** 2), weights=w)); Lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
bl = lambda w, i, k: c + k * (sum(v * G[n][i] for n, v in w.items()) - c)
base, bt = bl(W23, 0, 1.03), bl(W23, 1, 1.03); v22t = pd.read_csv("submissions/sub_v22a_noC_k105.csv").set_index("ID").loc[te, "hardness"].values
print("구성원 OOF: " + " | ".join(f"{n} {rm(G[n][0]):.3f}" for n in G) + f" | Dj와 D 잔차상관 {np.corrcoef(G['Dj'][0] - y, G['D'][0] - y)[0, 1]:.3f}")
def show(nm, w, k=1.03):
    assert abs(sum(w.values()) - 1) < 1e-9, sum(w.values()); o, t = bl(w, 0, k), bl(w, 1, k); d = (o - y) ** 2 - (base - y) ** 2; cv = np.cov(y, o)
    print(f"{nm:46s} OOF {rm(o):.3f} pw13 {rm(o, w=w13):.3f} Public꼴 {rm(o, w=w166):.3f} Private꼴 {rm(o, w=w18):.3f} | 고 {rm(o, ~Lq):.3f} 저 {rm(o, Lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} {sum(d[fold == f].mean() < 0 for f in range(5))}/5 | 가운데 {d[mid].sum() / 500:+.2f} 양끝 {d[~mid].sum() / 500:+.2f} | 기울기 {cv[0, 1] / cv[1, 1]:.3f} test std {t.std():.2f} | v23a와 {np.sqrt(((t - bt) ** 2).mean()):.2f} v22a와 {np.sqrt(((t - v22t) ** 2).mean()):.2f}")
    return o, t
PRE = {"v23a": W23, "D→Dj": {**{k: v for k, v in W23.items() if k != "D"}, "Dj": W23["D"]},
       "D→Dj, A 절반을 Dj로": {**{k: v for k, v in W23.items() if k not in ("D", "A")}, "A": W23["A"] / 2, "Dj": W23["D"] + W23["A"] / 2},
       "D→Dj, A 전부 Dj로": {**{k: v for k, v in W23.items() if k not in ("D", "A")}, "Dj": W23["D"] + W23["A"]},
       "D→Dj, Dj 몫 ×1.3 (나머지 비례)": None, "D 유지 + Dd(DRUNet만) 별도 .15": None}
w = PRE["D→Dj"]; r = (1 - 1.3 * w["Dj"]) / (1 - w["Dj"]); PRE["D→Dj, Dj 몫 ×1.3 (나머지 비례)"] = {k: (1.3 * v if k == "Dj" else v * r) for k, v in w.items()}
PRE["D 유지 + Dd(DRUNet만) 별도 .15"] = {**{k: v * .85 for k, v in W23.items()}, "Dd": .15}
for nm, w in PRE.items(): show(nm, w)
for k in (1.0, 1.02, 1.05): show(f"D→Dj ×{k}", PRE["D→Dj"], k)
if "--save" in sys.argv:
    i = sys.argv.index("--save"); name, k = sys.argv[i + 1], float(sys.argv[i + 2]); pre = sys.argv[i + 3] if len(sys.argv) > i + 3 else "D→Dj"
    o, t = show(f"저장: {name} ({pre}, ×{k})", PRE[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, {n: round(v, 4) for n, v in PRE[pre].items()})
