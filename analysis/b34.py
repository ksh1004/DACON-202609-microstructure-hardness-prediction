"""v27 후보: v26b에서 고정 CNN 구성원을 U·DRT·F(ft1b) 세 경로 예측 평균(Jm3)으로 + 가중치 재조정(측정 블록 P 고정).
사용: python b34.py [--save 이름 배율 프리셋]"""
import sys, io, contextlib; sys.path.insert(0, "analysis")
_argv = sys.argv; sys.argv = ["b32.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b32.py", encoding="utf-8").read())
sys.argv = _argv
import b30
from scipy.optimize import nnls
G["Jf"] = grp([E + f"cnnfj4_ft1b_ff{f}{s}" for f in FF]); b30.G["Jf"] = G["Jf"]
G["Jm3"] = tuple((a_ + b_ + c_) / 3 for a_, b_, c_ in zip(G["J"], G["Jx"], G["Jf"])); b30.G["Jm3"] = G["Jm3"]
W26 = PRE["D→Fo, T→세 경로 평균"]; b30.base, b30.bt = b30.bl(W26, 0, 1.03), b30.bl(W26, 1, 1.03)
print("구성원 OOF: " + " | ".join(f"{n} {rm(G[n][0]):.3f}" for n in ("P", "A", "Jm", "Jf", "Jm3", "Fo", "Tm3")))
W27 = sw(W26, "Jm", "Jm3")
G["FoL"] = grp([E + f"mv16f_ft1b_L01234_ff{f}{s}" for f in FF]); b30.G["FoL"] = G["FoL"]  # F 경로 DINOv2 16보기, 층 9~17
print(f"FoL(층 9~17) OOF {rm(G['FoL'][0]):.3f} vs Fo {rm(G['Fo'][0]):.3f}")
W28 = sw(W26, "Fo", "FoL")
G["FoL3"] = grp([E + f"mv16f_ft1b_L3to17_ff{f}{s}" for f in FF]); b30.G["FoL3"] = G["FoL3"]  # 층 3~17 (8개)
print(f"FoL3(층 3~17) OOF {rm(G['FoL3'][0]):.3f}")
W28b = sw(W26, "Fo", "FoL3")
G["FoLm"] = tuple((a_ + b_) / 2 for a_, b_ in zip(G["FoL"], G["FoL3"])); b30.G["FoLm"] = G["FoLm"]  # 층 9~17과 3~17 예측 평균
print(f"FoLm((층9~17 + 층3~17)/2) OOF {rm(G['FoLm'][0]):.3f}")
W28c = sw(W26, "Fo", "FoLm")
def nested_fixP(W, k=1.03):
    names = list(W); X = np.stack([G[n][0] for n in names], 1); iP = names.index("P"); oth = [i for i in range(len(names)) if i != iP]; wP = W["P"]
    def fitw(Xs, ys):
        r = (ys - c) - wP * (Xs[:, iP] - c); w, _ = nnls(Xs[:, oth] - c, r); full = np.zeros(len(names)); full[iP] = wP; full[oth] = w / w.sum() * (1 - wP); return full
    o = np.zeros(len(y))
    for f in range(5):
        t, v = fold != f, fold == f; o[v] = c + k * (X[v] @ fitw(X[t], y[t]) - c)
    return o, dict(zip(names, np.round(fitw(X, y), 4)))
mid_w = lambda W, Wf, a: {n: (1 - a) * W[n] + a * Wf[n] for n in W}
PRE2 = {"v26b": W26, "v27 (CNN 세 경로)": W27}
o, Wf = nested_fixP(W27); print(f"\n[중첩·P고정 가중 적합] OOF {rm(o):.3f} | 전체 적합 가중 {Wf}")
for a in (0.5, 1.0): PRE2[f"v27 + 재조정 {a:.1f}"] = mid_w(W27, Wf, a)
o26, Wf26 = nested_fixP(W26); print(f"[v26b 구성원, 중첩·P고정] OOF {rm(o26):.3f} | 전체 적합 가중 {Wf26}")
for a in (0.5, 1.0): PRE2[f"v26b + 재조정 {a:.1f}"] = mid_w(W26, Wf26, a)
PRE2["v28 (Fo→층 9~17)"] = W28
o28, Wf28 = nested_fixP(W28); print(f"[v28 구성원, 중첩·P고정] OOF {rm(o28):.3f} | 전체 적합 가중 {Wf28}")
for a in (0.5, 1.0): PRE2[f"v28 + 재조정 {a:.1f}"] = mid_w(W28, Wf28, a)
PRE2["v28b (Fo→층 3~17)"] = W28b
o28b, Wf28b = nested_fixP(W28b); print(f"[v28b 구성원, 중첩·P고정] OOF {rm(o28b):.3f} | 전체 적합 가중 {Wf28b}")
for a in (0.5, 1.0): PRE2[f"v28b + 재조정 {a:.1f}"] = mid_w(W28b, Wf28b, a)
o28c, Wf28c = nested_fixP(W28c); print(f"[v28c 구성원, 중첩·P고정] OOF {rm(o28c):.3f} | 전체 적합 가중 {Wf28c}")
PRE2["v28c + 재조정 0.5"] = mid_w(W28c, Wf28c, 0.5)
if "--save" not in sys.argv:
    for nm, w in PRE2.items(): show(nm, w); print("    ", {k_: round(v, 4) for k_, v in w.items()})
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", PRE2[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, PRE2[pre])
