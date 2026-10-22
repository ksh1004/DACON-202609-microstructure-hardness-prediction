"""v30n + DRT 경로 깊이 디코더 구성원(NkX): Nk와 평균 / 별도 구성원 — 중첩 가중(P 고정).
사용: python b46_neckx.py [--save 이름 배율 프리셋]"""
import sys, io, contextlib; sys.path.insert(0, "analysis"); _argv46 = sys.argv; sys.argv = ["b44_depth.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b44_depth.py", encoding="utf-8").read())
sys.argv = _argv46
G["NkX"] = grp([E + f"daneckx_all_ff{f}{s}" for f in FF]); b30.G["NkX"] = G["NkX"]
G["Nm"] = tuple((a_ + b_) / 2 for a_, b_ in zip(G["Nk"], G["NkX"])); b30.G["Nm"] = G["Nm"]
print(f"구성원 OOF: Nk {rm(G['Nk'][0]):.3f} | NkX {rm(G['NkX'][0]):.3f} | (Nk+NkX)/2 {rm(G['Nm'][0]):.3f} | Nk-NkX 잔차상관 {np.corrcoef(G['Nk'][0] - y, G['NkX'][0] - y)[0, 1]:.3f}")
W30 = P5["v30n: v29b + Nk 재조정 0.5"]; b30.base, b30.bt = b30.bl(W30, 0, 1.03), b30.bl(W30, 1, 1.03)
o0, _ = nested_fixP(W30); print(f"  [v30n 구성원 중첩] {rm(o0):.3f}")
P7 = {"v30n (기준)": W30}
for nm, Wx in (("v31c: Nk→(Nk+NkX)/2", sw(W30, "Nk", "Nm")), ("v31d: v30n + NkX 별도", {**W30, "NkX": 0.0})):
    o_, Wf_ = nested_fixP(Wx if "별도" not in nm else {**Wx, "NkX": 0.04}); print(f"  [{nm} 중첩] {rm(o_):.3f} | 전체 적합 {dict((k, round(float(v), 3)) for k, v in Wf_.items())}")
    P7[nm] = Wx; P7[nm + " + 재조정 0.5"] = mid_w(Wx, Wf_, 0.5)
w18 = wg(18.0)
if "--save" not in sys.argv:
    for nm, w in P7.items():
        o, _ = show(nm, w); print(f"      Private형 {rm(o, w=w18):.3f}")
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", P7[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, {a: round(float(b), 4) for a, b in P7[pre].items()})
