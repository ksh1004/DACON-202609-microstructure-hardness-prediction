"""v30n + Depth-Anything 518 입력 구성원(D5): F 경로 16보기 그룹에 넣기(3자 평균) / 별도 구성원 — 중첩 가중(P 고정).
사용: python b45_da518.py [--save 이름 배율 프리셋]"""
import sys, io, contextlib; sys.path.insert(0, "analysis"); _argv45 = sys.argv; sys.argv = ["b44_depth.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b44_depth.py", encoding="utf-8").read())
sys.argv = _argv45
G["D5"] = grp([E + f"da16f518_ft1b_L01234_ff{f}{s}" for f in FF]); b30.G["D5"] = G["D5"]
G["FD3"] = tuple((a_ + b_ + c_) / 3 for a_, b_, c_ in zip(G["FoL"], G["DAo"], G["D5"])); b30.G["FD3"] = G["FD3"]
print(f"구성원 OOF: FDm {rm(G['FDm'][0]):.3f} | D5(518) {rm(G['D5'][0]):.3f} | (FoL+DAo+D5)/3 {rm(G['FD3'][0]):.3f} | D5-DAo 잔차상관 {np.corrcoef(G['D5'][0] - y, G['DAo'][0] - y)[0, 1]:.3f}")
W30 = P5["v30n: v29b + Nk 재조정 0.5"]; b30.base, b30.bt = b30.bl(W30, 0, 1.03), b30.bl(W30, 1, 1.03)
P6 = {"v30n (기준)": W30}
o0, _ = nested_fixP(W30); print(f"  [v30n 구성원 중첩] {rm(o0):.3f}")
for nm, Wx in (("v31a: FDm→(FoL+DAo+D5)/3", sw(W30, "FDm", "FD3")), ("v31b: v30n + D5 별도", {**W30, "D5": 0.0})):
    o_, Wf_ = nested_fixP(Wx if "별도" not in nm else {**Wx, "D5": 0.05}); print(f"  [{nm} 중첩] {rm(o_):.3f} | 전체 적합 {dict((k, round(float(v), 3)) for k, v in Wf_.items())}")
    P6[nm] = Wx; P6[nm + " + 재조정 0.5"] = mid_w(Wx, Wf_, 0.5)
w18 = wg(18.0)
if "--save" not in sys.argv:
    for nm, w in P6.items():
        o, _ = show(nm, w); print(f"      Private형 {rm(o, w=w18):.3f}")
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", P6[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, {a: round(float(b), 4) for a, b in P6[pre].items()})
