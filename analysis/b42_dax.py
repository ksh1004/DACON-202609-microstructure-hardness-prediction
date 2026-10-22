"""v30 후보: v29b의 F 경로 16보기 구성원 FDm=(DINOv2_F + DA_F)/2 에 DRT 경로 Depth-Anything(DAx)을 더함.
사용: python b42_dax.py [--save 이름 배율 프리셋]"""
import sys, io, contextlib; sys.path.insert(0, "analysis"); _argv42 = sys.argv; sys.argv = ["b39_da.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b39_da.py", encoding="utf-8").read())
sys.argv = _argv42
G["DAx"] = grp([E + f"da16x_drt10b_L01234_ff{f}{s}" for f in FF]); b30.G["DAx"] = G["DAx"]
G["M3"] = tuple((a_ + b_ + c_) / 3 for a_, b_, c_ in zip(G["FoL"], G["DAo"], G["DAx"])); b30.G["M3"] = G["M3"]
G["DAm"] = tuple((a_ + b_) / 2 for a_, b_ in zip(G["DAo"], G["DAx"])); b30.G["DAm"] = G["DAm"]
G["M2h"] = tuple(.5 * a_ + .25 * b_ + .25 * c_ for a_, b_, c_ in zip(G["FoL"], G["DAo"], G["DAx"])); b30.G["M2h"] = G["M2h"]
print("구성원 OOF: " + " | ".join(f"{n} {rm(G[n][0]):.3f}" for n in ("FoL", "DAo", "DAx", "FDm", "DAm", "M3", "M2h")))
W29 = P3["v29b: FoL→(FoL+DAo)/2 + 재조정 0.5"]; b30.base, b30.bt = b30.bl(W29, 0, 1.03), b30.bl(W29, 1, 1.03)
P4 = {"v29b (기준)": W29}
for nm, mem in (("v30a: FDm→(FoL+DAo+DAx)/3", "M3"), ("v30b: FDm→(DAo+DAx)/2", "DAm"), ("v30c: FDm→FoL/2+DAo/4+DAx/4", "M2h")):
    P4[nm] = sw(W29, "FDm", mem); o_, Wf_ = nested_fixP(P4[nm]); print(f"  [{nm} 중첩·P고정] OOF {rm(o_):.3f} | 전체 적합 {dict((k, round(float(v), 3)) for k, v in Wf_.items())}")
    P4[nm + " + 재조정 0.5"] = mid_w(P4[nm], Wf_, 0.5)
w18 = wg(18.0)
if "--save" not in sys.argv:
    for nm, w in P4.items():
        o, _ = show(nm, w); print(f"      Private형 {rm(o, w=w18):.3f} | 가운데 {rm(o, mid):.3f} 양끝 {rm(o, ~mid):.3f}")
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", P4[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, {a: round(float(b), 4) for a, b in P4[pre].items()})
