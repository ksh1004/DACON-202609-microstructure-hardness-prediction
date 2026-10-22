"""v29 후보: v28a의 F 경로 DINOv2 16보기 구성원(FoL)을 Depth-Anything-V2-L(DAo, 같은 설정)로 대체 / 평균 / 둘 다 + 가중 재조정(P 고정).
사용: python b39_da.py [--save 이름 배율 프리셋]"""
import sys, io, contextlib; sys.path.insert(0, "analysis"); _argv39 = sys.argv; sys.argv = ["b34.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b34.py", encoding="utf-8").read())
sys.argv = _argv39
G["DAo"] = grp([E + f"da16f_ft1b_L01234_ff{f}{s}" for f in FF]); b30.G["DAo"] = G["DAo"]
G["FDm"] = tuple((a_ + b_) / 2 for a_, b_ in zip(G["FoL"], G["DAo"])); b30.G["FDm"] = G["FDm"]
print(f"구성원 OOF: FoL {rm(G['FoL'][0]):.3f} | DAo {rm(G['DAo'][0]):.3f} | (FoL+DAo)/2 {rm(G['FDm'][0]):.3f} | 잔차상관 {np.corrcoef(G['FoL'][0] - y, G['DAo'][0] - y)[0, 1]:.3f}")
W28a = PRE2["v28 + 재조정 0.5"]; b30.base, b30.bt = b30.bl(W28a, 0, 1.03), b30.bl(W28a, 1, 1.03)
P3 = {"v28a (기준)": W28a, "v29a: FoL→DAo": sw(W28a, "FoL", "DAo"), "v29b: FoL→(FoL+DAo)/2": sw(W28a, "FoL", "FDm")}
for nm in list(P3)[1:]:
    o_, Wf_ = nested_fixP(P3[nm]); print(f"  [{nm} 중첩·P고정] OOF {rm(o_):.3f} | 전체 적합 {Wf_}")
    P3[nm + " + 재조정 0.5"] = mid_w(P3[nm], Wf_, 0.5)
o_, Wf_ = nested_fixP({**W28a, "DAo": 0.0}); print(f"  [FoL·DAo 둘 다, 중첩·P고정] OOF {rm(o_):.3f} | 전체 적합 {Wf_}")
if "--save" not in sys.argv:
    for nm, w in P3.items(): show(nm, w)
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", P3[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, {a: round(float(b), 4) for a, b in P3[pre].items()})
