"""마지막 후보: J3 + F 경로 고정 CNN 4종(0.125씩) 결합(J3c). 블렌드에서 CNN 구성원 Jm 유지 / 제거 비교 (중첩·P 고정).
사용: python b53_j3c.py [--save 이름 배율 프리셋]"""
import sys, io, contextlib; sys.path.insert(0, "analysis"); _argv53 = sys.argv; sys.argv = ["b48_joint3.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b48_joint3.py", encoding="utf-8").read())
sys.argv = _argv53
G["J3c"] = grp([E + f"joint3c_ff{f}{s}" for f in FF]); b30.G["J3c"] = G["J3c"]
print(f"구성원 OOF: J3 {rm(G['J3'][0]):.3f} | J3c {rm(G['J3c'][0]):.3f} | Jm {rm(G['Jm'][0]):.3f}")
W32 = P8["v32b: v29b에서 FDm→J3 (Nk 없음) + 재조정 0.5"]; b30.base, b30.bt = b30.bl(W32, 0, 1.03), b30.bl(W32, 1, 1.03)
o0, _ = nested_fixP(W32); print(f"  [v32b 중첩] {rm(o0):.3f}")
P13 = {"v32b (기준)": W32}
Wa = sw(W32, "J3", "J3c"); oa, Wfa = nested_fixP(Wa); print(f"  [J3→J3c, Jm 유지 중첩] {rm(oa):.3f} | 전체 적합 {dict((k, round(float(v), 3)) for k, v in Wfa.items())}")
Wb = {k: v for k, v in Wa.items() if k != "Jm"}; Wb["J3c"] += W32["Jm"]; ob, Wfb = nested_fixP(Wb); print(f"  [J3→J3c, Jm 제거 중첩] {rm(ob):.3f} | 전체 적합 {dict((k, round(float(v), 3)) for k, v in Wfb.items())}")
P13["v37a: J3→J3c (Jm 유지)"] = Wa; P13["v37a + 재조정 0.5"] = mid_w(Wa, Wfa, 0.5)
P13["v37b: J3→J3c (Jm 제거)"] = Wb; P13["v37b + 재조정 0.5"] = mid_w(Wb, Wfb, 0.5)
w18 = wg(18.0)
if "--save" not in sys.argv:
    for nm, w in P13.items():
        o, _ = show(nm, w); print(f"      Private형 {rm(o, w=w18):.3f} | 가운데 {rm(o, mid):.3f} 양끝 {rm(o, ~mid):.3f}")
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", P13[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, {a: round(float(b), 4) for a, b in P13[pre].items()})
