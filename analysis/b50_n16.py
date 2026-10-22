"""v34 후보: v32b의 결합 구성원 J3(디코더 8보기)를 J3n16(디코더 16보기 = D4 × 격자 이동 2)으로.
사용: python b50_n16.py [--save 이름 배율 프리셋]"""
import sys, io, contextlib; sys.path.insert(0, "analysis"); _argv50 = sys.argv; sys.argv = ["b48_joint3.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b48_joint3.py", encoding="utf-8").read())
sys.argv = _argv50
G["J3n"] = grp([E + f"joint3n16_ff{f}{s}" for f in FF]); b30.G["J3n"] = G["J3n"]
print(f"구성원 OOF: J3 {rm(G['J3'][0]):.3f} | J3n16 {rm(G['J3n'][0]):.3f} | 잔차상관 {np.corrcoef(G['J3'][0] - y, G['J3n'][0] - y)[0, 1]:.3f}")
W32 = P8["v32b: v29b에서 FDm→J3 (Nk 없음) + 재조정 0.5"]; b30.base, b30.bt = b30.bl(W32, 0, 1.03), b30.bl(W32, 1, 1.03)
o0, _ = nested_fixP(W32); print(f"  [v32b 중첩] {rm(o0):.3f}")
P10 = {"v32b (기준)": W32}
Wx = sw(W32, "J3", "J3n"); o_, Wf_ = nested_fixP(Wx); print(f"  [v34: J3→J3n16 중첩] {rm(o_):.3f} | 전체 적합 {dict((k, round(float(v), 3)) for k, v in Wf_.items())}")
P10["v34: J3→J3n16"] = Wx
w18 = wg(18.0)
if "--save" not in sys.argv:
    for nm, w in P10.items():
        o, _ = show(nm, w); print(f"      Private형 {rm(o, w=w18):.3f} | 가운데 {rm(o, mid):.3f} 양끝 {rm(o, ~mid):.3f}")
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", P10[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, {a: round(float(b), 4) for a, b in P10[pre].items()})
