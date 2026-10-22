"""v36 후보: J3 + TIPSv2 F 경로(상대 비중 0.5) 한 Ridge 결합(J3t, 층 8~14 / 6~14). TIPS 구성원 Tm3은 중첩 적합이 결정.
사용: python b52_j3t.py [--save 이름 배율 프리셋]"""
import sys, io, contextlib; sys.path.insert(0, "analysis"); _argv52 = sys.argv; sys.argv = ["b48_joint3.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b48_joint3.py", encoding="utf-8").read())
sys.argv = _argv52
for nm in ("t814", "t614"):
    G[f"J3{nm}"] = grp([E + f"joint3{nm}_ff{f}{s}" for f in FF]); b30.G[f"J3{nm}"] = G[f"J3{nm}"]
print("구성원 OOF: " + " | ".join(f"{n} {rm(G[n][0]):.3f}" for n in ("J3", "J3t814", "J3t614", "Tm3")))
W32 = P8["v32b: v29b에서 FDm→J3 (Nk 없음) + 재조정 0.5"]; b30.base, b30.bt = b30.bl(W32, 0, 1.03), b30.bl(W32, 1, 1.03)
o0, _ = nested_fixP(W32); print(f"  [v32b 중첩] {rm(o0):.3f}")
P12 = {"v32b (기준)": W32}
for nm in ("J3t814", "J3t614"):
    Wx = sw(W32, "J3", nm); o_, Wf_ = nested_fixP(Wx); print(f"  [J3→{nm} 중첩] {rm(o_):.3f} | 전체 적합 {dict((k, round(float(v), 3)) for k, v in Wf_.items())}")
    P12[f"v36: J3→{nm}"] = Wx; P12[f"v36: J3→{nm} + 재조정 0.5"] = mid_w(Wx, Wf_, 0.5)
w18 = wg(18.0)
if "--save" not in sys.argv:
    for nm, w in P12.items():
        o, _ = show(nm, w); print(f"      Private형 {rm(o, w=w18):.3f} | 가운데 {rm(o, mid):.3f} 양끝 {rm(o, ~mid):.3f}")
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", P12[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, {a: round(float(b), 4) for a, b in P12[pre].items()})
