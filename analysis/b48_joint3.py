"""v32 후보: F 경로 16보기 구성원을 'DINOv2 + Depth-Anything 백본 + DA 깊이 디코더' 결합 Ridge(J3, 디코더 상대분산 0.3)로.
사용: python b48_joint3.py [--save 이름 배율 프리셋]"""
import sys, io, contextlib; sys.path.insert(0, "analysis"); _argv48 = sys.argv; sys.argv = ["b44_depth.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b44_depth.py", encoding="utf-8").read())
sys.argv = _argv48
GN = "0.3"
G["J3"] = grp([E + f"joint3_g{GN}_ff{f}{s}" for f in FF]); b30.G["J3"] = G["J3"]
G["JDA"] = grp([E + f"jointDA_ft1b_L01234_ff{f}{s}" for f in FF]); b30.G["JDA"] = G["JDA"]
print("구성원 OOF: " + " | ".join(f"{n} {rm(G[n][0]):.3f}" for n in ("FDm", "JDA", "J3", "Nk")) + f" | J3-FDm 잔차상관 {np.corrcoef(G['J3'][0] - y, G['FDm'][0] - y)[0, 1]:.3f}")
W30 = P5["v30n: v29b + Nk 재조정 0.5"]; W29 = P5["v29b (기준)"]; b30.base, b30.bt = b30.bl(W30, 0, 1.03), b30.bl(W30, 1, 1.03)
o0, _ = nested_fixP(W30); print(f"  [v30n 중첩] {rm(o0):.3f}")
P8 = {"v30n (기준)": W30}
cands = (("v32a: v30n에서 FDm→J3 (Nk 유지)", sw(W30, "FDm", "J3")),
         ("v32b: v29b에서 FDm→J3 (Nk 없음)", sw(W29, "FDm", "J3")),
         ("v32c: v30n에서 FDm→JDA (디코더 결합 없음)", sw(W30, "FDm", "JDA")))
for nm, Wx in cands:
    o_, Wf_ = nested_fixP(Wx); print(f"  [{nm} 중첩] {rm(o_):.3f} | 전체 적합 {dict((k, round(float(v), 3)) for k, v in Wf_.items())}")
    P8[nm] = Wx; P8[nm + " + 재조정 0.5"] = mid_w(Wx, Wf_, 0.5)
w18 = wg(18.0)
if "--save" not in sys.argv:
    for nm, w in P8.items():
        o, _ = show(nm, w); print(f"      Private형 {rm(o, w=w18):.3f} | 가운데 {rm(o, mid):.3f} 양끝 {rm(o, ~mid):.3f}")
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", P8[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, {a: round(float(b), 4) for a, b in P8[pre].items()})
