"""v26 후보: v25a에서 DINOv2 구성원을 미세조정 DRUNet(ft1b) 경로 포함 결합으로, TIPSv2를 세 경로 평균으로.
사용: python b32.py [--tag ft1b] [--save 이름 배율 프리셋]"""
import sys, os, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    from b30 import *
FT = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "ft1b"
ok = lambda n: all(os.path.exists(E + n.format(f) + "/oof.csv") for f in FF)
for k, n in (("J4", f"mv16j4_{FT}_ff{{}}{s}"), ("DXF", f"mv16dxf_{FT}_ff{{}}{s}"), ("DF", f"mv16df_{FT}_ff{{}}{s}"), ("Fo", f"mv16f_{FT}_ff{{}}{s}"), ("Tf", f"tips16_ft{FT}_ff{{}}{s}")):
    if ok(n): G[k] = grp([E + n.format(f) for f in FF])
    else: print("미완료:", k)
if "Tf" in G: G["Tm3"] = tuple((a_ + b_ + c_) / 3 for a_, b_, c_ in zip(G["T"], G["Tx"], G["Tf"])); G["Tfm"] = tuple((a_ + b_) / 2 for a_, b_ in zip(G["T"], G["Tf"]))
print("구성원 OOF: " + " | ".join(f"{n} {rm(G[n][0]):.3f} (고 {rm(G[n][0], ~Lq):.2f} 저 {rm(G[n][0], Lq):.2f})" for n in G if n not in ("T", "J", "Tx", "Jx")))
sw = lambda w, a_, b_: {**{k: v for k, v in w.items() if k != a_}, b_: w[a_]}
PRE = {"v25a": W25}
for d in ("J4", "DXF", "DF", "Fo"):
    if d in G:
        PRE[f"D→{d}"] = sw(W25, "Dj3", d)
        if "Tm3" in G: PRE[f"D→{d}, T→세 경로 평균"] = sw(sw(W25, "Dj3", d), "Tm", "Tm3")
if "--save" not in sys.argv:
    for nm, w in PRE.items(): show(nm, w)
    for d in ("DXF", "DF"):
        if d in G:
            for k in (1.0, 1.05): show(f"D→{d} ×{k}", PRE[f"D→{d}"], k)
else:
    i = sys.argv.index("--save"); name, k, pre = sys.argv[i + 1], float(sys.argv[i + 2]), sys.argv[i + 3]
    o, t = show(f"저장: {name}", PRE[pre], k); pd.DataFrame({"ID": te, "hardness": t}).to_csv(f"submissions/{name}", index=False); print("→", name, PRE[pre])
