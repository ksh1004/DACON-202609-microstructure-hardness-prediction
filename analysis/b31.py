import sys, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    from b30 import *
G["Tj"] = grp([E + f"tipsj_{TAG}_ff{f}{s}" for f in FF])
print("Tm", round(rm(G["Tm"][0]), 3), "Tj", round(rm(G["Tj"][0]), 3), "Tx", round(rm(G["Tx"][0]), 3))
show("v25a", W25); show("T: 평균 → 결합", {**{k: v for k, v in W25.items() if k != "Tm"}, "Tj": W25["Tm"]})
