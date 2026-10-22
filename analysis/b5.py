import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, ct = grp(CNN); qo, qt = grp([E + f"dnpp0.25_ff{s}" for s in FF])
P = {s: oof(E + ("mr_v14_pp_sc" if s == 48 else f"mr_v14_pp_sc_ff{s}")) for s in FF}
w18 = np.exp(-0.5 * ((y - y.mean()) / 18.15) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
r = lambda o, w=None: np.sqrt(np.average((o - y) ** 2, weights=w))
wps = (0, .1, .2, .3, .4, .5, .6, .7)
for nm, w in (("OOF", None), ("pw13(Public형)", w13), ("pw18(Private형)", w18)):
    print(f"\n[{nm}]  P 비중 → " + "  ".join(f"{x:>6}" for x in wps))
    for s in FF:
        print(f"  P=σ{s:<3d}          " + "  ".join(f"{r(.15 * co + (.85 - x) * qo + x * P[s], w):6.3f}" for x in wps))
    print("  σ32·64·96 평균    " + "  ".join(f"{np.mean([r(.15 * co + (.85 - x) * qo + x * P[s], w) for s in (32, 64, 96)]):6.3f}" for x in wps))
