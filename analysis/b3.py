import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, ct = grp(CNN); do, dt = grp([E + f"dn2b_fflowpass{s}_low_L11-17" for s in FF]); qo, qt = grp([E + f"dnpp0.25_ff{s}" for s in FF])
po, pt = oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")
def wts(sd): 
    w = np.exp(-0.5 * ((y - y.mean()) / sd) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); return w
w18 = wts(18.15)
def m(o): 
    e = (o - y) ** 2; return np.sqrt(e.mean()), np.sqrt(np.average(e, weights=w13)), np.sqrt(np.average(e, weights=w18))
print("구성                         OOF    pw13(Public형)  pw18(Private형)")
rows = [("v12b  .2C+.8D", .2 * co + .8 * do), ("v14b  .1C+.3D+.6P", .1 * co + .3 * do + .6 * po), ("v14c  .15C+.55D+.3P", .15 * co + .55 * do + .3 * po)]
for wc in (0.1, 0.15):
    for wp in (0.2, 0.3, 0.4, 0.5, 0.6):
        rows.append((f"신규 {wc}C+{1 - wc - wp:.2f}Dpp+{wp}P", wc * co + (1 - wc - wp) * qo + wp * po))
for n, o in rows: print(f"{n:28s} {m(o)[0]:.3f}   {m(o)[1]:.3f}          {m(o)[2]:.3f}")
