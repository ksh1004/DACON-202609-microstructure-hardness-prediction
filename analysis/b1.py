import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, ct = grp(CNN); do, dt = grp([E + f"dn2b_fflowpass{s}_low_L11-17" for s in FF])
po, pt = oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")
base = .15 * co + .55 * do + .3 * po; bt = .15 * ct + .55 * dt + .3 * pt
rep("DINO ff4 (hetero3)", do); 
for g in ("0.25", "0.5"):
    qo, qt = grp([E + f"dnpp{g}_ff{s}" for s in FF]); rep(f"DINO+상풀링{g} ff4", qo, do)
rep("v14c 구성", base)
for g in ("0.25", "0.5"):
    qo, qt = grp([E + f"dnpp{g}_ff{s}" for s in FF])
    for wc, wd, wp in ((.15, .55, .3), (.15, .65, .2), (.15, .85, 0), (.2, .8, 0), (.1, .6, .3)):
        o = wc * co + wd * qo + wp * po; t = wc * ct + wd * qt + wp * pt
        rep(f"g{g} {wc}C+{wd}Dpp+{wp}P", o, base); print(f"{'':40s} test RMS차(v14c 대비) {np.sqrt(np.mean((t - bt) ** 2)):.3f}, test std {t.std():.2f}")
