import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, ct = grp(CNN); qo, qt = grp([E + f"dnpp0.25_ff{s}" for s in FF]); po, pt = oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")
w18 = np.exp(-0.5 * ((y - y.mean()) / 18.15) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
base = .15 * co + .55 * qo + .3 * po; bt = .15 * ct + .55 * qt + .3 * pt
rep("DINO+상풀링 ff4 (v15a의 DINO 그룹)", qo)
for g in ("0.05", "0.15"):
    mo, mt = grp([E + f"dnppm{g}_ff{s}" for s in FF]); rep(f"  + 혼합 기여량 G{g}", mo, qo)
rep("v15a 구성", base); print(f"{'':40s} pw18(Private형) {np.sqrt(np.average((base - y) ** 2, weights=w18)):.3f}")
for g in ("0.05", "0.15"):
    mo, mt = grp([E + f"dnppm{g}_ff{s}" for s in FF])
    for wc, wd, wp in ((.15, .55, .3), (.1, .6, .3), (.15, .65, .2)):
        o = wc * co + wd * mo + wp * po; t = wc * ct + wd * mt + wp * pt
        rep(f"G{g} {wc}C+{wd}Dm+{wp}P", o, base); print(f"{'':40s} pw18 {np.sqrt(np.average((o - y) ** 2, weights=w18)):.3f} | test RMS차(v15a 대비) {np.sqrt(np.mean((t - bt) ** 2)):.3f}")
