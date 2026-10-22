import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, _ = grp(CNN); qo, _ = grp([E + f"dnpp0.25_ff{s}" for s in FF]); po = oof(E + "mr_v14_pp_sc")
o = .15*co+.55*qo+.3*po; r = y - o
qs = np.quantile(o, [0, .05, .1, .25, .5, .75, .9, .95, 1])
print("예측값 구간별: 실제−예측 평균 (±표준오차)")
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (o >= lo) & (o <= hi)
    print(f"  예측[{lo:5.1f},{hi:5.1f}] n={m.sum():3d} 실제−예측 {r[m].mean():+6.2f} ± {r[m].std()/np.sqrt(m.sum()):.2f} | 구간 내 RMSE {np.sqrt((r[m]**2).mean()):5.2f}")
