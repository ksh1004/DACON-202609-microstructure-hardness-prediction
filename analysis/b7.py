import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, _ = grp(CNN); qo, _ = grp([E + f"dnpp0.25_ff{s}" for s in FF]); po = oof(E + "mr_v14_pp_sc")
o = .15 * co + .55 * qo + .3 * po; r = o - y
print("== 노이즈 σ 구간별 (v15a OOF)")
bins = [0, 4, 7, 10, 12, 14, 16, 18, 30]
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (sig >= lo) & (sig < hi)
    if m.sum(): print(f"  σ[{lo:2d},{hi:2d}) n={m.sum():3d} 경도 평균 {y[m].mean():6.1f} std {y[m].std():5.1f} | RMSE {np.sqrt((r[m]**2).mean()):5.2f} 치우침 {r[m].mean():+5.2f} | 예측 std {o[m].std():5.1f} R² {1-(r[m]**2).mean()/y[m].var():.2f}")
print("== 경도 구간별")
qs = np.quantile(y, [0, .1, .25, .5, .75, .9, 1])
for lo, hi in zip(qs[:-1], qs[1:]):
    m = (y >= lo) & (y <= hi)
    print(f"  경도[{lo:5.1f},{hi:5.1f}] n={m.sum():3d} RMSE {np.sqrt((r[m]**2).mean()):5.2f} 치우침 {r[m].mean():+6.2f} | 고품질 {np.sqrt((r[m&(sig<10)]**2).mean()):5.2f} 저품질 {np.sqrt((r[m&(sig>=10)]**2).mean()):5.2f}")
print("보정 기울기(y ~ a+b·pred): 전체 %.3f, 고품질 %.3f, 저품질 %.3f" % tuple(np.polyfit(o[m], y[m], 1)[0] for m in (np.ones(len(y), bool), sig < 10, sig >= 10)))
print("제곱오차 상위 5%%(25장)가 전체 제곱오차에서 차지하는 비중 %.2f, 그중 저품질 %d장" % (np.sort(r**2)[-25:].sum() / (r**2).sum(), (sig[np.argsort(r**2)[-25:]] >= 10).sum()))
