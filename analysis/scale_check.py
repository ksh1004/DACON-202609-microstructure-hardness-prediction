"""블렌드가 넓어질수록 예측이 평균으로 수축 → 예측 폭 배율 k(전체/모드별)의 교차검증 이득. 배율은 폴드 밖에서 추정."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, ct = grp(CNN); po, pt = oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")
ao, at = grp([E + f"dnpp0.25_ff{s}" for s in FF]); do, dt = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF]); to, tt = grp([E + f"tips16_dnppm0.05_ff{s}" for s in FF])
fo, ft = grp([E + f"cnnf_{n}_ff{s}" for n in ("dn121", "r50a1", "r50a2") for s in FF])
dm = lambda r, a, b, c: (r * a * .275 / .70, r * a * .425 / .70, r * b, r * c)
def mk(w): return w[0] * co + w[1] * po + w[2] * ao + w[3] * do + w[4] * to + w[5] * fo, w[0] * ct + w[1] * pt + w[2] * at + w[3] * dt + w[4] * tt + w[5] * ft
hq = sig < 10; sgt = pd.read_csv("outputs/features/test_features.csv").set_index("ID").loc[te, "q_noise_sigma"].values; hqt = sgt < 10
for nm, w in (("v15a", (.15, .30, .55, 0, 0, 0)), ("v18a", (.15, .15) + dm(.70, .7, .3, 0)), ("v19a", (.15, .15) + dm(.70, .45, .25, .30)), ("v19b", (0, .15) + dm(.85, .45, .25, .30)), ("v19c", (0, 0) + dm(1, .45, .25, .30))):
    o, t = mk(w); b, a = np.polyfit(o, y, 1)
    bh = np.polyfit(o[hq], y[hq], 1)[0]; bl = np.polyfit(o[~hq], y[~hq], 1)[0]
    print(f"\n[{nm}] OOF {np.sqrt(((o - y) ** 2).mean()):.3f} | 예측 표준편차 OOF {o.std():.2f} / test {t.std():.2f} (실제 {y.std():.2f}) | 기울기(실제=a+b·예측) 전체 {b:.3f}, 고품질 {bh:.3f}, 저품질 {bl:.3f}")
    # 폴드 밖 배율 추정: 전체 배율 / 모드별 배율
    og, om = o.copy(), o.copy()
    for k in range(5):
        tr, va = fold != k, fold == k; mu = o[tr].mean()
        kk = np.polyfit(o[tr] - mu, y[tr] - y[tr].mean(), 1)[0]; og[va] = y[tr].mean() + kk * (o[va] - mu)
        for m in (hq, ~hq):
            mm = o[tr & m].mean(); km = np.polyfit(o[tr & m] - mm, y[tr & m] - y[tr & m].mean(), 1)[0]; om[va & m] = y[tr & m].mean() + km * (o[va & m] - mm)
    rep("   원래", o); rep("   전체 배율(폴드 밖)", og, o); rep("   모드별 배율(폴드 밖)", om, o)
    for kfix in (1.03, 1.05, 1.08):
        ok = o.mean() + kfix * (o - o.mean()); rep(f"   고정 배율 {kfix}", ok, o)
    okl = o.copy(); okl[~hq] = o[~hq].mean() + 1.10 * (o[~hq] - o[~hq].mean()); rep("   저품질만 1.10", okl, o)
