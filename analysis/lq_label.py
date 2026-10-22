"""노이즈 수준(촬영 모드)에 따라 라벨의 분포·예측과의 관계가 달라지는가?"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from scipy import stats
co, _ = grp(CNN); qo, _ = grp([E + f"dnpp0.25_ff{s}" for s in FF]); po = oof(E + "mr_v14_pp_sc"); do, _ = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF])
p = .15 * co + .275 * qo + .425 * do + .15 * po   # v17b OOF
F = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[ids]
print("노이즈 σ 분포 분위(0,10,25,50,75,90,100%):", np.quantile(sig, [0, .1, .25, .5, .75, .9, 1]).round(1))
print(f"\n{'σ 구간':>10s} {'n':>4s} {'y평균':>7s} {'y표준편차':>8s} {'왜도':>6s} {'y 5%':>6s} {'y 95%':>6s} | {'예측std':>7s} {'잔차std':>7s} {'R²':>6s} {'기울기 b':>8s} {'±':>5s} {'편향':>6s}")
edges = [0, 5.5, 7, 8.5, 10, 12, 14, 16, 18, 30]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (sig >= lo) & (sig < hi)
    if m.sum() < 8: continue
    yy, pp = y[m], p[m]; b, a = np.polyfit(pp, yy, 1); se = (yy - a - b * pp).std() / (pp.std() * np.sqrt(m.sum()))
    print(f"{lo:4.1f}~{hi:4.1f} {m.sum():4d} {yy.mean():7.1f} {yy.std():8.2f} {stats.skew(yy):+6.2f} {np.quantile(yy,.05):6.1f} {np.quantile(yy,.95):6.1f} | {pp.std():7.2f} {(yy-pp).std():7.2f} {1-((yy-pp)**2).mean()/yy.var():+6.2f} {b:8.2f} {se:5.2f} {(yy-pp).mean():+6.2f}")
hq = sig < 10
print("\n분산 동일성 검정(Levene) 고품질 vs 저품질 라벨: p = %.4f | 평균 차 t검정 p = %.3f" % (stats.levene(y[hq], y[~hq]).pvalue, stats.ttest_ind(y[hq], y[~hq]).pvalue))
print("KS 검정(분포 동일성): p = %.4f" % stats.ks_2samp(y[hq], y[~hq]).pvalue)
for nm, m in (("고품질", hq), ("저품질", ~hq)):
    print(f"{nm}: n={m.sum()} y 분위(1,5,25,50,75,95,99%) {np.quantile(y[m], [.01,.05,.25,.5,.75,.95,.99]).round(1)} | 예측 분위 {np.quantile(p[m], [.01,.05,.25,.5,.75,.95,.99]).round(1)}")
# 블러·대비 등 다른 품질 지표와 라벨 분산
for c in ("q_blur_index", "q_contrast", "q_brightness", "q_noise_sigma"):
    if c in F:
        v = F[c].values; qs = np.quantile(v, [0, .2, .4, .6, .8, 1]);
        print(f"{c:16s} 5분위별 y표준편차: " + " ".join(f"{y[(v>=qs[i])&(v<=qs[i+1])].std():5.1f}" for i in range(5)) + " | 잔차 표준편차: " + " ".join(f"{(y-p)[(v>=qs[i])&(v<=qs[i+1])].std():5.1f}" for i in range(5)))
print("품질 지표 열:", [c for c in F.columns if c.startswith("q_")])
