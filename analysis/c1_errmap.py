"""v15a OOF 제곱 잔차가 어떤 이미지 유형에 몰려 있는지 진단 (평균 잔차가 아니라 오차 크기)."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from scipy.stats import spearmanr
co, ct = grp(CNN); qo, qt = grp([E + f"dnpp0.25_ff{s}" for s in FF]); po, pt = oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")
o = .15 * co + .55 * qo + .3 * po
rep("v15a 구성", o); rep("  DINO+상풀링 그룹", qo); rep("  측정 블록 P", po); rep("  CNN9", co)
r = o - y; a = np.abs(r); se = r ** 2
F = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[ids]
num = F.select_dtypes("number").replace([np.inf, -np.inf], np.nan); num = num.fillna(num.median())
print("\n== |잔차|와의 Spearman 상위 (전체 / 고품질 / 저품질) ==")
rows = []
hq = sig < 10
for c in num.columns:
    v = num[c].values
    if np.std(v) == 0: continue
    rows.append((c, spearmanr(v, a)[0], spearmanr(v[hq], a[hq])[0], spearmanr(v[~hq], a[~hq])[0], spearmanr(v, r)[0], spearmanr(v, y)[0]))
d = pd.DataFrame(rows, columns=["feat", "abs_all", "abs_hq", "abs_lq", "signed", "rho_y"]).set_index("feat")
print(d.reindex(d.abs_all.abs().sort_values(ascending=False).index).head(25).round(3).to_string())
print("\n== 부호 있는 잔차와의 Spearman 상위 ==")
print(d.reindex(d.signed.abs().sort_values(ascending=False).index).head(12).round(3).to_string())
# 방향: 주방향 각도별 오차
ang = [c for c in num.columns if c.startswith("nui_")]
print("\nnuisance 열:", ang)
coh = num["an_local_coherence"].values
print("\n== 정렬도 5분위별 RMSE (v15a / DINO그룹 / CNN) ==")
qs = np.quantile(coh, [0, .2, .4, .6, .8, 1.0])
for i in range(5):
    m = (coh >= qs[i]) & (coh <= qs[i + 1]) if i == 4 else (coh >= qs[i]) & (coh < qs[i + 1])
    print(f"  coh {qs[i]:.3f}~{qs[i+1]:.3f} n={m.sum():3d} y평균 {y[m].mean():.1f} y표준편차 {y[m].std():.1f} | v15a {np.sqrt(se[m].mean()):.2f} DINO {np.sqrt(((qo-y)[m]**2).mean()):.2f} CNN {np.sqrt(((co-y)[m]**2).mean()):.2f} | 편향 {r[m].mean():+.2f}")
for c in ang:
    th = num[c].values
    print(f"\n== {c}: 범위 {th.min():.2f}~{th.max():.2f}; 정렬도 상위 40% 이미지에서 각도 구간별 RMSE ==")
    top = coh >= np.quantile(coh, .6)
    t = th[top]; rr = r[top]
    lo, hi = t.min(), t.max(); edges = np.linspace(lo, hi, 7)
    for i in range(6):
        m = (t >= edges[i]) & (t <= edges[i + 1])
        if m.sum(): print(f"  {edges[i]:7.2f}~{edges[i+1]:7.2f} n={m.sum():3d} RMSE {np.sqrt((rr[m]**2).mean()):.2f} 편향 {rr[m].mean():+.2f}")
print("\n== 경도 10분위별 RMSE·편향 ==")
qs = np.quantile(y, np.linspace(0, 1, 11))
for i in range(10):
    m = (y >= qs[i]) & ((y < qs[i + 1]) if i < 9 else (y <= qs[i + 1]))
    print(f"  y {qs[i]:6.1f}~{qs[i+1]:6.1f} n={m.sum():3d} RMSE {np.sqrt(se[m].mean()):6.2f} 편향 {r[m].mean():+6.2f} | 제곱오차 비중 {se[m].sum()/se.sum()*100:4.1f}%")
print("\n잔차 표준편차 %.3f, 예측-실제 기울기(실제=a+b*예측) b=%.3f, 예측 표준편차 %.2f / 실제 %.2f" % (r.std(), np.polyfit(o, y, 1)[0], o.std(), y.std()))
np.save("analysis/v15a_oof.npy", o)
