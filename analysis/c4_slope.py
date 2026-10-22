"""결정립 수(유효 표본 수)에 따른 보정 기울기: 굵은 결정립 이미지에서 예측을 더 축소해야 하는가?"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
v15 = np.load("analysis/v15a_oof.npy")
F = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[ids]
hq = sig < 10
def table(cov, name, m, nb=5):
    v = F[cov].values; qs = np.quantile(v[m], np.linspace(0, 1, nb + 1))
    print(f"\n[{name}] {cov} {nb}분위: n | 값 범위 | y표준편차 | 예측표준편차 | 잔차표준편차 | R² | 기울기 b (y=a+b·예측) ± se")
    for i in range(nb):
        k = m & (v >= qs[i]) & ((v < qs[i + 1]) if i < nb - 1 else (v <= qs[i + 1]))
        yy, pp = y[k], v15[k]; b, a = np.polyfit(pp, yy, 1); res = yy - (a + b * pp)
        se = res.std() / (pp.std() * np.sqrt(k.sum()))
        print(f"  n={k.sum():3d} | {qs[i]:8.2f}~{qs[i+1]:8.2f} | {yy.std():5.1f} | {pp.std():5.1f} | {(yy-pp).std():5.1f} | {1-((yy-pp)**2).mean()/yy.var():+.2f} | {b:.2f} ± {se:.2f}")
for nm, m in (("고품질", hq), ("저품질", ~hq), ("전체", np.ones(len(y), bool))):
    table("gr_count", nm, m)
table("sp_acf_len_0.5", "전체", np.ones(len(y), bool), 4)
table("an_local_coherence", "전체", np.ones(len(y), bool))
table("gr_lb_density_snr", "전체", np.ones(len(y), bool))
print("\ngr_count 분포: 전체", np.quantile(F.gr_count, [0, .1, .25, .5, .75, .9, 1]).round(0), " 고품질", np.quantile(F.gr_count[hq], [0, .1, .5, .9, 1]).round(0), " 저품질", np.quantile(F.gr_count[~hq], [0, .1, .5, .9, 1]).round(0))
# 2모수 보정 교차검증: y = a + (b0 + b1*z)·(pred - mu),  z = 표준화 log(결정립 수) 등
def cvcal(z, label, m=None):
    m = np.ones(len(y), bool) if m is None else m
    out = v15.copy()
    for k in range(5):
        t, v = (fold != k) & m, (fold == k) & m
        mu = v15[t].mean(); zt = (z - z[t].mean()) / z[t].std()
        A = np.column_stack([np.ones(t.sum()), v15[t] - mu, zt[t] * (v15[t] - mu), zt[t]])
        w = np.linalg.lstsq(A, y[t], rcond=None)[0]
        out[v] = np.column_stack([np.ones(v.sum()), v15[v] - mu, zt[v] * (v15[v] - mu), zt[v]]) @ w
    d = (out - y) ** 2 - (v15 - y) ** 2
    print(f"  {label:34s} RMSE {np.sqrt(((v15-y)[m]**2).mean()):.3f} → {np.sqrt(((out-y)[m]**2).mean()):.3f} | Δ제곱 {d[m].mean():+.2f}±{d[m].std()/np.sqrt(m.sum()):.2f} | 폴드 개선 {sum(d[m & (fold==k)].mean() < 0 for k in range(5))}/5 | 마지막 폴드 계수 b0 {w[1]:.2f} b1 {w[2]:+.3f}")
print("\n== 기울기를 공변량에 따라 바꾸는 2모수 보정 (교차검증) ==")
for c in ("gr_count", "gr_area_mean", "sp_acf_len_0.5", "gr_lb_density_snr", "an_local_coherence", "gr_grad_density", "tx_glcm_correlation_d6", "sp_centroid"):
    v = F[c].values.astype(float); z = np.log(np.clip(v, 1e-6, None)) if (v > 0).all() else v
    cvcal(z, c)
    cvcal(z, c + " (고품질만)", hq)
