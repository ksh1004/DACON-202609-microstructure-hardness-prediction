"""잔차의 비선형 구조 점검: (1) 주요 물리량 구간별 잔차 평균·분산, (2) 수작업 특징으로 잔차를 GBM/RF로 설명 (5폴드, 같은 폴드)."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
import lightgbm as lgb
from sklearn.ensemble import ExtraTreesRegressor
v15 = np.load("analysis/v15a_oof.npy"); r = y - v15  # 실제 - 예측 (양수 = 과소예측)
F = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[ids]
num = F.select_dtypes("number").replace([np.inf, -np.inf], np.nan); num = num.fillna(num.median())
num = num[[c for c in num.columns if not c.startswith("nui_")]]
hq = sig < 10
def binned(name, v, m, nb=8):
    qs = np.quantile(v[m], np.linspace(0, 1, nb + 1)); out = []
    for i in range(nb):
        k = m & (v >= qs[i]) & ((v < qs[i + 1]) if i < nb - 1 else (v <= qs[i + 1]))
        out.append(f"{r[k].mean():+5.1f}/{r[k].std():4.1f}/{y[k].mean():5.0f}")
    print(f"  {name:24s} " + " | ".join(out))
print("== 구간별 [잔차 평균 / 잔차 표준편차 / 경도 평균] — 고품질만, 8분위 (왼쪽=작은 값) ==")
for c in ("gr_count", "gr_area_mean", "gr_eq_diam_mean", "sp_acf_len_0.5", "gr_lb_density_snr", "ph_dark_frac_0.08", "an_local_coherence", "gr_area_cv", "gr_area_p90_p10", "pore_area_frac", "pore_cand_count", "het_dark_std", "gr_aspect_wmean"):
    if c in num: binned(c, num[c].values, hq)
print("== 같은 표, 전체 ==")
for c in ("gr_count", "sp_acf_len_0.5", "ph_dark_frac_0.08", "an_local_coherence", "gr_area_p90_p10"):
    if c in num: binned(c, num[c].values, np.ones(len(y), bool))
X = num.values
def cv(make, Xm, mask, tgt):
    p = np.zeros(len(y))
    for k in range(5):
        t, v = (fold != k) & mask, (fold == k) & mask
        p[v] = make().fit(Xm[t], tgt[t]).predict(Xm[v])
    return 1 - ((tgt - p)[mask] ** 2).sum() / ((tgt - tgt[mask].mean())[mask] ** 2).sum(), p
mk_l = lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.02, num_leaves=6, min_child_samples=15, subsample=0.8, subsample_freq=1, colsample_bytree=0.5, reg_lambda=5, verbose=-1, random_state=0)
mk_e = lambda: ExtraTreesRegressor(n_estimators=500, min_samples_leaf=8, max_features=0.3, random_state=0, n_jobs=8)
allm = np.ones(len(y), bool)
print("\n== 수작업 특징(%d개) → v15a 잔차, 비선형 모델 CV R² ==" % X.shape[1])
for nm, mk in (("LightGBM", mk_l), ("ExtraTrees", mk_e)):
    for mn, m in (("전체", allm), ("고품질", hq), ("저품질", ~hq)):
        r2, p = cv(mk, X, m, r)
        new = np.sqrt((((r - p)[m]) ** 2).mean())
        print(f"  {nm:10s} {mn}: R² {r2:+.4f} | RMSE {np.sqrt((r[m]**2).mean()):.3f} → {new:.3f}")
# 예측값 자체를 특징에 추가 (예측 수준별 비선형 보정)
X2 = np.column_stack([X, v15])
for nm, mk in (("LightGBM+pred", mk_l), ("ExtraTrees+pred", mk_e)):
    r2, p = cv(mk, X2, allm, r)
    print(f"  {nm:16s} 전체: R² {r2:+.4f} → RMSE {np.sqrt(((r - p) ** 2).mean()):.3f}")
# 잔차 제곱(오차 크기) 설명력: 이분산 구조
r2, p = cv(mk_e, X2, allm, np.abs(r)); print(f"\n  |잔차| 설명 ExtraTrees: R² {r2:+.4f}")
