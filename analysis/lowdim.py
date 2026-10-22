"""고품질 이미지에서 '정확히 측정된 소수의 물리 요인' + 비선형 저차원 모델. 상호작용이 핵심이라면 선형(11.85)보다 크게 좋아져 DINO(9.2)에 근접해야 한다."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as Ck
import lightgbm as lgb
S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
F = pd.read_csv(f"outputs/features_cache/{S48}_train.csv").set_index("ID").loc[ids]
hq = sig < 10
lg = lambda v: np.log(np.clip(v, 1e-6, None))
fac = {"2상분율>8%": F["ph_dark_frac_0.08"], "2상분율>15%": F["ph_dark_frac_0.15"], "어둡기 75분위": F["ph_rel_q75"], "기공 분율": F["pore_area_frac"], "기공 수": F["pore_count"],
       "log 결정립 수": lg(F["gr_count"]), "크기 변동계수": F["gr_area_cv"], "log 크기 p90/p10": lg(F["gr_area_p90_p10"]), "종횡비": F["gr_aspect_wmean"],
       "국소 정렬도": F["an_local_coherence"], "전역 정렬도": F["an_coherence"], "ACF 종횡비": F["an_acf_aspect"], "2상 공간 불균일": F["het_dark_std"], "ACF 길이": F["sp_acf_len_0.5"]}
X = np.column_stack([v.values.astype(float) for v in fac.values()]); X = np.where(np.isfinite(X), X, np.nanmedian(X, 0))
do, _ = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF])
def cv(make, Z, m, extra=None):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = (fold != k) & m, (fold == k) & m; mu, sd = Z[t].mean(0), Z[t].std(0) + 1e-9
        o[v] = make().fit((Z[t] - mu) / sd, y[t]).predict((Z[v] - mu) / sd)
    return np.sqrt(((o - y)[m] ** 2).mean()), o
mk = {"Ridge 선형": lambda: RidgeCV(alphas=np.logspace(-3, 3, 30)),
      "GP (RBF ARD)": lambda: GaussianProcessRegressor(Ck(1.0) * RBF(np.ones(X.shape[1]) * 3.0, (0.3, 100)) + WhiteKernel(0.3, (1e-2, 2)), normalize_y=True, n_restarts_optimizer=1, random_state=0),
      "LightGBM (얕게)": lambda: lgb.LGBMRegressor(n_estimators=400, learning_rate=0.02, num_leaves=5, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3, verbose=-1, random_state=0)}
class Poly:
    def fit(self, A, b): self.p = PolynomialFeatures(2, include_bias=False).fit(A); Z = self.p.transform(A); self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-9; self.r = RidgeCV(alphas=np.logspace(-2, 4, 30)).fit((Z - self.mu) / self.sd, b); return self
    def predict(self, A): return self.r.predict((self.p.transform(A) - self.mu) / self.sd)
mk["Ridge 2차(곱 포함)"] = Poly
try:
    from tabicl import TabICLRegressor
    mk["TabICL"] = lambda: TabICLRegressor()
except Exception as e: print("TabICL 사용 불가:", str(e)[:80])
print(f"요인 {X.shape[1]}개: {list(fac)}")
print(f"\n{'모델':22s} {'고품질만(n=291)':>14s} {'전체(n=500)':>12s}   | 참고: DINO 16보기 고품질 {np.sqrt(((do-y)[hq]**2).mean()):.2f}, 전체 {np.sqrt(((do-y)**2).mean()):.2f}")
res = {}
for nm, f in mk.items():
    try:
        a, oa = cv(f, X, hq); b, ob = cv(f, X, np.ones(len(y), bool)); res[nm] = (oa, ob)
        print(f"{nm:22s} {a:14.3f} {b:12.3f}   | 고품질(전체 학습) {np.sqrt(((ob-y)[hq]**2).mean()):.3f}", flush=True)
    except Exception as e: print(nm, "실패:", str(e)[:120])
# 요인 모델과 DINO의 결합 (고품질, 예측 블렌드)
best = min(res, key=lambda k: ((res[k][0] - y)[hq] ** 2).mean()); ob = res[best][0]
print(f"\n최선 요인 모델 = {best}; DINO와 잔차상관(고품질) {np.corrcoef((ob-y)[hq], (do-y)[hq])[0,1]:.3f}")
for w in (0.1, 0.2, 0.3, 0.4): print(f"  고품질: DINO {1-w:.1f} + 요인 모델 {w:.1f} → {np.sqrt((((1-w)*do + w*ob - y)[hq]**2).mean()):.3f}")
np.save("analysis/lowdim_oof.npy", np.stack([res[best][0], res[best][1]]))
