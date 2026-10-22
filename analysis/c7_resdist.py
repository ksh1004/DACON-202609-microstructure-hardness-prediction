"""잔차 분포의 모양: 숨은 이산 요인이 있으면 다봉(혼합) 분포. 라벨 값 자체의 이산 구조(반올림·격자)도 확인."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from scipy import stats
from sklearn.mixture import GaussianMixture
v15 = np.load("analysis/v15a_oof.npy"); r = y - v15; hq = sig < 10
print("라벨 소수 자릿수 예:", y[:8], "| 고유값 수", len(np.unique(y)), "/", len(y))
fr = np.modf(y)[0]; print("소수부 분포(10구간):", np.histogram(fr, 10, (0, 1))[0])
d = np.diff(np.sort(y)); print("정렬 라벨 간격: 최소 %.4f 중앙 %.4f" % (d.min(), np.median(d)))
for nm, m in (("전체", np.ones(len(y), bool)), ("고품질", hq), ("저품질", ~hq)):
    x = r[m]; z = (x - x.mean()) / x.std()
    print(f"\n[{nm}] n={m.sum()} 잔차 std {x.std():.2f} 왜도 {stats.skew(x):+.2f} 초과첨도 {stats.kurtosis(x):+.2f} | Shapiro p {stats.shapiro(x)[1]:.3f}")
    print("   표준화 잔차 히스토그램(-3~3, 12구간):", np.histogram(z, 12, (-3, 3))[0])
    X = x.reshape(-1, 1); b = [GaussianMixture(k, n_init=5, random_state=0).fit(X).bic(X) for k in (1, 2, 3)]
    print("   GMM BIC (1,2,3성분):", [round(v, 1) for v in b], "→ 최소", int(np.argmin(b)) + 1, "성분")
    print("   |잔차| 분위(50/75/90/95/99%):", np.quantile(np.abs(x), [.5, .75, .9, .95, .99]).round(1), "| 정규분포라면", (x.std() * stats.halfnorm.ppf([.5, .75, .9, .95, .99])).round(1))
# 잔차를 예측값에 대해 보정한 뒤(기울기) 남는 분포
b1, a1 = np.polyfit(v15, y, 1); r2 = y - (a1 + b1 * v15)
print("\n기울기 보정 후 잔차: std %.2f 왜도 %+.2f 초과첨도 %+.2f" % (r2.std(), stats.skew(r2), stats.kurtosis(r2)))
# 라벨 자체의 혼합 구조
X = y.reshape(-1, 1); b = [GaussianMixture(k, n_init=5, random_state=0).fit(X).bic(X) for k in (1, 2, 3, 4)]
print("라벨 GMM BIC (1~4성분):", [round(v, 1) for v in b])
