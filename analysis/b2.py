import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from scipy import stats
# 폴드 평균 test 예측(test_pred.csv)과 OOF 예측은 같은 조건(400장 학습) → 분포 비교 가능
def tpf(d): return pd.read_csv(f"{d}/test_pred.csv").set_index("ID").loc[te, "hardness"].values
for name, ds in (("DINO ff4", [E + f"dn2b_fflowpass{s}_low_L11-17" for s in FF]), ("mr_v14_pp_sc", [E + "mr_v14_pp_sc"]), ("CNN9", CNN)):
    o = np.mean([oof(d) for d in ds], 0); t = np.mean([tpf(d) for d in ds], 0)
    # 폴드 평균은 5모델 평균이라 분산이 약간 줄어듦 → 참고용
    print(f"{name:14s} OOF예측 mean {o.mean():.2f} std {o.std():.2f} | test예측 mean {t.mean():.2f} std {t.std():.2f} | KS p={stats.ks_2samp(o, t).pvalue:.2f}"
          f" | >215 비율 train {np.mean(o > 215):.3f} test {np.mean(t > 215):.3f} | <175 비율 train {np.mean(o < 175):.3f} test {np.mean(t < 175):.3f}")
print("train 라벨: mean %.2f std %.2f" % (y.mean(), y.std()))
s_all, s_pub = y.std(), 16.6
for st in (17.0, 17.7, 18.0):
    print(f"test 전체 std가 {st}라면 Private std ≈ {np.sqrt((1000 * st ** 2 - 300 * s_pub ** 2) / 700):.2f}")
# 부트스트랩: train에서 300장 뽑았을 때 std 분포 → Public 16.6이 얼마나 낮은 쪽인지
rng = np.random.default_rng(0); sd = [rng.choice(y, 300, replace=False).std() for _ in range(20000)]
print("train에서 300장 무작위 추출 std: 평균 %.2f, 5~95%% [%.2f, %.2f], P(std<=16.6)=%.3f" % (np.mean(sd), *np.percentile(sd, [5, 95]), np.mean(np.array(sd) <= 16.6)))
