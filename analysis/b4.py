import sys; sys.path.insert(0, "analysis")
from blend_eval import *
a = oof(E + "dnpp0.25_ff48"); rep("상풀링 th0.08 (σ48)", a)
ths = [oof(E + f"dnpp0.25_ff48_th{t}") for t in ("0.05", "0.12")]
rep("상풀링 th 3종 평균 (σ48)", np.mean([a] + ths, 0), a)
# 잔차 분포: 두꺼운 꼬리인가?
from scipy import stats
r = a - y; print("잔차 첨도(정규=0) %.2f, |r|>2.5σ 비율 %.3f (정규 0.012), 최대 |r| %.1f" % (stats.kurtosis(r), np.mean(np.abs(r) > 2.5 * r.std()), np.abs(r).max()))
