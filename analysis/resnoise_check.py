"""디노이즈 후 남은 노이즈 수준이 이미지마다 다른가 (경도와 무관한 방해 요인 후보)."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from scipy.stats import spearmanr
X = np.load("outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz")["train"].astype(np.float32)
def mad_sigma(x):  # 대각 고주파(Haar HH) 중앙절대편차 기반 노이즈 추정
    hh = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) / 2
    return np.median(np.abs(hh)) / 0.6745
rs = np.array([mad_sigma(x) for x in X])
print("원본 노이즈 구간별 디노이즈 후 잔여 노이즈(중앙값, 5~95%):")
for lo, hi in ((0, 5.5), (5.5, 7), (7, 8.5), (8.5, 10), (10, 14), (14, 30)):
    m = (sig >= lo) & (sig < hi); print("  원본 {:4.1f}~{:4.1f} n={:3d}: 잔여 {:.2f} ({:.2f}~{:.2f})".format(lo, hi, m.sum(), np.median(rs[m]), *np.quantile(rs[m], [.05, .95])))
hq = sig < 10
print("고품질 안에서 잔여 노이즈와 원본 노이즈의 상관 {:.3f}, 경도와 {:+.3f}".format(spearmanr(rs[hq], sig[hq])[0], spearmanr(rs[hq], y[hq])[0]))
p = np.load("analysis/v15a_oof.npy"); r = y - p
print("고품질: 잔여 노이즈와 잔차 {:+.3f}, |잔차| {:+.3f}".format(spearmanr(rs[hq], r[hq])[0], spearmanr(rs[hq], np.abs(r[hq]))[0]))
np.save("analysis/resnoise.npy", rs)
