"""이미지별 밝기(기지 밝기)·대비가 경도와 무관한 방해 요인인지 확인."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from scipy import ndimage as ndi
from scipy.stats import spearmanr
from numpy.linalg import lstsq
X = np.load("outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz")["train"].astype(np.float32)
hq = sig < 10
rows = []
for x in X:
    c = ndi.grey_closing(x, size=5); m = np.quantile(c, .7); rel = (m - c) / m
    rows.append((m, x.mean(), np.quantile(x, .99) - np.quantile(x, .01), x.std(), (rel > .08).mean(), np.quantile(c, .95), np.quantile(c, .05)))
d = pd.DataFrame(rows, columns=["matrix_q70", "mean", "contrast", "std", "dark_frac", "q95", "q05"])
p = np.load("analysis/v15a_oof.npy"); r = y - p
q = lambda v, qs: np.quantile(v, qs).round(0)
print("기지 밝기(닫기 후 70 분위) 분포: 전체", q(d.matrix_q70, [0, .05, .5, .95, 1]), "| 고품질", q(d.matrix_q70[hq], [.05, .5, .95]), "| 저품질", q(d.matrix_q70[~hq], [.05, .5, .95]))
print("항목            경도rho  2상분율rho  잔차rho  |잔차|rho   (고품질만: 경도rho / 잔차rho)")
for c in d.columns:
    v = d[c].values
    print("{:14s} {:+7.3f} {:+10.3f} {:+8.3f} {:+9.3f}   ({:+.3f} / {:+.3f})".format(c, spearmanr(v, y)[0], spearmanr(v, d.dark_frac)[0], spearmanr(v, r)[0], spearmanr(v, np.abs(r))[0], spearmanr(v[hq], y[hq])[0], spearmanr(v[hq], r[hq])[0]))
def partial(a, b, ctrl):
    A = np.column_stack([np.ones(len(a)), ctrl]); ra = a - A @ lstsq(A, a, rcond=None)[0]; rb = b - A @ lstsq(A, b, rcond=None)[0]; return np.corrcoef(ra, rb)[0, 1]
for c in ("matrix_q70", "q95", "contrast"):
    print("{}: 2상 분율 통제 후 경도와 편상관 전체 {:+.3f} / 고품질 {:+.3f}".format(c, partial(d[c].values, y, d.dark_frac.values), partial(d[c].values[hq], y[hq], d.dark_frac.values[hq])))
