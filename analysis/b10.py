import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from scipy import stats
co, _ = grp(CNN); do, _ = grp([E + f"dn2b_fflowpass{s}_low_L11-17" for s in FF]); qo, _ = grp([E + f"dnpp0.25_ff{s}" for s in FF]); po = oof(E + "mr_v14_pp_sc")
o = .15*co+.55*qo+.3*po
print("왜도: 라벨 %.2f | 예측 v15a %.2f, CNN %.2f, DINO %.2f, P %.2f" % (stats.skew(y), stats.skew(o), stats.skew(co), stats.skew(do), stats.skew(po)))
print("상위 5% 예측 구간의 실제−예측: " + ", ".join(f"{n} {np.mean((y - x)[x >= np.quantile(x, .95)]):+.1f}" for n, x in (("CNN", co), ("DINO", do), ("Dpp", qo), ("P", po), ("v14c", .15*co+.55*do+.3*po))))
print("폴드별 (v15a 상위 5% 예측): " + " ".join(f"k{k}: {np.mean((y - o)[(fold == k) & (o >= np.quantile(o, .95))]):+.1f}(n={np.sum((fold == k) & (o >= np.quantile(o, .95)))})" for k in range(5)))
def cv(fn):
    out = o.copy()
    for k in range(5):
        t = fold != k; out[~t] = fn(o[t], y[t], o[~t])
    return out
def hinge(t0, free_b=False):
    def f(pt, yt, pv):
        X = np.c_[np.ones_like(pt), pt, np.maximum(pt - t0, 0)] if free_b else np.c_[np.maximum(pt - t0, 0)]
        if free_b:
            b = np.linalg.lstsq(X, yt, rcond=None)[0]; return np.c_[np.ones_like(pv), pv, np.maximum(pv - t0, 0)] @ b
        c = np.linalg.lstsq(X, yt - pt, rcond=None)[0][0]; return pv + c * np.maximum(pv - t0, 0)
    return f
def quad(pt, yt, pv):
    b = np.polyfit(pt, yt, 2); return np.polyval(b, pv)
rep("v15a (보정 없음)", o)
for t0 in (205, 210, 213, 216):
    rep(f"꺾은선(위쪽만 기울기 추가) t={t0}", cv(hinge(t0)), o)
rep("꺾은선 t=210 + 전체 기울기·절편", cv(hinge(210, True)), o)
rep("2차식", cv(quad), o)
for t0 in (210,):
    X = np.maximum(o - t0, 0); print(f"전체 적합 t={t0}: 추가 기울기 c = {np.sum(X * (y - o)) / np.sum(X * X):.2f}, 해당 이미지 {np.sum(o > t0)}장")
