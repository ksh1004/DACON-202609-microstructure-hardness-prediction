import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, _ = grp(CNN); do, _ = grp([E + f"dn2b_fflowpass{s}_low_L11-17" for s in FF]); qo, _ = grp([E + f"dnpp0.25_ff{s}" for s in FF]); po = oof(E + "mr_v14_pp_sc")
lq = sig >= 10
def slope(o, m): return np.polyfit(o[m], y[m], 1)[0]
print("블록별 보정 기울기 (고품질 / 저품질)  [표준오차 ≈ 0.04 / 0.07]")
for n, o in (("CNN9", co), ("DINO ff4", do), ("DINO+상풀링 ff4", qo), ("측정 블록 P", po), ("v14c 구성", .15*co+.55*do+.3*po), ("v15a 구성", .15*co+.55*qo+.3*po)):
    print(f"  {n:16s} {slope(o, ~lq):.3f} / {slope(o, lq):.3f}   폴드별 저품질: " + " ".join(f"{slope(o, lq & (fold == k)):.2f}" for k in range(5)))
o = .15*co+.55*qo+.3*po
def cvcal(mode):
    out = o.copy()
    for k in range(5):
        t = fold != k; v = ~t; mu = y[t].mean()
        if mode == "global":
            b = np.polyfit(o[t], y[t], 1); out[v] = np.polyval(b, o[v])
        elif mode == "permode":
            for m in (lq, ~lq):
                b = np.polyfit(o[t & m], y[t & m], 1); out[v & m] = np.polyval(b, o[v & m])
        elif mode == "lq_slope":  # 저품질만 평균 중심 확대 (기울기 1개)
            b = np.sum((o[t & lq] - mu) * (y[t & lq] - mu)) / np.sum((o[t & lq] - mu) ** 2); out[v & lq] = mu + b * (o[v & lq] - mu)
        elif mode.startswith("fix"):
            kk = float(mode[3:]); out[v & lq] = mu + kk * (o[v & lq] - mu)
    return out
rep("v15a 구성 (보정 없음)", o)
for md in ("global", "permode", "lq_slope", "fix1.05", "fix1.10", "fix1.15", "fix1.20"):
    rep(f"보정: {md}", cvcal(md), o)
