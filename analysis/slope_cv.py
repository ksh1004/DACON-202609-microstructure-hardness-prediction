"""블렌드 예측 폭 보정의 정직한 이득: 배율을 바깥 폴드를 뺀 나머지 OOF로 추정해 바깥 폴드에 적용(중첩).
(1) 전체 한 배율 (2) 모드별 배율 (3) 수축된 고정값. 그리고 예측 분위별 잔차(비선형 여부), 라벨 분포를 Private 꼴(std 18)로 재가중한 오차."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, ct = grp(CNN); po, pt = oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")
ao, at = grp([E + f"dnpp0.25_ff{s}" for s in FF]); do, dt = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF]); to, tt = grp([E + f"tips16_dnppm0.05_ff{s}" for s in FF])
fo, ft = grp([E + f"cnnf_{n}_ff{s}" for n in ("dn121", "r50a1", "r50a2") for s in FF])
W = {"v19a": [.15, .15, .12375, .19125, .175, .21], "v19b(noC)": [0, .15] + [x * .85 / .70 for x in (.12375, .19125, .175, .21)], "v19c(noC,noP)": [0, 0] + [x / .70 for x in (.12375, .19125, .175, .21)], "v18a": [.15, .15, .1925, .2975, .21, 0]}
CO = np.column_stack([co, po, ao, do, to, fo]); c = 194.16551625321193; Lq = sig >= 10
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
w18, w166 = wg(18.0), wg(16.6)
def kfit(p, m): return ((y[m] - c) * (p[m] - c)).sum() / ((p[m] - c) ** 2).sum()      # 원점(c) 통과 최소제곱 배율
rm = lambda q, w=None: np.sqrt(np.average((q - y) ** 2, weights=w))
for n, w in W.items():
    p = CO @ np.array(w); g = np.zeros(len(y)); mo = np.zeros(len(y)); ks = []
    for k in range(5):
        t, v = fold != k, fold == k; kg = kfit(p, t); g[v] = c + kg * (p[v] - c)
        kh, kl = kfit(p, t & ~Lq), kfit(p, t & Lq); mo[v] = c + np.where(Lq[v], kl, kh) * (p[v] - c); ks.append((kg, kh, kl))
    ks = np.array(ks); fx = lambda kh, kl: c + np.where(Lq, kl, kh) * (p - c)
    print(f"\n[{n}] 기준 OOF {rm(p):.3f} (Private꼴 std18 {rm(p, w18):.3f}, Public꼴 std16.6 {rm(p, w166):.3f}) | 폴드별 추정 배율 전체 {ks[:, 0].mean():.3f}±{ks[:, 0].std():.3f}, 고품질 {ks[:, 1].mean():.3f}±{ks[:, 1].std():.3f}, 저품질 {ks[:, 2].mean():.3f}±{ks[:, 2].std():.3f}")
    for nm, q in (("중첩: 전체 한 배율", g), ("중첩: 모드별 배율", mo), ("고정 1.05", fx(1.05, 1.05)), ("고정 1.08", fx(1.08, 1.08)), ("고정 고1.00/저1.10", fx(1.0, 1.10)), ("고정 고1.03/저1.12", fx(1.03, 1.12)), ("고정 고1.05/저1.15", fx(1.05, 1.15))):
        d = (q - y) ** 2 - (p - y) ** 2
        print(f"   {nm:22s} OOF {rm(q):.3f} ({rm(q) - rm(p):+.3f}) | std18 {rm(q, w18) - rm(p, w18):+.3f} | std16.6 {rm(q, w166) - rm(p, w166):+.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f} 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | 고품질 {rm(q[~Lq] * 0 + q[~Lq]) if 0 else np.sqrt(((q - y)[~Lq] ** 2).mean()) - np.sqrt(((p - y)[~Lq] ** 2).mean()):+.3f} 저품질 {np.sqrt(((q - y)[Lq] ** 2).mean()) - np.sqrt(((p - y)[Lq] ** 2).mean()):+.3f}")
    if n == "v19a":
        print("   예측 분위별 (실제−예측) 평균 [전체 / 고품질 / 저품질], 예측 중심 대비 위치:")
        for nm, m in (("전체", np.ones(len(y), bool)), ("고품질", ~Lq), ("저품질", Lq)):
            qs = np.quantile(p[m], [0, .1, .25, .5, .75, .9, 1]); r = []
            for a_, b_ in zip(qs[:-1], qs[1:]):
                mm = m & (p >= a_) & (p <= b_); r.append(f"{(p[mm] - c).mean():+6.1f}:{(y - p)[mm].mean():+5.1f}±{(y - p)[mm].std() / np.sqrt(mm.sum()):.1f}")
            print(f"     {nm}: " + "  ".join(r))
