"""v25a 기준 평가 틀. import 하면 G(구성원 OOF·test), W25, show(이름, 가중, 배율)를 제공. 직접 실행하면 v25a 오차 분해."""
import sys, os; sys.path.insert(0, "analysis")
from blend_eval import *
TAG = "drt10b_i10000"; s = "_a0.01"; c = y.mean()
G = dict(P=(oof(E + f"mr_v14_pp_sc{s}"), tp(E + f"mr_v14_pp_sc{s}")), A=grp([E + f"dnpp0.25_ff{f}{s}" for f in FF]),
         Dj3=grp([E + f"mv16j3_{TAG}_ff{f}{s}" for f in FF]), T=grp([E + f"tips16_dnppm0.05_ff{f}{s}" for f in FF]), J=grp([E + f"cnnfj4_ff{f}{s}" for f in FF]),
         Tx=grp([E + f"tips16_{TAG}_ff{f}{s}" for f in FF]), Jx=grp([E + f"cnnfj4_{TAG}_ff{f}{s}" for f in FF]))
G["Tm"] = tuple((a_ + b_) / 2 for a_, b_ in zip(G["T"], G["Tx"])); G["Jm"] = tuple((a_ + b_) / 2 for a_, b_ in zip(G["J"], G["Jx"]))
W25 = dict(P=.1765, A=.1456, Dj3=.2250, Tm=.2059, Jm=.2471)
wg = lambda sd: np.exp(-0.5 * ((y - y.mean()) / sd) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166 = wg(16.6)
rm = lambda o, m=None, w=None: np.sqrt(np.average(((o - y) ** 2) if m is None else ((o - y)[m] ** 2), weights=w)); Lq = sig >= 10; q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
bl = lambda w, i, k: c + k * (sum(v * G[n][i] for n, v in w.items()) - c); base, bt = bl(W25, 0, 1.03), bl(W25, 1, 1.03)
def show(nm, w, k=1.03):
    o, t = bl(w, 0, k), bl(w, 1, k); d = (o - y) ** 2 - (base - y) ** 2; cv = np.cov(y, o)
    print(f"{nm:34s} OOF {rm(o):.3f} pw13 {rm(o, w=w13):.3f} Public꼴 {rm(o, w=w166):.3f} | 고 {rm(o, ~Lq):.3f} 저 {rm(o, Lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} {sum(d[fold == f].mean() < 0 for f in range(5))}/5 | 가운데 {d[mid].sum() / 500:+.2f} 양끝 {d[~mid].sum() / 500:+.2f} | 기울기 {cv[0, 1] / cv[1, 1]:.3f} test RMS차 {np.sqrt(((t - bt) ** 2).mean()):.2f}", flush=True)
    return o, t
if __name__ == "__main__":
    print("구성원 OOF: " + " | ".join(f"{n} {rm(G[n][0]):.3f} (고 {rm(G[n][0], ~Lq):.2f} 저 {rm(G[n][0], Lq):.2f})" for n in G))
    o, _ = show("v25a", W25); e2 = (o - y) ** 2
    print(f"\n제곱오차 분해 (전체 MSE {e2.mean():.1f})")
    for lo, hi in ((0, 5), (5, 7), (7, 10), (10, 14), (14, 18), (18, 99)):
        m = (sig >= lo) & (sig < hi); print(f"  σ {lo:2d}~{hi:2d}: {m.sum():3d}장 RMSE {np.sqrt(e2[m].mean()):6.2f}  MSE 기여 {e2[m].sum() / 500:5.1f}  정답 std {y[m].std():5.1f}")
    for lo, hi in ((0, .1), (.1, .25), (.25, .75), (.75, .9), (.9, 1.01)):
        m = (y >= np.quantile(y, lo)) & (y <= np.quantile(y, min(hi, 1))) if hi > 1 else (y >= np.quantile(y, lo)) & (y < np.quantile(y, hi))
        print(f"  정답 분위 {lo:.2f}~{min(hi, 1):.2f}: RMSE {np.sqrt(e2[m].mean()):6.2f} 평균 잔차 {(o - y)[m].mean():+6.2f}  MSE 기여 {e2[m].sum() / 500:5.1f}")
