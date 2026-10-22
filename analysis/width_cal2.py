"""채점된 제출 6개의 OOF를 재가중 폭별로 계산해 실제 Public과 비교 (수준 차이와 제출 간 차이)."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, _ = grp(CNN); po = oof(E + "mr_v14_pp_sc"); ao, _ = grp([E + f"dnpp0.25_ff{s}" for s in FF]); do, _ = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF]); to, _ = grp([E + f"tips16_dnppm0.05_ff{s}" for s in FF])
p16 = oof(E + "mv16_mr_pp_sc")
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
S = {"v15a": (.15 * co + .55 * ao + .30 * po, 10.2913), "v16a": (.15 * co + .55 * do + .30 * p16, 10.3467), "v16_noP": (.15 * co + .85 * do, 10.3171), "v16_noC": (.65 * do + .35 * p16, 10.3509),
     "v17b": (.15 * co + .275 * ao + .425 * do + .15 * po, 10.2318), "v18a": (.15 * co + .15 * po + .1925 * ao + .2975 * do + .21 * to, 10.1232)}
pub = np.array([v[1] for v in S.values()]); print("제출: " + "  ".join(f"{k} {v[1]:.4f}" for k, v in S.items()))
for s in (13, 14, 15, 16, 16.6, 17.68):
    w = wg(s); o = np.array([np.sqrt(np.average((p - y) ** 2, weights=w)) for p, _ in S.values()]); d = pub - o
    print(f"폭 {s:<5}: 재가중 OOF " + " ".join(f"{v:.3f}" for v in o) + f" | Public−OOF 평균 {d.mean():+.3f}, 제출 간 차이의 불일치(std) {d.std():.3f}, 상관 {np.corrcoef(o, pub)[0, 1]:+.2f}")
