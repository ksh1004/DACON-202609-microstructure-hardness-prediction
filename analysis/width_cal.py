"""Public이 예측 폭 변화를 어떻게 평가하는지에 맞는 재가중 폭 찾기: v18a·v17b·v15a의 OOF 예측에서 재가중 폭 s별 최적 배율 k*(s)를 계산해
Public 역산값(v18a 1.003, v17b 0.996, v15a 0.988 — 예측분산비 0.931 가정)과 일치하는 폭을 찾는다."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, _ = grp(CNN); po = oof(E + "mr_v14_pp_sc"); ao, _ = grp([E + f"dnpp0.25_ff{s}" for s in FF]); do, _ = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF]); to, _ = grp([E + f"tips16_dnppm0.05_ff{s}" for s in FF])
c = 194.16551625321193; wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
P = {"v15a": .15 * co + .55 * ao + .30 * po, "v17b": .15 * co + .275 * ao + .425 * do + .15 * po, "v18a": .15 * co + .15 * po + .1925 * ao + .2975 * do + .21 * to}
print("재가중 폭 s별 OOF 최적 배율 k* (Public 역산: v15a 0.988, v17b 0.996, v18a 1.003 @분산비 0.931 / 분산비 0.88이면 +0.06):")
for s in (12, 13, 14, 15, 16, 16.6, 17.68, 18):
    w = wg(s); print(f"  폭 {s:<5}: " + "  ".join(f"{n} {np.sum(w * (y - c) * (p - c)) / np.sum(w * (p - c) ** 2):.3f}" for n, p in P.items()) + f" | 유효 표본 {w.sum() ** 2 / (w ** 2).sum():.0f}")
