"""예측 폭(기울기) 분해: 구성원 각각은 보정되어 있는데(기울기≈1) 평균하면 폭이 줄어드는지(블렌드 기울기>1) 확인.
기울기 = cov(y, p)/var(p) (y를 p에 회귀). 모드별, 그리고 단일 블록 → 그룹 평균 → 계열 블렌드 단계별."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
def sl(p, m=None):
    m = np.ones(len(y), bool) if m is None else m; c = np.cov(y[m], p[m]); s = c[0, 1] / c[1, 1]
    res = y[m] - (y[m].mean() + s * (p[m] - p[m].mean())); se = res.std() / (p[m].std() * np.sqrt(m.sum())); return s, se
H, Lq = sig < 10, sig >= 10
def show(n, p): 
    a, b, c = sl(p), sl(p, H), sl(p, Lq); print(f"{n:34s} RMSE {np.sqrt(((p - y) ** 2).mean()):6.3f} | 예측 std {p.std():5.2f} | 기울기 전체 {a[0]:.3f}±{a[1]:.3f}  고품질 {b[0]:.3f}±{b[1]:.3f}  저품질 {c[0]:.3f}±{c[1]:.3f}")
co, ct = grp(CNN); po, pt = oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")
ao, at = grp([E + f"dnpp0.25_ff{s}" for s in FF]); do, dt = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF]); to, tt = grp([E + f"tips16_dnppm0.05_ff{s}" for s in FF])
fo, ft = grp([E + f"cnnf_{n}_ff{s}" for n in ("dn121", "r50a1", "r50a2") for s in FF])
print("— 단일 블록 —"); show("CNN 1개 (convnext s42)", oof(CNN[0])); show("D16 σ48", oof(E + "mv16_dnppm0.05_ff48")); show("TIPS16 σ48", oof(E + "tips16_dnppm0.05_ff48")); show("고정 CNN dn121 σ48", oof(E + "cnnf_dn121_ff48"))
print("— 그룹(같은 계열 평균) —"); [show(n, p) for n, p in (("C (미세조정 CNN 9)", co), ("P (측정 블록)", po), ("A (DINOv2 4보기 ×4σ)", ao), ("D (DINOv2 16보기 ×4σ)", do), ("T (TIPSv2 ×4σ)", to), ("F (고정 CNN 3×4σ)", fo))]
print("— 블렌드 —"); dmix = (.275 * ao + .425 * do) / .70; f3 = .45 * dmix + .25 * to + .30 * fo; v19a = .15 * co + .15 * po + .70 * f3
show("DINOv2 혼합", dmix); show("3계열 (D.45 T.25 F.30)", f3); show("v19a", v19a)
print("\n— test 예측 표준편차(전체적합·폴드평균) vs OOF 예측 표준편차 —")
for n, o, t in (("C", co, ct), ("P", po, pt), ("A", ao, at), ("D", do, dt), ("T", to, tt), ("F", fo, ft)): print(f"  {n}: OOF {o.std():.2f} / test {t.std():.2f} (비 {t.std() / o.std():.3f})")
tm = (.275 * at + .425 * dt) / .70; t3 = .45 * tm + .25 * tt + .30 * ft; tv = .15 * ct + .15 * pt + .70 * t3; print(f"  v19a: OOF {v19a.std():.2f} / test {tv.std():.2f} (비 {tv.std() / v19a.std():.3f}) | train 라벨 std {y.std():.2f}")
# 이론값: 구성원이 각자 보정(기울기 s_i)일 때 평균의 기울기 = Σ w_i s_i var_i' ... 직접 계산: cov(y, Σw p)/var(Σw p)
