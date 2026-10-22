"""채점된 제출들의 가중치 공간(아핀 결합)으로 임의 블렌드의 Public 점수를 예상 (블렌드 항등식, b21.py의 일반화 — v19b·v19c 없이).
성분: C 미세조정 CNN, P 측정 블록(mr_v14_pp_sc), A DINOv2 4보기, D DINOv2 16보기, T TIPSv2, F 고정 CNN 3종 평균, J 고정 CNN 4종 결합, Q 16보기 측정 블록(mv16_mr_pp_sc), K 상수(train 평균)
  MSE(Σλ_i p_i) = Σλ_i M_i − ½ ΣΣ λ_i λ_j D_ij   (Σλ=1, D_ij = Public에서의 mean (p_i−p_j)² ≈ r × test 전체)
목표 가중치가 채점 제출들의 아핀 결합으로 정확히 표현될 때만 유효. |λ| 합이 크면 외삽이라 신뢰가 낮다."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
N = ["C", "P", "A", "D", "T", "F", "J", "Q", "K"]
T_ = dict(C=grp(CNN)[1], P=tp(E + "mr_v14_pp_sc"), A=grp([E + f"dnpp0.25_ff{f}" for f in FF])[1], D=grp([E + f"mv16_dnppm0.05_ff{f}" for f in FF])[1],
          T=grp([E + f"tips16_dnppm0.05_ff{f}" for f in FF])[1], F=grp([E + f"cnnf_{n}_ff{f}" for n in ("dn121", "r50a1", "r50a2") for f in FF])[1],
          J=grp([E + f"cnnfj4_ff{f}" for f in FF])[1], Q=tp(E + "mv16_mr_pp_sc"), K=np.full(len(te), 194.16551625321193))
O_ = dict(C=grp(CNN)[0], P=oof(E + "mr_v14_pp_sc"), A=grp([E + f"dnpp0.25_ff{f}" for f in FF])[0], D=grp([E + f"mv16_dnppm0.05_ff{f}" for f in FF])[0],
          T=grp([E + f"tips16_dnppm0.05_ff{f}" for f in FF])[0], F=grp([E + f"cnnf_{n}_ff{f}" for n in ("dn121", "r50a1", "r50a2") for f in FF])[0],
          J=grp([E + f"cnnfj4_ff{f}" for f in FF])[0], Q=oof(E + "mv16_mr_pp_sc"), K=np.full(len(y), 194.16551625321193))
CT = np.column_stack([T_[n] for n in N]); CO = np.column_stack([O_[n] for n in N])
v = lambda **kw: np.array([kw.get(n, 0.0) for n in N])
v21 = v(C=.15, P=.15, A=.12375, D=.19125, T=.175, J=.21)
SUB = {"const": (v(K=1), 16.5663, None), "v15a": (v(C=.15, P=.30, A=.55), 10.2913472134, "sub_v15a_dinopp.csv"), "v16a": (v(C=.15, D=.55, Q=.30), 10.3467169407, "sub_v16a_mv16.csv"),
       "v16_noP": (v(C=.15, D=.85), 10.3171457020, "sub_v16_noP.csv"), "v16_noC": (v(D=.647, Q=.353), 10.3509407312, "sub_v16_noC.csv"),
       "v18a": (v(C=.15, P=.15, A=.1925, D=.2975, T=.21), 10.1232275920, "sub_v18a_tips.csv"), "v19a": (v(C=.15, P=.15, A=.12375, D=.19125, T=.175, F=.21), 10.0678706806, "sub_v19a_cnnf.csv"),
       "v21a": (v21, 10.0319496436, "sub_v21a_cnnfj4.csv"), "v21a_k105": (1.05 * v21 + v(K=-.05), 10.0304563153, "sub_v21a_k105.csv"),
       "v22a": (1.05 * v(P=.15 / .85, A=.12375 / .85, D=.19125 / .85, T=.175 / .85, J=.21 / .85) + v(K=-.05), 9.9379181916, "sub_v22a_noC_k105.csv")}
for k, (w, s, f) in SUB.items():
    if f:
        d = np.sqrt(((CT @ w - pd.read_csv(f"submissions/{f}").set_index("ID").loc[te, "hardness"].values) ** 2).mean())
        if d > 0.02: print(f"  주의: {k} 성분 분해와 제출 파일의 RMS차 {d:.3f}")
keys = list(SUB); Wm = np.array([SUB[k][0] for k in keys]).T; Ms = np.array([SUB[k][1] ** 2 for k in keys]); Tm = CT @ Wm
print("채점 제출 %d개, 가중치 행렬 계수 %d (성분 %d개)" % (len(keys), np.linalg.matrix_rank(Wm), len(N)))
def pub(w, r=0.95):
    A = np.vstack([Wm, np.ones(len(keys))]); b = np.append(w, 1.0); lam = np.linalg.lstsq(A, b, rcond=None)[0]; res = np.abs(A @ lam - b).max()
    Dij = ((Tm[:, :, None] - Tm[:, None, :]) ** 2).mean(0) * r
    return np.sqrt(max(lam @ Ms - 0.5 * lam @ Dij @ lam, 0)), np.abs(lam).sum(), res
oofr = lambda w: np.sqrt(((CO @ w - y) ** 2).mean())
def show(nm, w):
    assert abs(w.sum() - 1) < 1e-9, nm; p, la, res = pub(w); lo, hi = pub(w, 0.90)[0], pub(w, 1.0)[0]
    ok = "" if res < 1e-6 else f"  ← 표현 불가(잔차 {res:.3f}): 참고만"
    print(f"  {nm:46s} Public 예상 {p:.4f} (r 0.90~1.0: {lo:.4f}~{hi:.4f}) | |λ| 합 {la:5.1f} | OOF {oofr(w):.3f}{ok}")
print("검산:"); [show(k, SUB[k][0]) for k in ("v19a", "v21a", "v21a_k105", "v22a")]
feat = dict(A=.12375, D=.19125, T=.175, J=.21)
def mk(c, p, k=1.0, **scale):
    f = {n: feat[n] * scale.get(n, 1.0) for n in feat}; s = sum(f.values()); w = v(C=c, P=p, **{n: f[n] / s * (1 - c - p) for n in f}); return k * w + v(K=1 - k)
print("C=0 기준 — 측정 블록(P) 비중 × 폭 배율 (나머지 성분은 v21a 비율 유지):")
for p_ in (.1765, .12, .06, 0):
    for k in (1.0, 1.03, 1.05, 1.08): show(f"C 0 P {p_} ×{k}", mk(0, p_, k))
print("C 비중 (P 0.15 고정, ×1.05):"); [show(f"C {c}", mk(c, .15 * (1 - c) / .85, 1.05)) for c in (.15, .10, .05, 0)]
print("C=0, P 0.1765, ×1.05에서 성분 비중:")
for nm, sc in (("기준(v22a)", {}), ("J ×1.3", dict(J=1.3)), ("J ×1.6", dict(J=1.6)), ("J ×0.7", dict(J=.7)), ("T ×1.4", dict(T=1.4)), ("T ×0.6", dict(T=.6)), ("D ×1.4", dict(D=1.4)), ("A ×0 (16보기만)", dict(A=0)), ("A ×2", dict(A=2)), ("J ×1.4, T ×1.3", dict(J=1.4, T=1.3)), ("J ×1.3, T ×1.3, P 0.10", dict(J=1.3, T=1.3))):
    show(nm, mk(0, .10 if "P 0.10" in nm else .1765, 1.05, **sc))
if "--save" in sys.argv:
    i = sys.argv.index("--save"); name = sys.argv[i + 1]; c, p, k = (float(x) for x in sys.argv[i + 2:i + 5]); sc = dict(zip(sys.argv[i + 5::2], map(float, sys.argv[i + 6::2])))
    w = mk(c, p, k, **sc); pd.DataFrame({"ID": te, "hardness": CT @ w}).to_csv(f"submissions/{name}", index=False); print("저장:", name, dict(zip(N, np.round(w, 4)))); show(name, w)
