"""가중치 진단용 제출 2종 생성 + 세 제출의 Public 점수가 나오면 임의 가중치의 Public 예상 점수를 계산.
a = v16a (.15 C, .55 D, .30 P) / b = noP (.15 C, .85 D, 0) / c = noC (0, .647 D, .353 P)
사용: python b16.py --save                      (파일 생성)
      python b16.py <Public a> <Public b> <Public c>   (가중치 평면 전체의 Public 예상)"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
co, ct = grp(CNN); do, dt = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF]); po, pt = oof(E + "mv16_mr_pp_sc"), tp(E + "mv16_mr_pp_sc")
W = {"a": (.15, .55, .30), "b": (.15, .85, 0.0), "c": (0.0, .55 / .85, .30 / .85)}
T = {k: w[0] * ct + w[1] * dt + w[2] * pt for k, w in W.items()}
O = {k: w[0] * co + w[1] * do + w[2] * po for k, w in W.items()}
if "--save" in sys.argv:
    for k, f in (("b", "sub_v16_noP.csv"), ("c", "sub_v16_noC.csv")):
        pd.DataFrame({"ID": te, "hardness": T[k]}).to_csv(f"submissions/{f}", index=False); print("→ submissions/" + f)
for k in W: rep(f"{k} = C{W[k][0]:.2f} D{W[k][1]:.2f} P{W[k][2]:.2f}", O[k], O["a"])
print("test 예측 쌍별 RMS차: a-b %.3f, a-c %.3f, b-c %.3f" % tuple(np.sqrt(((T[i] - T[j]) ** 2).mean()) for i, j in (("a", "b"), ("a", "c"), ("b", "c"))))
nums = [float(x) for x in sys.argv[1:] if x.replace(".", "").isdigit()]
if len(nums) == 3:
    M = dict(zip("abc", [v ** 2 for v in nums]))
    # 임의 가중치 w = (wC, wD, wP), 합 1 → w = αa + βb + γc (α+β+γ=1) 로 분해. MSE(Σ λ_i p_i) = Σ λ_i MSE_i − ½ Σ_ij λ_i λ_j D_ij (Σλ=1)
    A = np.array([W[k] for k in "abc"]).T  # 3×3
    Dm = {(i, j): ((T[i] - T[j]) ** 2).mean() for i in "abc" for j in "abc"}
    def pub(w):
        lam = np.linalg.solve(np.vstack([A[:2], np.ones(3)]), np.array([w[0], w[1], 1.0]))
        return np.sqrt(sum(l * M[k] for l, k in zip(lam, "abc")) - 0.5 * sum(lam[i] * lam[j] * Dm[("abc"[i], "abc"[j])] for i in range(3) for j in range(3)))
    def oofr(w): return np.sqrt(((w[0] * co + w[1] * do + w[2] * po - y) ** 2).mean())
    print("\n검산: a %.4f b %.4f c %.4f" % (pub(W["a"]), pub(W["b"]), pub(W["c"])))
    best = None
    print(f"{'wC':>5s} {'wD':>5s} {'wP':>5s} {'Public 예상':>10s} {'OOF':>7s}")
    for wc in (0, .05, .1, .15, .2, .25):
        for wp in (0, .1, .2, .3, .4, .5, .6):
            w = (wc, 1 - wc - wp, wp); v = pub(w)
            if best is None or v < best[0]: best = (v, w)
            print(f"{wc:5.2f} {1-wc-wp:5.2f} {wp:5.2f} {v:10.4f} {oofr(w):7.3f}")
    print("Public 예상 최소: %.4f at C%.2f D%.2f P%.2f" % (best[0], *best[1]))
