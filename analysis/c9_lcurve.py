"""정직한 학습 곡선 (32보기 σ48 특징, D 블록): 폴드 train에서 n장만 무작위로 써서 OOF. 오차² = a + b/n^c 외삽."""
import sys; sys.path.insert(0, "analysis")
from view_eval import *
from scipy.optimize import curve_fit
X = avg(S(ALL, (0, 1, 2, 3))); hq = sig < 10
res = []
for n in (80, 120, 160, 200, 250, 300, 350, 400):
    ms, mh, ml = [], [], []
    for r in range(1 if n == 400 else 4):
        rng = np.random.default_rng(n * 10 + r); o = np.zeros(len(y))
        for k in range(5):
            t = np.where(fold != k)[0]; v = fold == k
            keep = np.zeros(len(y), bool); keep[rng.choice(t, n, replace=False)] = True
            Z = design(X, keep); o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[keep], y[keep]).predict(Z[v])
        e = (o - y) ** 2; ms.append(e.mean()); mh.append(e[hq].mean()); ml.append(e[~hq].mean())
    res.append((n, np.mean(ms), np.mean(mh), np.mean(ml)))
    print(f"n={n:3d}: RMSE {np.sqrt(res[-1][1]):.3f} (고품질 {np.sqrt(res[-1][2]):.3f}, 저품질 {np.sqrt(res[-1][3]):.3f})", flush=True)
n, m = np.array([r[0] for r in res], float), np.array([r[1] for r in res])
for nm, col in (("전체", 1), ("고품질", 2), ("저품질", 3)):
    m = np.array([r[col] for r in res])
    for c in (0.5, 1.0):
        A = np.column_stack([np.ones(len(n)), n ** -c]); a, b = np.linalg.lstsq(A, m, rcond=None)[0]
        print(f"[{nm}] 오차² = a + b/n^{c}: 바닥 sqrt(a) = {np.sqrt(max(a, 0)):.2f}, n=500 → {np.sqrt(a + b * 500 ** -c):.2f}, n=1500 → {np.sqrt(a + b * 1500 ** -c):.2f}, 적합 잔차 {np.sqrt(((A @ [a, b] - m) ** 2).mean()):.2f}")
    try:
        (a, b, c), _ = curve_fit(lambda n, a, b, c: a + b * n ** -c, n, m, p0=(80, 3000, 0.8), maxfev=20000)
        print(f"[{nm}] 자유 지수: c = {c:.2f}, 바닥 {np.sqrt(max(a, 0)):.2f}, n=500 → {np.sqrt(a + b * 500 ** -c):.2f}")
    except Exception as e: print("fit 실패", e)
