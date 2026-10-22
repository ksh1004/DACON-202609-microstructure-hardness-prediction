# 학습곡선 + 전체적합 vs 폴드평균 시뮬레이션 (v12a DINO 블록 ff48)
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
ROOT = ""
tr = pd.read_csv(ROOT + "data/train.csv"); y = tr.hardness.values
fold = pd.read_csv(ROOT + "outputs/folds.csv").set_index("ID").loc[tr.ID, "fold"].values
z = np.load(ROOT + "outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_Sunet_cm7_am22_l1_i10000_boot_fflowpass48_low/emb.npz")
E = z["train"].astype(np.float64).reshape(500, 7, 6, 1024)[:, [1, 2, 3, 4]].reshape(500, -1)
AL = np.logspace(-1, 7, 60)
def fit(X, yy):
    sc = StandardScaler().fit(X); r = RidgeCV(alphas=AL).fit(sc.transform(X), yy); return lambda Z: r.predict(sc.transform(Z)), r.alpha_
rng = np.random.default_rng(0)
res = {f: [] for f in (0.5, 0.65, 0.8, 1.0)}
A, B = np.zeros(500), np.zeros(500)
for k in range(5):
    t, v = np.where(fold != k)[0], np.where(fold == k)[0]
    # full fit on 400
    p, al = fit(E[t], y[t]); B[v] = p(E[v])
    # fold-avg of inner 5 models (320 each)
    acc = 0
    for it, iv in KFold(5, shuffle=True, random_state=k).split(t):
        p2, _ = fit(E[t[it]], y[t[it]]); acc = acc + p2(E[v]) / 5
    A[v] = acc
    for f in res:
        reps = 1 if f == 1.0 else 4
        for r in range(reps):
            s = t if f == 1.0 else rng.choice(t, int(f * len(t)), replace=False)
            p3, _ = fit(E[s], y[s]); res[f].append(np.mean((p3(E[v]) - y[v]) ** 2))
    print(k, "alpha", al, flush=True)
rm = lambda p: np.sqrt(np.mean((p - y) ** 2))
print("fold-avg(5x320) ", rm(A), " full-fit(400) ", rm(B), " diff sq/img", np.mean((A-y)**2-(B-y)**2), "+-", np.std((A-y)**2-(B-y)**2)/np.sqrt(500))
for f in res: print(f"n={int(400*f)} RMSE {np.sqrt(np.mean(res[f])):.3f}")
