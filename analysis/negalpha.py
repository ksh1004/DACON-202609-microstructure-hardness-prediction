import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
ROOT = ""
tr = pd.read_csv(ROOT + "data/train.csv"); y = tr.hardness.values
fold = pd.read_csv(ROOT + "outputs/folds.csv").set_index("ID").loc[tr.ID, "fold"].values
z = np.load(ROOT + "outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_Sunet_cm7_am22_l1_i10000_boot_fflowpass48_low/emb.npz")
E = z["train"].astype(np.float64).reshape(500, 7, 6, 1024)[:, [1, 2, 3, 4]].reshape(500, -1)
Ss = []
fr = np.array([-0.9, -0.8, -0.6, -0.4, -0.2, 0, 0.5, 1, 2])  # alpha = fr * min eigen(S^2) per fold
P = np.zeros((len(fr), 500))
for k in range(5):
    t, v = fold != k, fold == k
    sc = StandardScaler().fit(E[t]); Xt, Xv = sc.transform(E[t]), sc.transform(E[v])
    ym = y[t].mean()
    U, S, Vt = np.linalg.svd(Xt, full_matrices=False)
    S2 = S ** 2; nz = S2 > 1e-6 * S2[0]
    smin = S2[nz].min(); Ss.append((S2[0], np.median(S2[nz]), smin, nz.sum()))
    Uy = U.T @ (y[t] - ym)
    for i, f in enumerate(fr):
        a = f * smin
        w = Vt[nz].T @ (S[nz] / (S2[nz] + a) * Uy[nz])
        P[i, v] = Xv @ w + ym
print("S2 max/median/min/rank per fold:", [tuple(round(x, 1) for x in s) for s in Ss])
for i, f in enumerate(fr):
    print(f"alpha={f:+.1f}*min  OOF {np.sqrt(np.mean((P[i] - y) ** 2)):.3f}")
