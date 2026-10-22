import numpy as np, pandas as pd
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import StandardScaler
ROOT = ""
tr = pd.read_csv(ROOT + "data/train.csv"); y = tr.hardness.values
fold = pd.read_csv(ROOT + "outputs/folds.csv").set_index("ID").loc[tr.ID, "fold"].values
z = np.load(ROOT + "outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_Sunet_cm7_am22_l1_i10000_boot_fflowpass48_low/emb.npz")
E = z["train"].astype(np.float64).reshape(500, 7, 6, 1024)[:, [1, 2, 3, 4]].reshape(500, -1)
# 폴드별 표준화 후 SVD로 빠른 ridge 경로
als = np.logspace(-2, 6, 33)
P = np.zeros((len(als), 500))
for k in range(5):
    t, v = fold != k, fold == k
    sc = StandardScaler().fit(E[t]); Xt, Xv = sc.transform(E[t]), sc.transform(E[v])
    ym = y[t].mean()
    U, S, Vt = np.linalg.svd(Xt, full_matrices=False)
    Uy = U.T @ (y[t] - ym)
    for i, a in enumerate(als):
        w = Vt.T @ (S / (S ** 2 + a) * Uy)
        P[i, v] = Xv @ w + ym
    r = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Xt, y[t]); print(k, "gcv alpha", r.alpha_)
for i, a in enumerate(als):
    print(f"alpha {a:10.2f}  OOF {np.sqrt(np.mean((P[i] - y) ** 2)):.3f}  " + " ".join(f"{np.sqrt(np.mean((P[i][fold==k]-y[fold==k])**2)):.2f}" for k in range(5)))
