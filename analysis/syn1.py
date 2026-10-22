import numpy as np, pandas as pd, re, warnings, sys
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT = ""
tr = pd.read_csv(ROOT + "data/train.csv"); y = tr.hardness.values.astype(float)
fold = pd.read_csv(ROOT + "outputs/folds.csv").set_index("ID").loc[tr.ID, "fold"].values
sig = pd.read_csv(ROOT + "outputs/features/train_features.csv").set_index("ID").loc[tr.ID, "q_noise_sigma"].values
low = sig >= 10
S = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
E = np.load(ROOT + f"outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_S{S}/emb.npz")["train"].astype(np.float64).reshape(500, 7, 6, 1024)[:, 1:5].reshape(500, -1)
PH = pd.read_csv(ROOT + f"outputs/features_cache/phase_{S}_train.csv").set_index("ID").loc[tr.ID]
pcols = [c for c in PH.columns if c.startswith("pf_")]; PH = PH[pcols].values.astype(float)
SD = ROOT + "outputs/synlow/" + (sys.argv[1] if len(sys.argv) > 1 else "cm10_x2_s0") + "/"
Es = np.load(SD + "emb.npz")["train"].astype(np.float64).reshape(-1, 7, 6, 1024)[:, 1:5].reshape(-1, 24576)
PHs = pd.read_csv(SD + "phase.csv")[pcols].values.astype(float)
meta = pd.read_csv(SD + "meta.csv"); idx = {i: k for k, i in enumerate(tr.ID)}
src = meta.ID.map(idx).values; ys = y[src]
def auc(A, B, npc=32):
    X = np.vstack([A, B]); l = np.r_[np.zeros(len(A)), np.ones(len(B))]
    clf = make_pipeline(StandardScaler(), PCA(npc, random_state=0), LogisticRegression(C=0.1, max_iter=3000))
    p = cross_val_predict(clf, X, l, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    return roc_auc_score(l, p)
print("AUC DINO: 실제저품질 vs 합성 %.3f | 실제저품질 vs 실제고품질 %.3f | 합성 vs 원본고품질 %.3f" % (auc(E[low], Es), auc(E[low], E[~low]), auc(E[~low], Es)))
print("AUC 상분율: 실제저품질 vs 합성 %.3f | 실제저품질 vs 실제고품질 %.3f" % (auc(PH[low], PHs, 16), auc(PH[low], PH[~low], 16)))
# 증강 커널 릿지
w13 = np.exp(-0.5 * ((y - y.mean()) / 13) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
rm = lambda p, m=slice(None), w=None: np.sqrt(np.average((p[m] - y[m]) ** 2, weights=None if w is None else w[m]))
def run(alpha, wsyn, gph=0.4, use_syn=True):
    oof = np.zeros(500)
    for k in range(5):
        t, v = fold != k, fold == k
        ss = np.isin(src, np.where(t)[0]) if use_syn else np.zeros(len(src), bool)
        def std(Xr, Xs):
            mu, sd = Xr[t].mean(0), Xr[t].std(0); keep = sd > 1e-10
            f = lambda Z: np.clip((Z[:, keep] - mu[keep]) / sd[keep], -6, 6)
            return f(Xr), f(Xs)
        Zr, Zs = std(E, Es); Pr, Ps = std(PH, PHs)
        sc = np.sqrt(gph * Zr.shape[1] / Pr.shape[1])
        Fr = np.hstack([Zr, Pr * sc]); Fs = np.hstack([Zs, Ps * sc])
        Ftr = np.vstack([Fr[t], Fs[ss]]); ytr = np.r_[y[t], ys[ss]]; wt = np.r_[np.ones(t.sum()), np.full(ss.sum(), wsyn)]
        ym = y[t].mean()
        Kt = Ftr @ Ftr.T
        a = np.linalg.solve(Kt + alpha * np.diag(1 / wt), ytr - ym)
        oof[v] = Fr[v] @ (Ftr.T @ a) + ym
    return oof
base = run(1.0, 1.0, use_syn=False)
print(f"기준(증강 없음) OOF {rm(base):.3f} pw13 {rm(base, w=w13):.3f} | 고품질 {rm(base, ~low):.3f} 저품질 {rm(base, low):.3f}")
for alpha in (1.0, 3e3):
    for wsyn in (0.25, 1.0):
        o = run(alpha, wsyn)
        print(f"증강 α={alpha:g} w_syn={wsyn}: OOF {rm(o):.3f} pw13 {rm(o, w=w13):.3f} | 고품질 {rm(o, ~low):.3f} 저품질 {rm(o, low):.3f}", flush=True)
