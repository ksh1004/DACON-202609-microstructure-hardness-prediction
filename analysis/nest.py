# 중첩 시뮬레이션: v13a 설계(DINO + 위상 0.4 + pgs 0.15)에서 단일 400장 적합 vs 내부 5폴드(320장) 평균 — 외부 폴드에서 평가
import numpy as np, pandas as pd, re
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
ROOT = ""
tr = pd.read_csv(ROOT + "data/train.csv"); y = tr.hardness.values
fold = pd.read_csv(ROOT + "outputs/folds.csv").set_index("ID").loc[tr.ID, "fold"].values
S = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"; FC = ROOT + "outputs/features_cache/"
E = np.load(ROOT + f"outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Phetero3_S{S}/emb.npz")["train"].astype(np.float64).reshape(500, 7, 6, 1024)[:, 1:5].reshape(500, -1)
def ld(pre, rx):
    A = pd.read_csv(FC + pre + "_train.csv").set_index("ID").loc[tr.ID]; c = [x for x in A.columns if re.match(rx, x)]
    X = A[c].replace([np.inf, -np.inf], np.nan).values.astype(float); return X
G1 = np.hstack([ld("phase_" + S, "pf_"), ld(S, "ph_")]); G2 = ld(S, "(pore|gr|sp)_")
groups = [(0.4, G1), (0.15, G2)] if True else []
def design(t, rows):
    mu, sd = E[t].mean(0), E[t].std(0); sd[sd == 0] = 1
    parts = [(E[rows] - mu) / sd]
    dvar = E.shape[1]
    for g, F in groups:
        med = np.nanmedian(F[t], 0); F2 = np.where(np.isnan(F), med, F)
        m_, s_ = F2[t].mean(0), F2[t].std(0); keep = s_ > 1e-10
        parts.append(np.clip((F2[rows][:, keep] - m_[keep]) / s_[keep], -6, 6) * np.sqrt(g * dvar / keep.sum()))
    return np.hstack(parts)
def fitpred(tidx, vidx):
    t = np.zeros(500, bool); t[tidx] = True
    A = design(t, np.arange(500))
    r = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(A[tidx], y[tidx]); return r.predict(A[vidx]), r.alpha_
import sys
mode = sys.argv[1] if len(sys.argv) > 1 else "v13a"
if mode == "dino": groups = []
FULL, AVG = np.zeros(500), np.zeros(500)
for k in range(5):
    t, v = np.where(fold != k)[0], np.where(fold == k)[0]
    FULL[v], al = fitpred(t, v)
    acc = 0; als = []
    for it, iv in KFold(5, shuffle=True, random_state=k).split(t):
        p, a2 = fitpred(t[it], v); acc = acc + p / 5; als.append(a2)
    AVG[v] = acc
    print(k, "alpha full", round(al, 2), "inner", [round(a, 1) for a in als], flush=True)
rm = lambda p: np.sqrt(np.mean((p - y) ** 2))
d = (FULL - y) ** 2 - (AVG - y) ** 2
print(mode, "full(400) %.3f | avg(5x320) %.3f | Δsq %.1f ± %.1f | RMS diff %.3f" % (rm(FULL), rm(AVG), d.mean(), d.std() / np.sqrt(500), np.sqrt(np.mean((FULL - AVG) ** 2))))
