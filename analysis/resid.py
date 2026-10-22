import numpy as np, pandas as pd
ROOT = ""
tr = pd.read_csv(ROOT + "data/train.csv"); y = tr.hardness.values
L = lambda d: pd.read_csv(ROOT + d + "/oof.csv").set_index("ID").loc[tr.ID, "pred"].values
cnn = np.mean([L(f"outputs/cnn/s512_{m}_s{s}") for m in ("convnext_tiny", "effnet_b2", "resnet50") for s in (42, 43, 44)], 0)
dn = np.mean([L(f"outputs/emb_reg/dn2b_fflowpass{s}_low_L11-17") for s in (32, 48, 64, 96)], 0)
ens = 0.2 * cnn + 0.8 * dn
sig = pd.read_csv(ROOT + "outputs/features/train_features.csv").set_index("ID").loc[tr.ID, "q_noise_sigma"].values
rm = lambda p, m: np.sqrt(np.mean((p[m] - y[m]) ** 2))
low = sig >= 10
for nm, p in (("cnn", cnn), ("dn", dn), ("ens", ens)):
    print(f"{nm}: all {rm(p, slice(None)):.3f} | high(σ<10, n={(~low).sum()}) {rm(p, ~low):.3f} | low(n={low.sum()}) {rm(p, low):.3f}")
e = ens - y
print("sq-err share low:", (e[low] ** 2).sum() / (e ** 2).sum())
q = pd.qcut(y, 5, labels=False)
for i in range(5):
    m = q == i; print(f"  y-quintile {i}: mean y {y[m].mean():.1f} bias {e[m].mean():+.2f} rmse {rm(ens, m):.2f}")
print("pred std", ens.std(), "y std", y.std(), "slope", np.polyfit(ens, y, 1))
print("corr cnn-dn resid", np.corrcoef(cnn - y, dn - y)[0, 1])
# 앙상블 가중 스캔
for w in (0, .1, .15, .2, .25, .3): print(w, round(rm(w * cnn + (1 - w) * dn, slice(None)), 3))
m = ens.mean()
for s in (1.0, 1.05, 1.1, 1.15): print("stretch", s, round(rm(m + s * (ens - m), slice(None)), 3))
# 폴드 내 보정 (중첩)
fold = pd.read_csv(ROOT + "outputs/folds.csv").set_index("ID").loc[tr.ID, "fold"].values
cal = np.zeros(500)
for k in range(5):
    t, v = fold != k, fold == k
    b = np.polyfit(ens[t], y[t], 1); cal[v] = np.polyval(b, ens[v])
print("nested linear calib", rm(cal, slice(None)))
