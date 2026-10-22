"""블렌드 후보 평가 도구: OOF·pw13·경도 구간별·모드별 오차, 폴드별 비교, test 예측 차이."""
import numpy as np, pandas as pd
tr = pd.read_csv("data/train.csv"); y = tr.hardness.values; ids = tr.ID
te = pd.read_csv("data/sample_submission.csv").ID
fold = pd.read_csv("outputs/folds.csv").set_index("ID").loc[ids, "fold"].values
sig = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[ids, "q_noise_sigma"].values
E = "outputs/emb_reg/"
def oof(d): return pd.read_csv(f"{d}/oof.csv").set_index("ID").loc[ids, "pred"].values
def tp(d):
    for n in ("test_pred_full.csv", "test_pred.csv"):
        try: return pd.read_csv(f"{d}/{n}").set_index("ID").loc[te, "hardness"].values
        except FileNotFoundError: pass
def grp(ds): return np.mean([oof(d) for d in ds], 0), np.mean([tp(d) for d in ds], 0)
w13 = np.exp(-0.5 * ((y - y.mean()) / 13) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2)
mid = (y > np.quantile(y, .1)) & (y < np.quantile(y, .9))
def rep(name, o, base=None):
    r = lambda m=None, w=None: np.sqrt(np.average(((o - y) ** 2)[m] if m is not None else (o - y) ** 2, weights=w))
    s = f"{name:40s} OOF {r():.3f} pw13 {r(w=w13):.3f} 중간80% {r(mid):.3f} 고품질 {r(sig < 10):.3f} 저품질 {r(sig >= 10):.3f}"
    if base is not None:
        d = (o - y) ** 2 - (base - y) ** 2
        s += f" | Δ제곱 {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f} 폴드 개선 {sum(d[fold == k].mean() < 0 for k in range(5))}/5"
    print(s)
CNN = [f"outputs/cnn/s512_{m}_s{s}" for m in ("convnext_tiny", "effnet_b2", "resnet50") for s in (42, 43, 44)]
FF = (32, 48, 64, 96)
