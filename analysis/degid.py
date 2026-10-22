# 열화 모델 진단: 원본 단계 통계로 실제 저품질 vs 합성 저품질 판별 + 통계별 차이
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy import ndimage as ndi
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score
ROOT = ""
N = 256; fy, fx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij"); FR = np.sqrt(fx ** 2 + fy ** 2)
w1 = np.hanning(N); W = np.outer(w1, w1); Wn = (W ** 2).sum()
edges = np.r_[0.004, np.geomspace(0.008, 0.5, 14)]
def haar(x):
    hh = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) / 2; return np.median(np.abs(hh)) / 0.6745
def stats(x):
    f = {}
    s = haar(x); f["noise"] = s
    P = np.abs(np.fft.fft2((x - x.mean()) * W)) ** 2 / Wn
    for i in range(len(edges) - 1):
        f[f"psd{i}"] = np.log10(max(P[(FR >= edges[i]) & (FR < edges[i + 1])].mean(), 1e-3))
        f[f"psdS{i}"] = np.log10(max(P[(FR >= edges[i]) & (FR < edges[i + 1])].mean() - s ** 2, 1e-1))
    for q in (1, 5, 25, 50, 75, 95, 99): f[f"q{q}"] = np.percentile(x, q) - x.mean()
    for sm in (1.5, 3, 6):
        g = ndi.gaussian_filter(x, sm)
        for q in (1, 10, 50, 90, 99): f[f"s{sm}_q{q}"] = np.percentile(g, q) - g.mean()
        f[f"s{sm}_std"] = g.std()
        gr = np.hypot(*np.gradient(g)); f[f"s{sm}_grad_q90"] = np.percentile(gr, 90); f[f"s{sm}_grad_q99"] = np.percentile(gr, 99)
    r = x - ndi.gaussian_filter(x, 1.0); r = r - r.mean()
    f["res_lag1"] = 0.5 * (np.mean(r[:, 1:] * r[:, :-1]) + np.mean(r[1:] * r[:-1])) / r.var()
    lp = ndi.gaussian_filter(x, 32); f["illum32"] = lp.std(); f["illum16"] = (ndi.gaussian_filter(x, 16) - lp).std()
    f["mean"] = x.mean(); f["skew"] = ((x - x.mean()) ** 3).mean() / x.std() ** 3
    f["sat_lo"] = (x <= 0).mean(); f["sat_hi"] = (x >= 255).mean()
    return f
if __name__ == "__main__":
    import sys
    tr = pd.read_csv(ROOT + "data/train.csv")
    sig = pd.read_csv(ROOT + "outputs/features/train_features.csv").set_index("ID").loc[tr.ID, "q_noise_sigma"].values
    raw = np.stack([np.array(Image.open(ROOT + f"data/train/{i}.png")) for i in tr.ID]).astype(float)
    syn = np.load(ROOT + (sys.argv[1] if len(sys.argv) > 1 else "outputs/synlow/cm10_x2_s0/images.npz"))["noisy"].astype(float)
    R = pd.DataFrame([stats(x) for x in raw[sig >= 10]]); S = pd.DataFrame([stats(x) for x in syn])
    X = pd.concat([R, S]).values; l = np.r_[np.zeros(len(R)), np.ones(len(S))]
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=5000))
    p = cross_val_predict(clf, X, l, cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    print(f"원본 통계 판별 AUC 실제저품질 vs 합성: {roc_auc_score(l, p):.3f}")
    rows = []
    for c in R.columns:
        a, b = R[c].values, S[c].values
        auc = roc_auc_score(np.r_[np.zeros(len(a)), np.ones(len(b))], np.r_[a, b]); auc = max(auc, 1 - auc)
        rows.append((auc, c, np.median(a), np.median(b)))
    for auc, c, ma, mb in sorted(rows, reverse=True)[:18]: print(f"  {c:14s} AUC {auc:.3f} | 실제 {ma:9.3f} 합성 {mb:9.3f}")
