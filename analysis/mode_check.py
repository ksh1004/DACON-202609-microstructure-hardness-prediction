"""촬영 모드(고품질/저품질) 판정 점검: 지금은 q_noise_sigma ≥ 10 하나로 가르는데, 이미지별 블러 지표(노이즈 바닥을 뺀 고주파/중주파 파워 비)로 보면
σ 10~14 구간이 두 모드의 혼합인지 확인. 결과 표: analysis/mode_feats.csv (train+test)"""
import sys, os; sys.path.insert(0, "analysis")
import numpy as np, pandas as pd
from PIL import Image
N = 256; w1 = np.hanning(N); W = np.outer(w1, w1); Wn = (W ** 2).sum()
fy, fx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij"); FR = np.sqrt(fx ** 2 + fy ** 2)
def haar_sigma(x):
    hh = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) / 2
    return np.median(np.abs(hh)) / 0.6745
def feats(x):
    x = x.astype(np.float64); s = haar_sigma(x); P = np.abs(np.fft.fft2((x - x.mean()) * W)) ** 2 / Wn
    band = lambda a, b: P[(FR >= a) & (FR < b)].mean()
    return dict(hsig=s, S_lo=band(0.02, 0.06) - s ** 2, S_mid=band(0.06, 0.12) - s ** 2, S_hi=band(0.22, 0.32) - s ** 2, S_vhi=band(0.32, 0.45) - s ** 2, floor=band(0.45, 0.5), std=x.std())
out = "analysis/mode_feats.csv"
if not os.path.exists(out):
    rows = []
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        for i in pd.read_csv("data/" + csv).ID:
            rows.append({"ID": i, "split": split, **feats(np.array(Image.open(f"data/{split}/{i}.png").convert("L")))})
    df = pd.DataFrame(rows)
    q = pd.concat([pd.read_csv("outputs/features/train_features.csv"), pd.read_csv("outputs/features/test_features.csv")]).set_index("ID")
    df["sigma"] = q.loc[df.ID, "q_noise_sigma"].values
    df["logR"] = np.log10(np.clip(df.S_hi, 1e-2, None) / np.clip(df.S_mid, 1e-2, None)); df["logRv"] = np.log10(np.clip(df.S_vhi, 1e-2, None) / np.clip(df.S_mid, 1e-2, None))
    df.to_csv(out, index=False)
df = pd.read_csv(out)
print("이미지 크기 256 가정 | 행", len(df))
for sp in ("train", "test"):
    d = df[df.split == sp]; print(f"[{sp}] σ 구간별 logR(고주파/중주파, 노이즈 바닥 제거) 분위 10/25/50/75/90 — 블러가 있으면 낮음")
    for lo, hi in ((4, 6), (6, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 17), (17, 21)):
        m = (d.sigma >= lo) & (d.sigma < hi)
        if m.sum(): print(f"  σ {lo:2d}-{hi:2d} n={m.sum():4d} " + " ".join(f"{x:+.2f}" for x in np.percentile(d.logR[m], [10, 25, 50, 75, 90])) + f" | logR>-1.0 비율 {np.mean(d.logR[m] > -1.0):.2f}")
h = np.histogram(df.logR, bins=np.arange(-3.2, 0.21, 0.2)); print("logR 전체 히스토그램:", dict(zip(np.round(h[1][:-1], 1), h[0])))
