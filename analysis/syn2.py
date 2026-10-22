import numpy as np, pandas as pd
from scipy import ndimage as ndi
from PIL import Image
ROOT = ""
tr = pd.read_csv(ROOT + "data/train.csv")
sig = pd.read_csv(ROOT + "outputs/features/train_features.csv").set_index("ID").loc[tr.ID, "q_noise_sigma"].values
raw = np.stack([np.array(Image.open(ROOT + f"data/train/{i}.png")) for i in tr.ID]).astype(float)
syn = np.load(ROOT + "outputs/synlow/cm10_x2_s0/images.npz")["noisy"].astype(float)
dn = np.load(ROOT + "outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz")["train"].astype(float)
sdn = np.load(ROOT + "outputs/synlow/cm10_x2_s0/images.npz")["denoised"].astype(float)  # 이미 조명보정됨
low, high = sig >= 10, sig < 10
def haar(x):
    hh = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) / 2; return np.median(np.abs(hh)) / 0.6745
N = 256; fy, fx = np.meshgrid(np.fft.fftfreq(N), np.fft.fftfreq(N), indexing="ij"); FR = np.sqrt(fx ** 2 + fy ** 2)
w1 = np.hanning(N); W = np.outer(w1, w1); Wn = (W ** 2).sum()
bands = [(0.01, 0.03), (0.03, 0.06), (0.06, 0.1), (0.1, 0.15), (0.15, 0.22), (0.22, 0.3)]
def stats(x, d):
    s = haar(x); P = np.abs(np.fft.fft2((x - x.mean()) * W)) ** 2 / Wn
    sp = [P[(FR >= a) & (FR < b)].mean() - s ** 2 for a, b in bands]
    lp = ndi.gaussian_filter(x, 32)
    return dict(mean=x.mean(), std=x.std(), p1=np.percentile(x, 1), p99=np.percentile(x, 99), noise=s,
                illum=lp.std(), dn_std=d.std(), dn_p1=np.percentile(d, 1), dn_p99=np.percentile(d, 99),
                **{f"psd{i}": np.log10(max(v, 1e-3)) for i, v in enumerate(sp)})
R = pd.DataFrame([stats(raw[i], dn[i]) for i in np.where(low)[0]])
H = pd.DataFrame([stats(raw[i], dn[i]) for i in np.where(high)[0]])
Sy = pd.DataFrame([stats(syn[i], sdn[i]) for i in range(len(syn))])
q = lambda D, c: " ".join(f"{v:7.2f}" for v in np.percentile(D[c], [25, 50, 75]))
print(f"{'stat':8s} | 실제 고품질 25/50/75      | 실제 저품질               | 합성 저품질")
for c in R.columns: print(f"{c:8s} | {q(H, c)} | {q(R, c)} | {q(Sy, c)}")
