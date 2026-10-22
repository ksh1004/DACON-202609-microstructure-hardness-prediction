import numpy as np, pandas as pd, torch, sys
from scipy import stats as st
from PIL import Image
ROOT = ""
sys.path.insert(0, ROOT + "src")
from train_denoiser import UNet, denoise
tr = pd.read_csv(ROOT + "data/train.csv")
sig = pd.read_csv(ROOT + "outputs/features/train_features.csv").set_index("ID").loc[tr.ID, "q_noise_sigma"].values
raw = np.stack([np.array(Image.open(ROOT + f"data/train/{i}.png")) for i in tr.ID]).astype(float)
dn = np.load(ROOT + "outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz")["train"].astype(float)
Z = np.load(ROOT + "outputs/synlow/cm10_x2_s0/images.npz"); syn = Z["noisy"].astype(float)
net = UNet().cuda(); net.load_state_dict(torch.load(ROOT + "weights/denoiser_cm7_am22_l1_i10000_boot.pt", map_location="cuda"))
sdn = denoise(net, torch.from_numpy(Z["noisy"]).unsqueeze(1)).squeeze(1).numpy().astype(float)  # 조명보정 전 디노이즈
def resid_stats(x, d):
    r = x - d; r = r - r.mean()
    c1 = np.mean(r[:, 1:] * r[:, :-1]) / r.var(); c2 = np.mean(r[1:, :] * r[:-1, :]) / r.var()
    c_diag = np.mean(r[1:, 1:] * r[:-1, :-1]) / r.var(); c3 = np.mean(r[:, 2:] * r[:, :-2]) / r.var()
    # 밝기 의존: 밝은 영역 vs 어두운 영역 잔차 표준편차 비
    m = d > np.median(d); dep = r[m].std() / r[~m].std()
    return dict(std=r.std(), lag1=(c1 + c2) / 2, diag=c_diag, lag2=c3, kurt=st.kurtosis(r.ravel()), bright_dep=dep)
lo, hi = np.where(sig >= 10)[0], np.where(sig < 7)[0]
R = pd.DataFrame([resid_stats(raw[i], dn[i]) for i in lo]); H = pd.DataFrame([resid_stats(raw[i], dn[i]) for i in hi])
S = pd.DataFrame([resid_stats(syn[i], sdn[i]) for i in range(len(syn))])
q = lambda D, c: " ".join(f"{v:7.3f}" for v in np.percentile(D[c], [25, 50, 75]))
print(f"{'stat':10s} | 실제 고품질(σ<7)         | 실제 저품질               | 합성 저품질")
for c in R.columns: print(f"{c:10s} | {q(H, c)} | {q(R, c)} | {q(S, c)}")
