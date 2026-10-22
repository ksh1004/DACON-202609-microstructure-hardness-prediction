"""디노이저가 저주파(결정립 규모) 밝기를 보존하는가? 원본을 가우시안 σ로 평활한 것(노이즈 평균 → 편향 없음)과 비교."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from PIL import Image
from scipy import ndimage as ndi
dn = np.load("outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz")["train"].astype(float)
ff = np.load("outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz")["train"].astype(float)
raw = np.stack([np.asarray(Image.open(f"data/train/{i}.png"), float) for i in ids])
rows = []
for i in range(len(ids)):
    for s in (4, 8):
        gr, gd, gf = (ndi.gaussian_filter(a[i], s) for a in (raw, dn, ff))
        hp = lambda a: a - ndi.gaussian_filter(a, 48)  # 조명(초저주파) 제외한 결정립 규모 성분
        r_, d_, f_ = hp(gr), hp(gd), hp(gf)
        rows.append((i, s, r_.std(), d_.std(), np.polyfit(r_.ravel(), d_.ravel(), 1)[0], np.corrcoef(r_.ravel(), d_.ravel())[0, 1], gr.mean() - gd.mean(),
                     np.polyfit(r_.ravel(), f_.ravel(), 1)[0]))
d = pd.DataFrame(rows, columns=["i", "s", "std_raw", "std_dn", "slope_dn", "corr", "bias", "slope_ff"])
d["mode"] = np.where(sig[d.i] >= 10, "저품질", "고품질")
print(d.groupby(["s", "mode"])[["std_raw", "std_dn", "slope_dn", "corr", "bias", "slope_ff"]].median().round(3))
lq = d[(d.s == 8) & (d["mode"] == "저품질")]
print("저품질 σ8: 기울기(디노이즈/원본) 5~95%% [%.2f, %.2f], 상관 5%% %.2f" % (*lq.slope_dn.quantile([.05, .95]), lq.corr.quantile(.05)))
for name in ("TRAIN_000229", "TRAIN_000209", "TRAIN_000175", "TRAIN_000150"):
    i = list(ids).index(name); print(name, d[(d.i == i)][["s", "std_raw", "std_dn", "slope_dn", "corr", "slope_ff"]].round(3).values.tolist())
