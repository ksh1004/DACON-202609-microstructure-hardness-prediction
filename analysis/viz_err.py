import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from PIL import Image, ImageDraw
out = sys.argv[1]
co, _ = grp(CNN); qo, _ = grp([E + f"dnpp0.25_ff{s}" for s in FF]); po = oof(E + "mr_v14_pp_sc")
o = .15 * co + .55 * qo + .3 * po; r = o - y
dn = np.load("outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz")["train"]
def montage(idx, name, title):
    W = 256; n = len(idx); cols = 6; rows = (n + cols - 1) // cols
    im = Image.new("L", (cols * (W + 4), rows * (2 * W + 26)), 255); d = ImageDraw.Draw(im)
    for j, i in enumerate(idx):
        x0, y0 = (j % cols) * (W + 4), (j // cols) * (2 * W + 26)
        im.paste(Image.open(f"data/train/{ids[i]}.png"), (x0, y0 + 22)); im.paste(Image.fromarray(dn[i]), (x0, y0 + 22 + W))
        d.text((x0 + 2, y0 + 2), f"{ids[i][-3:]} y={y[i]:.0f} pred={o[i]:.0f} s={sig[i]:.0f}", fill=0)
    im.save(f"{out}/{name}.png"); print(name, title, [(ids[i], round(y[i]), round(o[i]), round(sig[i])) for i in idx])
hq = sig < 10
under = np.argsort(r)  # 예측이 실제보다 낮은 순 (실제가 훨씬 단단)
over = under[::-1]
montage([i for i in under if hq[i]][:6], "hq_under", "고품질: 실제가 예측보다 훨씬 단단")
montage([i for i in over if hq[i]][:6], "hq_over", "고품질: 실제가 예측보다 훨씬 무름")
# 대조군: 고품질, 정확히 맞춘 고경도/저경도
acc = np.abs(r) < 3
montage([i for i in np.argsort(-y) if hq[i] and acc[i]][:6], "hq_hard_ok", "고품질: 고경도 정확")
montage([i for i in np.argsort(y) if hq[i] and acc[i]][:6], "hq_soft_ok", "고품질: 저경도 정확")
lq = ~hq
montage([i for i in under if lq[i]][:6], "lq_under", "저품질: 실제가 예측보다 훨씬 단단")
montage([i for i in over if lq[i]][:6], "lq_over", "저품질: 실제가 예측보다 훨씬 무름")
