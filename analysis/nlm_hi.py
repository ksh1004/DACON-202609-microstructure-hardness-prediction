import numpy as np, pandas as pd, sys
from concurrent.futures import ProcessPoolExecutor
from skimage.restoration import denoise_nl_means
ROOT = ""
def f(args):
    img, s, h = args
    x = img.astype(np.float64) / 255
    d = denoise_nl_means(x, h=h * s / 255, sigma=s / 255, patch_size=5, patch_distance=6, fast_mode=True)
    return np.clip(np.round(d * 255), 0, 255).astype(np.uint8)
if __name__ == "__main__":
    h = float(sys.argv[1])
    z = np.load(ROOT + "outputs/denoised/unet_cm7_am22_l1_i10000_boot.npz")
    out = {}
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        ids = pd.read_csv(ROOT + "data/" + csv).ID
        sig = pd.read_csv(ROOT + f"outputs/features/{split}_features.csv").set_index("ID").loc[ids, "q_noise_sigma"].values
        D = z[split].copy(); idx = np.where(sig < 10)[0]
        with ProcessPoolExecutor(6) as ex:
            res = list(ex.map(f, [(D[i], sig[i], h) for i in idx], chunksize=8))
        for i, r in zip(idx, res): D[i] = r
        out[split] = D; print(split, len(idx), flush=True)
    np.savez_compressed(ROOT + f"outputs/denoised/unet_cm7_am22_l1_i10000_boot_nlmhi{h:g}.npz", **out)
