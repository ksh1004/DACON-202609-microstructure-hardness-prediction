import numpy as np, pandas as pd, sys
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor
ROOT = ""
T = np.round(np.arange(-0.2, 0.601, 0.025), 3); TA = np.arange(80, 201, 4)
def one(img):
    x = img.astype(float); f = {}
    for s in (7, 8, 9, 10):
        g = ndi.gaussian_filter(x, s)
        for nm, ref in (("mean", g.mean()), ("med", np.median(g)), ("p75", np.percentile(g, 75))):
            rel = (ref - g) / ref
            for t in T: f[f"{nm}{s}_gt{t:+.3f}"] = float((rel > t).mean())
        for t in TA: f[f"abs{s}_lt{t}"] = float((g < t).mean())
    return f
if __name__ == "__main__":
    S = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"; z = np.load(ROOT + f"outputs/denoised/{S}.npz")
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        ids = pd.read_csv(ROOT + "data/" + csv).ID
        with ProcessPoolExecutor(4) as ex: rows = list(ex.map(one, list(z[split]), chunksize=16))
        pd.DataFrame(rows).assign(ID=ids.values).to_csv(ROOT + f"outputs/features_cache/smcdf4_{S}_{split}.csv", index=False)
    print("done")
