import numpy as np, pandas as pd, sys
from scipy import ndimage as ndi
from concurrent.futures import ProcessPoolExecutor
ROOT = ""
sys.path.insert(0, ROOT + "src")
from phase_features import matrix_level
T = np.round(np.arange(-0.2, 0.601, 0.025), 3)
SC = (7, 8, 9, 10)
def one(img):
    x = img.astype(float); f = {}
    for s in SC:
        g = ndi.gaussian_filter(x, s); m = matrix_level(g); rel = (m - g) / m
        for t in T: f[f"sc{s}_gt{t:+.3f}"] = float((rel > t).mean())
        for q in (5, 25, 50, 75, 95): f[f"sc{s}_q{q}"] = float(np.percentile(rel, q))
    return f
if __name__ == "__main__":
    for S in ("unet_cm7_am22_l1_i10000_boot_fflowpass48_low", "unet_boot2_i10000_fflowpass48_low", "unet_cm7_am22_l1_i10000_boot_c256_fflowpass48_low"):
        z = np.load(ROOT + f"outputs/denoised/{S}.npz")
        for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
            ids = pd.read_csv(ROOT + "data/" + csv).ID
            with ProcessPoolExecutor(4) as ex: rows = list(ex.map(one, list(z[split]), chunksize=16))
            pd.DataFrame(rows).assign(ID=ids.values).to_csv(ROOT + f"outputs/features_cache/smcdf3_{S}_{split}.csv", index=False)
        print(S, flush=True)
