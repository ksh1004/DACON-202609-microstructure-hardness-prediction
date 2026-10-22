import numpy as np, pandas as pd, sys
from scipy import ndimage as ndi
from skimage import morphology
from concurrent.futures import ProcessPoolExecutor
ROOT = ""
sys.path.insert(0, ROOT + "src")
from phase_features import matrix_level
def one(img):
    x = img.astype(float); f = {}
    c = morphology.closing(x, morphology.disk(2)); m = matrix_level(c); rel = (m - c) / m
    for t in (0.04, 0.08, 0.12, 0.18):
        b = (rel > t).astype(float)
        for s in (6, 8, 12, 16):
            lf = ndi.gaussian_filter(b, s)
            for q in (5, 10, 25, 50, 75, 90, 95): f[f"lf_t{t}_s{s}_q{q}"] = float(np.percentile(lf, q))
            f[f"lf_t{t}_s{s}_std"] = float(lf.std())
    return f
if __name__ == "__main__":
    S = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"; z = np.load(ROOT + f"outputs/denoised/{S}.npz")
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        ids = pd.read_csv(ROOT + "data/" + csv).ID
        with ProcessPoolExecutor(4) as ex: rows = list(ex.map(one, list(z[split]), chunksize=16))
        pd.DataFrame(rows).assign(ID=ids.values).to_csv(ROOT + f"outputs/features_cache/locfrac_{S}_{split}.csv", index=False)
    print("done")
