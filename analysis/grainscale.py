import numpy as np, pandas as pd, sys
from scipy import ndimage as ndi
from skimage import morphology
from concurrent.futures import ProcessPoolExecutor
ROOT = ""
sys.path.insert(0, ROOT + "src")
from phase_features import matrix_level
T = np.round(np.arange(-0.2, 0.601, 0.025), 3)
def cdf(g, p, f):
    m = matrix_level(g); rel = (m - g) / m
    for t in T: f[f"{p}_gt{t:+.3f}"] = float((rel > t).mean())
    for q in (5, 25, 50, 75, 95): f[f"{p}_q{q}"] = float(np.percentile(rel, q))
def one(img):
    x = img.astype(float); f = {}
    for s in (9, 13, 17): cdf(ndi.median_filter(x, size=s), f"md{s}", f)
    c = morphology.closing(x, morphology.disk(2))
    for s in (6, 8, 10): cdf(ndi.gaussian_filter(c, s), f"cg{s}", f)
    for s in (4, 8):
        mu = ndi.gaussian_filter(x, s); v = np.sqrt(np.clip(ndi.gaussian_filter(x * x, s) - mu * mu, 0, None)) / matrix_level(mu)
        for q in (5, 25, 50, 75, 95): f[f"ls{s}_q{q}"] = float(np.percentile(v, q))
    return f
if __name__ == "__main__":
    S = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"; z = np.load(ROOT + f"outputs/denoised/{S}.npz")
    for split, csv in (("train", "train.csv"), ("test", "sample_submission.csv")):
        ids = pd.read_csv(ROOT + "data/" + csv).ID
        with ProcessPoolExecutor(4) as ex: rows = list(ex.map(one, list(z[split]), chunksize=16))
        pd.DataFrame(rows).assign(ID=ids.values).to_csv(ROOT + f"outputs/features_cache/gscale_{S}_{split}.csv", index=False)
    print("done")
