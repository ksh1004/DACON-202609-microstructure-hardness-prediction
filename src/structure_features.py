"""Fixed per-image structure descriptors; no labels or fitted dataset statistics.

NLM + h-minima watershed, count/area-weighted phase ratios, boundary
stereology and multiscale variograms. Original features remain unchanged.
Extract train first; test is only extracted for final inference.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation
from skimage.restoration import denoise_nl_means, estimate_sigma

ROOT = Path(__file__).resolve().parents[1]
VERSION = 1


def views(img):
    x = img.astype(np.float32) / 255
    sigma = float(estimate_sigma(x, channel_axis=None)) if np.ptp(x) > 1e-8 else 0.
    if not np.isfinite(sigma):
        sigma = 0.
    d = denoise_nl_means(x, h=0.7 * sigma, sigma=sigma, patch_size=5,
                        patch_distance=6, fast_mode=True, channel_axis=None) if sigma > 1e-8 else x.copy()
    edge = morphology.black_tophat(d, morphology.disk(2))
    scale = max(float(np.percentile(edge, 99)), 1 / 255)
    edge = np.clip(edge / scale, 0, 1)
    return x, d, edge, sigma


def dist_stats(v, prefix, weights=None):
    v = np.asarray(v, dtype=float)
    if len(v) == 0:
        v = np.zeros(1)
        weights = None
    mu = float(np.mean(v))
    f = {prefix + '_mean': mu, prefix + '_std': float(np.std(v)),
         prefix + '_cv': float(np.std(v) / (abs(mu) + 1e-6))}
    for q, z in zip([10, 25, 50, 75, 90], np.percentile(v, [10, 25, 50, 75, 90])):
        f[prefix + f'_q{q}'] = float(z)
    if weights is not None:
        f[prefix + '_wmean'] = float(np.average(v, weights=weights))
    return f


def grain_stats(d, prefix, h):
    smooth = ndi.gaussian_filter(d, 0.7)
    grad = filters.sobel(smooth)
    grad = grad / max(float(np.percentile(grad, 99)), 1e-6)
    markers = measure.label(morphology.h_minima(grad, h))
    ws = segmentation.watershed(grad, markers)
    props = [p for p in measure.regionprops(ws, intensity_image=d) if p.area >= 12]
    f = {prefix + '_count': float(len(props))}
    area = np.array([p.area for p in props])
    diam = np.sqrt(4 * area / np.pi)
    ar = np.array([p.axis_major_length / max(p.axis_minor_length, 1) for p in props])
    depth = np.array([p.intensity_mean for p in props])
    ref = max(float(np.percentile(d, 75)), 1e-5)
    depth = (ref - depth) / ref
    for n, v in [('area', area), ('diam', diam), ('aspect', ar), ('depth', depth)]:
        f.update(dist_stats(v, prefix + '_' + n, area if len(area) else None))
    f[prefix + '_inv_sqrt_diam'] = float(np.mean(1 / np.sqrt(diam))) if len(diam) else 0.
    interior = [p.area for p in props if p.bbox[0] > 0 and p.bbox[1] > 0
                and p.bbox[2] < d.shape[0] and p.bbox[3] < d.shape[1]]
    f.update(dist_stats(interior, prefix + '_interior_area'))
    for t in [.04, .08, .15, .25]:
        m = depth > t
        f[prefix + f'_phase_count_{t}'] = float(m.mean()) if len(m) else 0.
        f[prefix + f'_phase_area_{t}'] = float(area[m].sum() / area.sum()) if len(area) else 0.
    if props:
        angle = np.array([p.orientation for p in props])
        eccentricity = np.array([p.eccentricity for p in props])
        for name, w in [('count', np.ones(len(props))), ('area', area), ('elongation', area * eccentricity ** 2)]:
            w = w / max(float(w.sum()), 1e-9)
            f[prefix + '_alignment_' + name] = float(abs(np.sum(w * np.exp(2j * angle))))
    else:
        for name in ['count', 'area', 'elongation']:
            f[prefix + '_alignment_' + name] = 0.
    return f


def extract(img):
    x, d, edge, sigma = views(img)
    f = {'v2_noise': sigma}
    for h in [.04, .10]:
        f.update(grain_stats(d, f'v2_gr{h}', h))
    contrast = max(float(np.percentile(d, 99) - np.percentile(d, 1)), 1e-5)
    for r in [1, 2, 4]:
        b = morphology.black_tophat(d, morphology.disk(r)) / contrast
        f.update(dist_stats(b.ravel(), f'v2_boundary{r}'))
        for t in [.04, .10, .20]:
            mask = b > t
            skel = morphology.skeletonize(mask)
            f[f'v2_boundary{r}_area{t}'] = float(mask.mean())
            f[f'v2_boundary{r}_length{t}'] = float(skel.mean())
    for name, z in [('raw', x), ('nlm', d)]:
        previous = None
        for s in [1, 2, 3, 4, 6, 8, 12, 16, 24, 32]:
            vx = float(np.mean((z[:, s:] - z[:, :-s]) ** 2))
            vy = float(np.mean((z[s:] - z[:-s]) ** 2))
            v = (vx + vy) / 2
            f[f'v2_var_{name}_{s}'] = v / (contrast ** 2)
            if previous is not None:
                f[f'v2_var_delta_{name}_{s}'] = (v - previous) / contrast ** 2
            previous = v
    clo = morphology.closing(d, morphology.disk(2))
    ref = max(float(np.percentile(clo, 75)), 1e-5)
    rel = (ref - clo) / ref
    for t in [.04, .08, .12, .18, .25, .35]:
        mask = rel > t
        f[f'v2_phase_area_{t}'] = float(mask.mean())
        p = measure.regionprops(measure.label(mask))
        areas = [a.area for a in p if a.area >= 8]
        f[f'v2_phase_count_{t}'] = float(len(areas))
        f.update(dist_stats(areas, f'v2_phase_component_{t}'))
    for t in [.25, .40]:
        p = measure.regionprops(measure.label(rel > t))
        pores = [a for a in p if 8 <= a.area <= 350 and a.solidity >= .65]
        f[f'v2_pore_count_{t}'] = float(len(pores))
        f[f'v2_pore_frac_{t}'] = sum(a.area for a in pores) / d.size
    for scale in [.7, 1.5, 3., 6.]:
        gy, gx = np.gradient(ndi.gaussian_filter(d, scale))
        xx, yy, xy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
        f[f'v2_align_global_{scale}'] = float(np.hypot(xx - yy, 2 * xy) / (xx + yy + 1e-12))
        for rho in [3, 9]:
            xx, yy, xy = [ndi.gaussian_filter(z, rho) for z in [gx * gx, gy * gy, gx * gy]]
            co = np.hypot(xx - yy, 2 * xy) / (xx + yy + 1e-12)
            f[f'v2_align_local_{scale}_{rho}'] = float(co.mean())
            f[f'v2_align_local_sd_{scale}_{rho}'] = float(co.std())
    view = np.rint(np.clip(np.stack([d, edge]), 0, 1) * 255).astype(np.uint8)
    assert all(np.isfinite(v) for v in f.values())
    return f, view


def extract_one(item):
    split, sid = item
    with Image.open(ROOT / 'data' / split / f'{sid}.png') as im:
        f, v = extract(np.array(im))
    return {'ID': sid, **f}, v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', choices=['train', 'test'], default='train')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    source = 'train.csv' if a.split == 'train' else 'sample_submission.csv'
    ids = pd.read_csv(ROOT / 'data' / source).ID.tolist()
    if a.limit:
        ids = ids[:a.limit]
    out = ROOT / 'outputs' / 'structure_v1'
    out.mkdir(parents=True, exist_ok=True)
    suffix = f'_first{a.limit}' if a.limit else ''
    rows, tensors, t0 = [], [], time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i, (row, tensor) in enumerate(pool.map(extract_one, [(a.split, sid) for sid in ids])):
            rows.append(row); tensors.append(tensor)
            if (i + 1) % 25 == 0:
                print(f'{a.split} {i + 1}/{len(ids)} {time.time()-t0:.1f}s', flush=True)
    pd.DataFrame(rows).to_csv(out / f'{a.split}_features{suffix}.csv', index=False)
    np.savez_compressed(out / f'{a.split}_views{suffix}.npz', ID=np.array(ids), views=np.stack(tensors), version=VERSION)
    print(f'Done: {len(rows)} images, {len(rows[0])-1} features, {time.time()-t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
