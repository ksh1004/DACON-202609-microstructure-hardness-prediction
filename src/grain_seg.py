"""결정립계 선 기반 결정립 분할 + 결정립 단위 측정 — 디노이즈(+조명보정) 이미지 캐시(npz), 이미지별 고정 연산.

기존 structure_features(기울기 watershed)는 얇은 결정립계 선 양쪽을 따로 잡아 띠 모양 조각을 대량 생성(과분할)한다.
여기서는 black top-hat으로 얇은 어두운 선(결정립계)을 검출 → 선 사이 연결 영역 = 결정립 → 선 화소는 이웃 결정립에 배정(watershed).
- gs2_ : 결정립 수·면적 분포(평균·중앙·CV·분위), 1/√지름 평균(Hall–Petch), 결정립계 길이밀도(골격 화소/면적)
- gp2_ : 결정립 평균 밝기의 기지 대비 상대 어둡기 → 어두운 결정립 비율(개수·면적, 임계값 여러 개), 결정립 어둡기 분위
- ga2_ : 결정립 종횡비·정렬도(2θ 평균 합성 길이, 면적 가중)
선이 거의 안 보이는 이미지(블러된 저품질)는 결정립 밝기 차이 경계(기울기)를 함께 쓴다.

실행: python src/grain_seg.py --src outputs/denoised/unet_cm7_am22_l1_i10000_boot_fflowpass48_low.npz
산출물: outputs/features_cache/grain2_<stem>_{train,test}.csv
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import filters, measure, morphology, segmentation

from phase_features import matrix_level

ROOT = Path(__file__).resolve().parents[1]
DARK_T = (0.03, 0.05, 0.08, 0.12, 0.18, 0.25)


def segment(img: np.ndarray, lo: float = 0.35, hi: float = 0.7, wg: float = 0.5):
    """반환: 결정립 라벨(전체 분할), 결정립계 마스크, 실수 이미지.
    Sato 능선 필터(얇은 어두운 선) + 0.5×Sobel(밝기 경계), 이미지별 p99 정규화 → 이력 임계값(lo, hi)으로 연결된 경계망"""
    d = img.astype(np.float64)
    r = filters.sato(d, sigmas=[1.0, 1.5], black_ridges=True)
    g = filters.sobel(ndi.gaussian_filter(d, 1.2))
    e = r / max(np.percentile(r, 99), 1e-6) + wg * g / max(np.percentile(g, 99), 1e-6)
    B = morphology.remove_small_objects(filters.apply_hysteresis_threshold(e, lo, hi), 15)
    core = morphology.remove_small_objects(~B, 12, connectivity=1)
    lab = segmentation.watershed(e, measure.label(core, connectivity=1))
    return lab, B, d


def _one(img: np.ndarray) -> dict:
    lab, B, d = segment(img)
    m = matrix_level(morphology.closing(d, morphology.disk(2)))
    props = measure.regionprops(lab, intensity_image=d)
    f = {}
    H, W = lab.shape
    area = np.array([p.area for p in props], float)
    inner = np.array([p.bbox[0] > 0 and p.bbox[1] > 0 and p.bbox[2] < H and p.bbox[3] < W for p in props])
    a_in = area[inner] if inner.sum() >= 3 else area
    f["gs2_count"] = float(len(props))
    f["gs2_density"] = float(len(props) / (H * W) * 1e4)
    for nm, v in (("area", a_in), ("lnarea", np.log(a_in))):
        f[f"gs2_{nm}_mean"] = float(v.mean())
        f[f"gs2_{nm}_std"] = float(v.std())
        for q in (10, 25, 50, 75, 90):
            f[f"gs2_{nm}_q{q}"] = float(np.percentile(v, q))
    f["gs2_area_cv"] = float(a_in.std() / a_in.mean())
    f["gs2_area_wmean"] = float((a_in ** 2).sum() / a_in.sum())
    f["gs2_invsqrt_d"] = float(np.mean(1 / np.sqrt(np.sqrt(4 * a_in / np.pi))))
    f["gs2_bnd_len"] = float(morphology.skeletonize(B).sum() / (H * W))
    f["gs2_bnd_frac"] = float(B.mean())
    # 결정립 밝기 (선 화소 제외한 내부 평균)
    nl = lab.max() + 1
    cnt_in = np.bincount(lab[~B], minlength=nl)
    sum_in = np.bincount(lab[~B], weights=d[~B], minlength=nl)
    cnt_all = np.bincount(lab.ravel(), minlength=nl)
    sum_all = np.bincount(lab.ravel(), weights=d.ravel(), minlength=nl)
    mean_l = np.where(cnt_in > 0, sum_in / np.maximum(cnt_in, 1), sum_all / np.maximum(cnt_all, 1))
    inten = mean_l[[p.label for p in props]]
    rel = (m - inten) / m
    for t in DARK_T:
        dk = rel > t
        f[f"gp2_cnt_{t}"] = float(dk.mean())
        f[f"gp2_area_{t}"] = float(area[dk].sum() / area.sum())
    for q in (50, 75, 90, 95):
        f[f"gp2_rel_q{q}"] = float(np.percentile(rel, q))
    f["gp2_rel_std"] = float(rel.std())
    ar = np.array([p.axis_major_length / max(p.axis_minor_length, 1) for p in props])
    th = np.array([p.orientation for p in props])
    w = area / area.sum()
    f["ga2_aspect_mean"] = float(ar.mean())
    f["ga2_aspect_wmean"] = float((ar * w).sum())
    f["ga2_aspect_q75"] = float(np.percentile(ar, 75))
    f["ga2_align"] = float(np.abs(np.sum(w * np.exp(2j * th))))
    f["ga2_align_el"] = float(np.abs(np.sum(w * (ar - 1) * np.exp(2j * th))) / max(np.sum(w * (ar - 1)), 1e-6))
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    tr = pd.read_csv(ROOT / "data/train.csv")
    te = pd.read_csv(ROOT / "data/sample_submission.csv")
    z = np.load(ROOT / a.src)
    stem = Path(a.src).stem
    t0 = time.time()
    for split, ids in (("train", tr.ID), ("test", te.ID)):
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            rows = list(ex.map(_one, list(z[split]), chunksize=8))
        df = pd.DataFrame(rows).assign(ID=ids.values)
        df.to_csv(ROOT / "outputs/features_cache" / f"grain2_{stem}_{split}.csv", index=False)
        print(f"[grain2_{stem}] {split} {df.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
