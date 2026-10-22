"""미세구조 이미지 수작업(handcrafted) 특징 추출 모듈.

EDA와 이후 트리 기반 모델(LightGBM 등)에서 공통으로 사용한다.
모든 특징은 회전/반전에 대해 가능한 한 불변(invariant)이 되도록 설계했고,
방향(angle) 관련 특징은 'nui_' 접두어로 분리해 nuisance 분석에만 사용한다.

특징 그룹 접두어
- q_   : 영상 품질 (노이즈, 블러, 밝기, 대비)
- ph_  : 상(phase) 분율 — 기지(matrix) 대비 상대 어둡기 분포
- pore_: 기공(어두운 소형 블롭)
- gr_  : 결정립 분할(watershed) 기반 크기/형상/분포
- sp_  : 파워스펙트럼/자기상관 기반 특성 길이 (결정립 크기 프록시)
- an_  : 이방성(신장·정렬 정도)
- het_ : 공간적 불균일성 (4x4 타일 간 변동)
- tx_  : 일반 텍스처 (GLCM, LBP)
- nui_ : 방향 각도 등 nuisance (모델 입력에서 제외 권장)
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from scipy import stats
from skimage import feature, filters, measure, morphology, segmentation
from skimage.restoration import denoise_tv_chambolle, estimate_sigma

DARK_THRESHOLDS = (0.08, 0.15, 0.25, 0.35, 0.5)
REL_QUANTILES = (50, 75, 90, 95, 99, 99.5)
PSD_BANDS = np.array([1, 2, 4, 8, 16, 32, 64, 128])  # 반지름(주파수 인덱스) 경계


def preprocess(img: np.ndarray):
    """uint8 이미지 -> (raw float 0~255, TV 디노이즈, 추정 노이즈 σ, closing 이미지)."""
    x = img.astype(np.float64) / 255.0
    sigma = float(estimate_sigma(x))
    d = denoise_tv_chambolle(x, weight=max(0.02, 1.5 * sigma)) * 255.0
    clo = morphology.closing(d, morphology.disk(3))  # 얇은 결정립계 선 제거
    return x * 255.0, d, sigma * 255.0, clo


def matrix_level(clo: np.ndarray) -> float:
    """closing 이미지 히스토그램의 최빈값 = 기지(matrix) 상 밝기."""
    h, b = np.histogram(clo, bins=128, range=(0, 256))
    hs = ndi.gaussian_filter1d(h.astype(float), 2)
    return float(b[hs.argmax()] + 1.0)


def _quality(raw, d, sigma):
    f = {}
    f["q_noise_sigma"] = sigma
    f["q_mean"] = raw.mean()
    f["q_std"] = raw.std()
    p1, p50, p99 = np.percentile(d, [1, 50, 99])
    f["q_p01"], f["q_p50"], f["q_p99"] = p1, p50, p99
    f["q_contrast"] = p99 - p1
    f["q_blur_raw"] = measure.blur_effect(raw)
    f["q_blur_dn"] = measure.blur_effect(d)
    f["q_snr"] = d.std() / (sigma + 1e-6)
    return f


def _phase(clo, m):
    f = {}
    rel = (m - clo) / m  # 0=기지, 클수록 어두움(2상/기공)
    f["ph_matrix_level"] = m
    for t in DARK_THRESHOLDS:
        f[f"ph_dark_frac_{t:.2f}"] = float((rel > t).mean())
    qs = np.percentile(rel, REL_QUANTILES)
    for q, v in zip(REL_QUANTILES, qs):
        f[f"ph_rel_q{q}"] = float(v)
    f["ph_bright_frac_0.08"] = float((rel < -0.08).mean())  # 기지보다 밝은 결정립
    # 어두운 쪽 질량 (대비에 비례) 과 정규화 버전
    f["ph_dark_mass"] = float(np.clip(rel, 0, None).mean())
    f["ph_dark_mass_norm"] = f["ph_dark_mass"] / (qs[-1] + 1e-6)
    return f, rel


def _pores(rel):
    """작은 어두운 블롭: rel 맵의 white top-hat(disk 10) → 큰 2상 결정립은 제거됨.

    작은 2상 결정립도 top-hat을 통과하므로, 성분의 최대 어둡기가 이미지 내 최암부(기공 수준)의
    65% 이상인 성분만 기공으로 인정한다 (기공 깊이 ≈ 2상 깊이의 1.6~2.3배라는 관찰에 근거).
    """
    f = {}
    tp = morphology.white_tophat(rel, morphology.disk(10))
    tmax = float(np.percentile(tp, 99.95))
    darkest = float(np.percentile(rel, 99.95))
    thr = max(0.12, 0.45 * tmax)
    cand = morphology.remove_small_objects(tp > thr, max_size=8)
    lab = measure.label(cand)
    # 느슨한 후보(작은 2상 결정립 포함 가능) — 재현율 높음
    f["pore_cand_count"] = float(lab.max())
    f["pore_cand_frac"] = float(cand.mean())
    keep = [p.label for p in measure.regionprops(lab, intensity_image=rel) if p.intensity_max >= 0.65 * darkest]
    mask = np.isin(lab, keep)
    lab = measure.label(mask)
    props = measure.regionprops(lab, intensity_image=rel)
    f["pore_darkest_rel"] = darkest
    areas = np.array([p.area for p in props], dtype=float)
    f["pore_tophat_max"] = tmax
    f["pore_count"] = float(len(props))
    f["pore_area_frac"] = float(mask.mean())
    f["pore_mean_area"] = float(areas.mean()) if len(areas) else 0.0
    f["pore_max_area"] = float(areas.max()) if len(areas) else 0.0
    if len(props):
        ecc = np.array([p.eccentricity for p in props])
        dep = np.array([p.intensity_mean for p in props])
        f["pore_mean_ecc"] = float(ecc.mean())
        f["pore_mean_depth"] = float(dep.mean())
    else:
        f["pore_mean_ecc"] = 0.0
        f["pore_mean_depth"] = 0.0
    return f, mask


def _orientation_stats(props):
    """면적 가중 2θ 평균 벡터 길이 = 정렬 정도(0~1), 각도는 nuisance."""
    if not props:
        return 0.0, 0.0
    w = np.array([p.area for p in props], dtype=float)
    th = np.array([p.orientation for p in props])
    c = np.sum(w * np.cos(2 * th)) / w.sum()
    s = np.sum(w * np.sin(2 * th)) / w.sum()
    return float(np.hypot(c, s)), float(0.5 * np.arctan2(s, c))


def _grains(d):
    f = {}
    g = filters.sobel(d)
    gn = g / (np.percentile(g, 99) + 1e-6)
    markers = measure.label(morphology.remove_small_objects(gn < 0.08, max_size=15))
    ws = segmentation.watershed(g, markers=markers)
    props = [p for p in measure.regionprops(ws) if p.area >= 20]
    areas = np.array([p.area for p in props], dtype=float)
    n = len(props)
    f["gr_count"] = float(n)
    if n:
        f["gr_area_mean"] = areas.mean()
        f["gr_area_median"] = float(np.median(areas))
        f["gr_area_cv"] = areas.std() / areas.mean()
        f["gr_area_wmean"] = float((areas ** 2).sum() / areas.sum())  # 면적가중 평균
        f["gr_eq_diam_mean"] = float(np.mean([p.equivalent_diameter_area for p in props]))
        f["gr_area_skew"] = float(stats.skew(areas)) if n > 2 else 0.0
        f["gr_area_p90_p10"] = float(np.percentile(areas, 90) / max(np.percentile(areas, 10), 1))
        ar = np.array([p.axis_major_length / max(p.axis_minor_length, 1e-3) for p in props])
        f["gr_aspect_median"] = float(np.median(ar))
        f["gr_aspect_wmean"] = float(np.sum(ar * areas) / areas.sum())
        f["gr_solidity_mean"] = float(np.mean([p.solidity for p in props]))
        align, ang = _orientation_stats(props)
        f["gr_align"] = align
        f["nui_gr_angle"] = ang
    else:
        for k in ["gr_area_mean", "gr_area_median", "gr_area_cv", "gr_area_wmean", "gr_eq_diam_mean",
                  "gr_area_skew", "gr_area_p90_p10", "gr_aspect_median", "gr_aspect_wmean",
                  "gr_solidity_mean", "gr_align", "nui_gr_angle"]:
            f[k] = 0.0
    # 결정립계 밀도 프록시: 정규화 기울기 평균 & black top-hat(얇은 어두운 선)
    contrast = np.percentile(d, 99) - np.percentile(d, 1) + 1e-6
    f["gr_grad_density"] = float(g.mean() / contrast)
    bth = morphology.black_tophat(d, morphology.disk(2)) / contrast
    f["gr_boundary_density"] = float(bth.mean())
    f["gr_boundary_frac"] = float((bth > 0.08).mean())
    return f, ws


def _radial_profile(arr):
    h, w = arr.shape
    yy, xx = np.indices(arr.shape)
    r = np.hypot(yy - h // 2, xx - w // 2).astype(int)
    tbin = np.bincount(r.ravel(), arr.ravel())
    nr = np.bincount(r.ravel())
    return tbin / np.maximum(nr, 1)


def _spectral_raw(raw):
    """원본 PSD에서 백색잡음 바닥(고주파 r∈[100,128] 평균)을 뺀 구조 스펙트럼.

    TV 디노이징이 지워버리는 미세 결정립계 신호를 보존하기 위한 결정립 크기 프록시.
    """
    f = {}
    z = raw - raw.mean()
    win = np.outer(np.hanning(z.shape[0]), np.hanning(z.shape[1]))
    P = np.abs(np.fft.fftshift(np.fft.fft2(z * win))) ** 2
    prof = _radial_profile(P)[1:129]
    floor = prof[99:128].mean()
    sig = np.clip(prof - floor, 0, None)[:96]
    rr = np.arange(1, 97)
    tot = sig.sum() + 1e-12
    f["sp_raw_centroid"] = float((sig * rr).sum() / tot)
    cum = np.cumsum(sig) / tot
    f["sp_raw_median_freq"] = float(rr[min(np.searchsorted(cum, 0.5), 95)])
    f["sp_raw_q90_freq"] = float(rr[min(np.searchsorted(cum, 0.9), 95)])
    f["sp_raw_hf_frac"] = float(sig[16:].sum() / tot)  # r>=17 (주기 < 15px) 비중
    f["sp_raw_snr"] = float(tot / (floor * 96 + 1e-12))
    return f


def _boundary_light(raw, sigma):
    """약한 디노이즈(TV 0.6σ) 후 black top-hat → 미세 결정립계 선 밀도."""
    f = {}
    d2 = denoise_tv_chambolle(raw / 255.0, weight=max(0.01, 0.6 * sigma / 255.0)) * 255.0
    bth = morphology.black_tophat(d2, morphology.disk(2))
    contrast = np.percentile(d2, 99) - np.percentile(d2, 1) + 1e-6
    f["gr_lb_density"] = float(bth.mean() / contrast)
    f["gr_lb_density_snr"] = float(bth.mean() / (sigma + 1e-6))
    # 행/열 방향 라인 인터셉트: 경계 마스크의 0→1 전이 수
    mask = bth > (np.percentile(bth, 75) + 1.0 * bth.std())
    nx = np.abs(np.diff(mask.astype(np.int8), axis=1)).sum() / 2
    ny = np.abs(np.diff(mask.astype(np.int8), axis=0)).sum() / 2
    L = mask.shape[0] * mask.shape[1]
    f["gr_lb_intercept_len"] = float(2 * L / max(nx + ny, 1))  # 평균 인터셉트 길이(px)
    return f


def _spectral(d):
    f = {}
    z = d - d.mean()
    win = np.outer(np.hanning(z.shape[0]), np.hanning(z.shape[1]))
    F = np.fft.fftshift(np.fft.fft2(z * win))
    P = np.abs(F) ** 2
    prof = _radial_profile(P)[1:129]  # r=1..128
    rr = np.arange(1, 129)
    tot = prof.sum() + 1e-12
    f["sp_centroid"] = float((prof * rr).sum() / tot)
    cum = np.cumsum(prof) / tot
    f["sp_median_freq"] = float(rr[np.searchsorted(cum, 0.5)])
    f["sp_q80_freq"] = float(rr[np.searchsorted(cum, 0.8)])
    for lo, hi in zip(PSD_BANDS[:-1], PSD_BANDS[1:]):
        f[f"sp_band_{lo}_{hi}"] = float(prof[lo - 1:hi - 1].sum() / tot)
    # 자기상관 (Wiener–Khinchin), 반지름 프로파일로 상관 길이
    acf = np.real(np.fft.fftshift(np.fft.ifft2(np.abs(np.fft.fft2(z)) ** 2)))
    acf /= acf.max()
    ap = _radial_profile(acf)
    f["sp_acf_len_0.5"] = float(np.argmax(ap < 0.5)) if (ap < 0.5).any() else 128.0
    f["sp_acf_len_0.2"] = float(np.argmax(ap < 0.2)) if (ap < 0.2).any() else 128.0
    # PSD 관성 텐서 → 이방성
    h, w = P.shape
    yy, xx = np.indices(P.shape)
    yy = yy - h // 2
    xx = xx - w // 2
    rad = np.hypot(yy, xx)
    m = (rad >= 2) & (rad <= 64)
    Pm = P * m
    Ixx = (Pm * xx ** 2).sum()
    Iyy = (Pm * yy ** 2).sum()
    Ixy = (Pm * xx * yy).sum()
    ev = np.linalg.eigvalsh(np.array([[Ixx, Ixy], [Ixy, Iyy]]))
    f["an_psd_ratio"] = float(ev[1] / max(ev[0], 1e-12))
    # ACF 이방성: 0.5 등고선 영역의 2차 모멘트
    am = acf > 0.5
    lab = measure.label(am)
    c = lab[h // 2, w // 2]
    if c > 0:
        rp = measure.regionprops((lab == c).astype(int))[0]
        f["an_acf_aspect"] = float(rp.axis_major_length / max(rp.axis_minor_length, 1e-3))
        f["nui_acf_angle"] = float(rp.orientation)
    else:
        f["an_acf_aspect"] = 1.0
        f["nui_acf_angle"] = 0.0
    return f


def _structure_tensor(d):
    f = {}
    gy, gx = np.gradient(ndi.gaussian_filter(d, 1.5))
    Jxx, Jyy, Jxy = (gx * gx).mean(), (gy * gy).mean(), (gx * gy).mean()
    tr = Jxx + Jyy + 1e-12
    f["an_coherence"] = float(np.sqrt((Jxx - Jyy) ** 2 + 4 * Jxy ** 2) / tr)
    f["nui_st_angle"] = float(0.5 * np.arctan2(2 * Jxy, Jxx - Jyy))
    # 국소 coherence 평균 (rho=6)
    sxx = ndi.gaussian_filter(gx * gx, 6)
    syy = ndi.gaussian_filter(gy * gy, 6)
    sxy = ndi.gaussian_filter(gx * gy, 6)
    lc = np.sqrt((sxx - syy) ** 2 + 4 * sxy ** 2) / (sxx + syy + 1e-9)
    f["an_local_coherence"] = float(lc.mean())
    return f


def _heterogeneity(d, rel, g):
    f = {}
    T = 4
    s = d.shape[0] // T
    tiles_mean, tiles_dark, tiles_grad = [], [], []
    for i in range(T):
        for j in range(T):
            sl = (slice(i * s, (i + 1) * s), slice(j * s, (j + 1) * s))
            tiles_mean.append(d[sl].mean())
            tiles_dark.append((rel[sl] > 0.15).mean())
            tiles_grad.append(g[sl].mean())
    tiles_mean, tiles_dark, tiles_grad = map(np.array, (tiles_mean, tiles_dark, tiles_grad))
    f["het_mean_cv"] = float(tiles_mean.std() / (tiles_mean.mean() + 1e-6))
    f["het_dark_std"] = float(tiles_dark.std())
    f["het_grad_cv"] = float(tiles_grad.std() / (tiles_grad.mean() + 1e-9))
    return f


def _texture(d):
    f = {}
    lo, hi = np.percentile(d, [0.5, 99.5])
    q = np.clip((d - lo) / (hi - lo + 1e-6) * 31, 0, 31).astype(np.uint8)
    dists = [1, 3, 6, 12]
    glcm = feature.graycomatrix(q, distances=dists, angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                                levels=32, symmetric=True, normed=True)
    for prop in ["contrast", "homogeneity", "energy", "correlation"]:
        v = feature.graycoprops(glcm, prop).mean(axis=1)  # 각도 평균 → 회전 불변
        for dd, vv in zip(dists, v):
            f[f"tx_glcm_{prop}_d{dd}"] = float(vv)
    lbp = feature.local_binary_pattern(q, P=8, R=1, method="uniform")
    hist, _ = np.histogram(lbp, bins=10, range=(0, 10), density=True)
    for k, v in enumerate(hist):
        f[f"tx_lbp_{k}"] = float(v)
    return f


def extract_features(img: np.ndarray) -> dict:
    raw, d, sigma, clo = preprocess(img)
    m = matrix_level(clo)
    f = {}
    f.update(_quality(raw, d, sigma))
    ph, rel = _phase(clo, m)
    f.update(ph)
    pf, _ = _pores(rel)
    f.update(pf)
    gf, _ = _grains(d)
    f.update(gf)
    f.update(_boundary_light(raw, sigma))
    f.update(_spectral(d))
    f.update(_spectral_raw(raw))
    f.update(_structure_tensor(d))
    f.update(_heterogeneity(d, rel, filters.sobel(d)))
    f.update(_texture(d))
    return f


def debug_maps(img: np.ndarray) -> dict:
    """시각화용 중간 산출물."""
    raw, d, sigma, clo = preprocess(img)
    m = matrix_level(clo)
    _, rel = _phase(clo, m)
    _, pore_mask = _pores(rel)
    _, ws = _grains(d)
    return dict(raw=raw, denoised=d, closed=clo, rel=rel, pore_mask=pore_mask, watershed=ws, sigma=sigma, matrix=m)
