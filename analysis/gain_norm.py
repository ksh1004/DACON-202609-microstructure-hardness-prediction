"""이미지별 노출 정규화: 기지 밝기(5x5 닫기 후 70 분위)를 고정값 160으로 맞추는 곱셈 보정 (이미지별 고정 규칙, 라벨·다른 이미지 미사용).
사용: python gain_norm.py <입력 npz> [--mode gain|affine]  → outputs/denoised/<stem>_gn.npz (float32)"""
import sys, numpy as np
from scipy import ndimage as ndi
src = sys.argv[1]; z = np.load(src); out = {}
for sp in ("train", "test"):
    X = z[sp].astype(np.float32); Y = np.empty_like(X)
    for i, x in enumerate(X):
        m = np.quantile(ndi.grey_closing(x, size=5), .7); Y[i] = np.clip(x * (160.0 / m), 0, 255)
    out[sp] = Y; print(sp, X.shape, "기지 밝기 보정 전 5~95%:", np.quantile([np.quantile(ndi.grey_closing(x, size=5), .7) for x in X[:200]], [.05, .95]).round(1))
np.savez(src.replace(".npz", "_gn.npz"), **out); print("->", src.replace(".npz", "_gn.npz"))
