"""twopath.kmat만 따로 (twopath는 import 시 U·DRUNet 경로의 보기별 특징을 읽으므로, 그 파일이 없는 PC에서도 쓰도록)."""
import numpy as np
from blend_eval import fold
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def kmat(X, NL, C):
    X = X.astype(np.float64).reshape(500, NL, 11, C); Ks = []
    for k in range(5):
        t = fold != k; K = np.zeros((500, 500))
        for blocks, g in GR:
            Z = X[:, :, list(blocks)].reshape(500, -1); mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10; Z = np.clip((Z[:, keep] - mu[keep]) / sd[keep], -6, 6); K += g * (Z @ Z.T) / keep.sum()
        Ks.append(K)
    return np.array(Ks)
