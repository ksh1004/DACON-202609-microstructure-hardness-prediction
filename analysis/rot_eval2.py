"""회전 대조 실험: 각도별 단독 OOF + 0° 특징과의 차이 크기 (표준화 단위)."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
D = "outputs/probe2_rot/unet_cm7_am22_l1_i10000_boot_fflowpass48_low/"
NL, NB, C = 7, 11, 1024
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def design(A, t):
    A = A.reshape(len(A), NL, NB, C)[:, [1, 2, 3, 4]]
    outs, ref = [], None
    for blocks, g in GR:
        Z = A[:, :, list(blocks)].reshape(len(A), -1)
        mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10
        Z = np.clip((Z[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(Z * s)
    return np.hstack(outs)
def cvoof(A, Atest=None):
    """Atest가 있으면: A로 학습, Atest(같은 이미지의 다른 보기)로 평가"""
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k
        if Atest is None:
            Z = design(A, t); o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]).predict(Z[v])
        else:
            both = np.vstack([A, Atest]); tt = np.r_[t, np.zeros(len(y), bool)]
            Z = design(both, tt); o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[:len(y)][t], y[t]).predict(Z[len(y):][v])
    return o
L = lambda a: np.load(D + f"a{a:g}_train.npy").astype(np.float64)
A0 = L(0); base = cvoof(A0); rep("0°", base)
sd = A0.std(0) + 1e-8
for a in (90, 5, 10, 22.5, 45):
    Aa = L(a)
    dz = (Aa - A0) / sd
    blk = np.sqrt((dz.reshape(len(y), NL, NB, C) ** 2).mean((0, 1, 3)))  # 블록별 이동 크기 (표준편차 단위)
    o = cvoof(Aa); rep(f"{a:g}° 단독", o, base)
    print(f"      0° 대비 특징 이동 RMS(표준편차 단위) 전체 {np.sqrt((dz**2).mean()):.3f} | 블록별 " + " ".join(f"{b:.2f}" for b in blk)
          + f" | 평균 이동(체계적) {np.abs(dz.mean(0)).mean():.3f}")
    if a in (5, 45): rep(f"      0°로 학습 → {a:g}° 보기로 평가", cvoof(A0, Aa), base)
