"""정직한 시험대(여집합 보기 묶음 a180)에서 회귀기 변형: PLS, 상관 선별 Ridge, 상관 가중 Ridge. 선별·가중은 폴드 안에서만."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
D = "outputs/probe2_rot/unet_cm7_am22_l1_i10000_boot_fflowpass48_low/"
NL, NB, C = 7, 11, 1024
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
A = np.load(D + "a180_train.npy").astype(np.float64)
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
Zs = [design(A, fold != k) for k in range(5)]
def cv(fit_predict):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k
        o[v] = fit_predict(Zs[k][t], y[t], Zs[k][v])
    return o
ridge = lambda Xt, yt, Xv: RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Xt, yt).predict(Xv)
base = cv(ridge); rep("Ridge (기준)", base)
for nc in (2, 4, 6, 10, 16):
    rep(f"PLS {nc}성분", cv(lambda Xt, yt, Xv: PLSRegression(n_components=nc, scale=False).fit(Xt, yt).predict(Xv).ravel()), base)
def screen(top):
    def f(Xt, yt, Xv):
        c = np.abs((Xt - Xt.mean(0)).T @ (yt - yt.mean())) / (Xt.std(0) * len(yt) * yt.std() + 1e-12)
        idx = np.argsort(-c)[:top]; return ridge(Xt[:, idx], yt, Xv[:, idx])
    return f
for top in (1000, 5000, 15000): rep(f"상관 상위 {top}개 → Ridge", cv(screen(top)), base)
def weighted(p):
    def f(Xt, yt, Xv):
        c = np.abs((Xt - Xt.mean(0)).T @ (yt - yt.mean())) / (Xt.std(0) * len(yt) * yt.std() + 1e-12)
        w = (c / c.mean()) ** p; return ridge(Xt * w, yt, Xv * w)
    return f
for p in (0.5, 1.0, 2.0): rep(f"|상관|^{p} 가중 Ridge", cv(weighted(p)), base)
