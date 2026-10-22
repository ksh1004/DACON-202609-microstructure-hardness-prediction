"""라벨 이상치 제거 실험: 폴드 train 안에서 LOO 잔차가 큰 상위 q% 표본을 빼고 Ridge 재적합 → 바깥 폴드 예측. (DINOv2 16보기 σ48 블록)"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
P = "outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass48_low_sh2/emb.npz"
A = np.load(P)["train"].astype(np.float64).reshape(len(y), 7, 11, 1024)[:, 1:5]
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def design(t):
    outs, ref = [], None
    for blocks, g in GR:
        B = A[:, :, list(blocks)].reshape(len(A), -1); mu, sd = B[t].mean(0), B[t].std(0); keep = sd > 1e-10
        B = np.clip((B[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(B * s)
    return np.hstack(outs)
res = {q: np.zeros(len(y)) for q in (0, 1, 2, 4, 8)}; res.update({f"hi{q}": np.zeros(len(y)) for q in (2, 4)})
for k in range(5):
    t, v = np.where(fold != k)[0], np.where(fold == k)[0]; Z = design(fold != k)
    r0 = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]); res[0][v] = r0.predict(Z[v]); al = r0.alpha_
    Zc = Z[t] - Z[t].mean(0); K = Zc @ Zc.T; n = len(t); H = K @ np.linalg.inv(K + al * np.eye(n)) + 1.0 / n
    loo = (y[t] - r0.predict(Z[t])) / (1 - np.diag(H))
    order = np.argsort(-np.abs(loo))
    for q in (1, 2, 4, 8):
        keep = np.ones(n, bool); keep[order[:int(round(n * q / 100))]] = False
        res[q][v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t][keep], y[t][keep]).predict(Z[v])
    # 대조: LOO 잔차가 큰 표본을 빼는 대신, 예측값 쪽으로 절반 당긴 라벨로 대체(완화)
    for q in (2, 4):
        yy = y[t].copy(); idx = order[:int(round(n * q / 100))]; yy[idx] = yy[idx] - 0.5 * loo[idx]
        res[f"hi{q}"][v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], yy).predict(Z[v])
    print(f"폴드 {k}: α={al:.0f}, |LOO 잔차| 상위 1% 평균 {np.abs(loo[order[:4]]).mean():.1f}, 중앙값 {np.median(np.abs(loo)):.1f}", flush=True)
rep("기준 (전부 사용)", res[0])
for q in (1, 2, 4, 8): rep(f"LOO 잔차 상위 {q}% 제거", res[q], res[0])
for q in (2, 4): rep(f"상위 {q}% 라벨을 절반만 당김", res[f"hi{q}"], res[0])
