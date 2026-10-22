"""상 조건부 풀링 블록 구성 변형을 σ 4종에서 비교 (폴드별 커널 재사용으로 빠르게)."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
def krr_oof(blocks):  # blocks: [(A, G)], 첫 묶음 기준 상대 분산
    out = np.zeros(len(y))
    for k in range(5):
        t = fold != k; K = 0; Kv = 0; ref = None
        for A, g in blocks:
            mu, sd = A[t].mean(0), A[t].std(0); keep = sd > 1e-10
            Z = np.clip((A[:, keep] - mu[keep]) / sd[keep], -6, 6)
            if ref is None: ref, s2 = keep.sum(), 1.0
            else: s2 = g * ref / keep.sum()
            K = K + s2 * (Z[t] @ Z[t].T); Kv = Kv + s2 * (Z[~t] @ Z[t].T)
        best = None; ym = y[t].mean()
        for al in np.logspace(0, 6, 19):
            Mi = np.linalg.inv(K + al * np.eye(t.sum())); a = Mi @ (y[t] - ym); loo = a / np.diag(Mi)
            if best is None or (loo ** 2).mean() < best[0]: best = ((loo ** 2).mean(), a)
        out[~t] = Kv @ best[1] + ym
    return out
V = {"기준 het L11-17 + pp L11-17 G.25": ((1, 2, 3, 4), (1, 2, 3, 4), (6, 7, 8, 9), .25),
     "pp 층 9-21 전체": ((1, 2, 3, 4), (0, 1, 2, 3, 4, 5, 6), (6, 7, 8, 9), .25),
     "pp 층 13-19": ((1, 2, 3, 4), (2, 3, 4, 5), (6, 7, 8, 9), .25),
     "pp 층 9-15": ((1, 2, 3, 4), (0, 1, 2, 3), (6, 7, 8, 9), .25),
     "pp 평균만": ((1, 2, 3, 4), (1, 2, 3, 4), (6, 8), .25),
     "pp 표준편차만": ((1, 2, 3, 4), (1, 2, 3, 4), (7, 9), .25),
     "pp 2상만(평균·std)": ((1, 2, 3, 4), (1, 2, 3, 4), (6, 7), .25),
     "pp 기지만(평균·std)": ((1, 2, 3, 4), (1, 2, 3, 4), (8, 9), .25),
     "G 0.1": ((1, 2, 3, 4), (1, 2, 3, 4), (6, 7, 8, 9), .1), "G 1.0": ((1, 2, 3, 4), (1, 2, 3, 4), (6, 7, 8, 9), 1.0)}
res = {v: [] for v in V}
for s in FF:
    z = np.load(f"outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2_Sunet_cm7_am22_l1_i10000_boot_fflowpass{s}_low/emb.npz")["train"].astype(np.float64)
    z = z.reshape(len(z), 7, 10, 1024)
    for v, (hl, pl, pb, g) in V.items():
        o = krr_oof([(z[:, hl][:, :, :6].reshape(len(z), -1), 1), (z[:, pl][:, :, pb].reshape(len(z), -1), g)])
        res[v].append(np.sqrt(((o - y) ** 2).mean()))
    print(f"σ{s} 완료", flush=True)
print(f"{'변형':34s} " + " ".join(f"σ{s:<5d}" for s in FF) + " 평균   기준 대비")
b = np.array(res["기준 het L11-17 + pp L11-17 G.25"])
for v, r in res.items(): print(f"{v:34s} " + " ".join(f"{x:6.3f}" for x in r) + f" {np.mean(r):6.3f}  {np.mean(r) - b.mean():+.3f} ({(np.array(r) < b - 1e-9).sum()}/4 개선)")
