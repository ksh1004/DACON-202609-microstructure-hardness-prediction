"""16보기 특징(mv16 emb, 9~21층 7개)에서 층 구간 재검증. v15b 블록 구성(H 1 + 상풀링 0.25 + 기여량 0.05). σ48과 σ64 두 곳에서 일관성 확인."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
NB, C = 11, 1024
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def load(s):
    p = f"outputs/probe2/vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass{s}_low_sh2/emb.npz"
    return np.load(p)["train"].astype(np.float64).reshape(len(y), 7, NB, C)
def cvoof(A, layers):
    A = A[:, list(layers)]; o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k; outs, ref = [], None
        for blocks, g in GR:
            Z = A[:, :, list(blocks)].reshape(len(A), -1); mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10
            Z = np.clip((Z[:, keep] - mu[keep]) / sd[keep], -6, 6)
            if ref is None: ref, s = keep.sum(), 1.0
            else: s = np.sqrt(g * ref / keep.sum())
            outs.append(Z * s)
        Z = np.hstack(outs); o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]).predict(Z[v])
    return o
LN = [9, 11, 13, 15, 17, 19, 21]
sets = {"11-17 (현재)": (1, 2, 3, 4), "9-21 전체": tuple(range(7)), "9-15": (0, 1, 2, 3), "13-19": (2, 3, 4, 5), "15-21": (3, 4, 5, 6), "11-19": (1, 2, 3, 4, 5), "9-17": (0, 1, 2, 3, 4), "13-17": (2, 3, 4), "11-15": (1, 2, 3), "11·15·19": (1, 3, 5)}
for s in (48, 64):
    A = load(s); base = None; print(f"\n== σ{s} ==")
    for nm, ls in sets.items():
        o = cvoof(A, ls); base = o if base is None else base; rep(f"층 {nm}", o, base); sys.stdout.flush()
    print("단일 층:", " ".join(f"L{LN[i]} {np.sqrt(((cvoof(A, (i,)) - y) ** 2).mean()):.3f}" for i in range(7)), flush=True)
