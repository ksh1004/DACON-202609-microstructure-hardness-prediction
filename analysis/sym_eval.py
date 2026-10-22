"""D4 대칭화 특징 평가. 기준: 보기별 통계 평균(naive) — 0° 묶음(조정에 쓰임, 낙관), 180° 묶음(여집합, 정직), 8보기.
대칭화: symA(0° 묶음 4보기), symB(여집합 4보기), symALL(8보기). 신뢰도(ICC)는 서로 겹치지 않는 A·B로 계산."""
import sys, os; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
stem = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
sfx = sys.argv[1] if len(sys.argv) > 1 else ""
D = f"outputs/probe2_rot/{stem}/"
NL, NB, C = 7, 11, 1024
names = ["평균", "std", "het2", "het4", "het8", "het4내부std", "2상평균", "2상std", "기지평균", "기지std", "기여량"]
def design(A, t, GR):
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
GR0 = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def cvoof(A, GR=GR0):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k
        Z = design(A, t, GR); o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]).predict(Z[v])
    return o
L = lambda n: np.load(D + n + "_train.npy").astype(np.float64)
nA, nB = L("a0"), L("a180"); n8 = (nA + nB) / 2
sA, sB, s8 = L("symA" + sfx), L("symB" + sfx), L("symALL" + sfx)
def icc(a, b):
    noise = ((a - b) ** 2).mean(0) / 2; return (1 - noise / np.maximum(a.var(0), 1e-12)).reshape(NL, NB, C)
print("블록별 신뢰도(ICC, 층 위치 1~4 평균) — 기존(보기별 통계 평균) → 대칭화")
i0, i1 = icc(nA, nB), icc(sA, sB)
print("  " + " | ".join(f"{n} {i0[1:5, b].mean():.2f}→{i1[1:5, b].mean():.2f}" for b, n in enumerate(names)))
R = {}
for nm, A in (("기존 0° 묶음 (조정에 쓰인 것)", nA), ("기존 여집합 묶음 (정직한 기준)", nB), ("기존 8보기", n8),
              ("대칭화 0° 묶음", sA), ("대칭화 여집합 묶음", sB), ("대칭화 8보기", s8)):
    R[nm] = cvoof(A)
b4, b8 = R["기존 여집합 묶음 (정직한 기준)"], R["기존 8보기"]
for nm in R: rep(nm, R[nm], b4 if "여집합" in nm or "0°" in nm else b8)
np.savez("analysis/sym_oof" + sfx + ".npz", **{f"r{i}": R[k] for i, k in enumerate(R)})
# 대칭화 8보기에서 블록 구성 민감도 (조정이 아니라 구조 확인용): 불균일성 블록 비중
print("\n대칭화 8보기 — 블록 구성 변화")
for nm, GR in (("hetero 없이 (평균·std + 상풀링 + 기여량)", (((0, 1), 1.0), ((6, 7, 8, 9), 0.25 * 3), ((10,), 0.05 * 3))),
               ("기본", GR0),
               ("hetero 블록을 별도 묶음 G=1", (((0, 1), 1.0), ((2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.75), ((10,), 0.15))),
               ("hetero 별도 묶음 G=2", (((0, 1), 1.0), ((2, 3, 4, 5), 2.0), ((6, 7, 8, 9), 0.75), ((10,), 0.15))),
               ("hetero 별도 묶음 G=4", (((0, 1), 1.0), ((2, 3, 4, 5), 4.0), ((6, 7, 8, 9), 0.75), ((10,), 0.15)))):
    rep("  " + nm, cvoof(s8, GR), R["대칭화 8보기"])
print("기존 8보기 — 같은 구성 변화")
for nm, GR in (("hetero 없이", (((0, 1), 1.0), ((6, 7, 8, 9), 0.75), ((10,), 0.15))),
               ("hetero 별도 묶음 G=2", (((0, 1), 1.0), ((2, 3, 4, 5), 2.0), ((6, 7, 8, 9), 0.75), ((10,), 0.15)))):
    rep("  " + nm, cvoof(n8, GR), b8)
