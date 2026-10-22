"""DINOv2-L + 다른 백본을 한 Ridge로 결합(특징 수준) vs 예측 블렌드. σ48, D4 8보기, train만.
사용: python joint_eval.py <접미사> <NL> <C> <층 위치들(쉼표)>"""
import sys; sys.path.insert(0, "analysis")
ARGS = sys.argv[1:]; sys.argv = sys.argv[:1]
import view_eval as VE
from view_eval import *
tag, NLm, Cm = ARGS[0], int(ARGS[1]), int(ARGS[2]); pos = [int(v) for v in ARGS[3].split(",")]
L8 = avg(S(ALL))
X = np.mean([np.load(D + f"train_s0_k{k}{tag}.npy").astype(np.float32) for k in range(8)], 0).reshape(len(y), NLm, 11, Cm)[:, pos].reshape(len(y), -1)
def des(A, t, nl, c):
    VE.NL, VE.C = nl, c; Z = VE.design(A, t); VE.NL, VE.C = 4, 1024; return Z
def cv(gS):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k
        ZL = des(L8, t, 4, 1024); ZS = des(X, t, len(pos), Cm)
        nL, nS = 4 * 6 * 1024, len(pos) * 6 * Cm
        Z = np.hstack([ZL, ZS * np.sqrt(gS * nL / nS)]) if gS > 0 else ZL
        o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]).predict(Z[v])
    return o
oL = cv(0); rep("DINOv2-L 단독", oL)
VE.NL, VE.C = len(pos), Cm; oS = VE.cvoof(X); VE.NL, VE.C = 4, 1024; rep("새 모델 단독", oS, oL)
rep("예측 블렌드 0.7 : 0.3", .7 * oL + .3 * oS, oL)
for g in (0.25, 0.5, 1.0, 2.0):
    rep(f"한 Ridge로 결합 (새 모델 상대 분산 {g})", cv(g), oL); sys.stdout.flush()
