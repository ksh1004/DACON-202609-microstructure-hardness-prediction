"""다른 백본의 정직한 비교 (σ48, D4 8보기, phase2m 블록 구성): 단독 OOF·층 구간·DINOv2-L(8보기)과의 블렌드.
사용: python model_eval.py <파일 접미사(_m...)> <층 수 NL> <채널 C> [층 번호들...]"""
import sys; sys.path.insert(0, "analysis")
ARGS = sys.argv[1:]; sys.argv = sys.argv[:1]  # view_eval이 argv[1]을 입력 이미지 이름으로 읽으므로 비워 둔다
import view_eval as VE
from view_eval import *
tag, NLm, Cm = ARGS[0], int(ARGS[1]), int(ARGS[2]); LN = ARGS[3:] or [str(i) for i in range(NLm)]
L8 = avg(S(ALL)); oL = cvoof(L8); rep("DINOv2-L 8보기 (기준)", oL)
X = np.mean([np.load(D + f"train_s0_k{k}{tag}.npy").astype(np.float32) for k in range(8)], 0).reshape(len(y), NLm, 11, Cm)
def ev(layers, GR=GR0):
    VE.NL, VE.C = len(layers), Cm
    o = VE.cvoof(X[:, list(layers)].reshape(len(y), -1), GR); VE.NL, VE.C = 4, 1024; return o
res = {}
print("단일 층:", " ".join(f"L{LN[i]} {np.sqrt(((ev((i,)) - y) ** 2).mean()):.3f}" for i in range(NLm)), flush=True)
cands = {"전체": tuple(range(NLm))}
for a in range(NLm - 3): cands[f"{LN[a]}~{LN[a+3]}"] = tuple(range(a, a + 4))
for nm, ls in cands.items():
    res[nm] = ev(ls); rep(f"{tag[2:22]} 층 {nm}", res[nm], oL); sys.stdout.flush()
best = min(res, key=lambda k: ((res[k] - y) ** 2).mean()); oB = res[best]
print(f"최선 층 구간 {best} | DINOv2-L과 잔차상관 {np.corrcoef(oL - y, oB - y)[0, 1]:.3f}, 예측 RMS차 {np.sqrt(((oL - oB) ** 2).mean()):.2f}")
for w in (0.2, 0.3, 0.5, 0.7): rep(f"예측 블렌드 L {1-w:.1f} + 새 모델 {w:.1f}", (1 - w) * oL + w * oB, oL)
np.save(f"analysis/model_oof{tag}.npy", oB)
# hetero·상풀링 없이 평균·std만 (모델 자체의 정보량 비교)
H2 = (((0, 1), 1.0),)
VE.NL, VE.C = 4, 1024; oL2 = VE.cvoof(L8, H2)
print("평균·std 블록만: DINOv2-L %.3f vs 새 모델(%s) %.3f" % (np.sqrt(((oL2 - y) ** 2).mean()), best, np.sqrt(((ev(cands[best], H2) - y) ** 2).mean())))
