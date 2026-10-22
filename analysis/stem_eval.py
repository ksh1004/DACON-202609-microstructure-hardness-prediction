"""다른 입력 이미지(전처리 변형)의 8보기 특징을 기존(σ48 기준)과 정직하게 비교. 여러 stem을 주면 특징 평균도 평가.
사용: python stem_eval.py <stem1> [<stem2> ...]"""
import sys, os; sys.path.insert(0, "analysis")
STEMS = sys.argv[1:]; sys.argv = sys.argv[:1]
from view_eval import *
P = "outputs/probe2_views/"
def L8(stem, shifts=(0,)):
    fs = [P + stem + f"/train_s{s}_k{k}.npy" for s in shifts for k in range(8)]
    return np.mean([np.load(f).astype(np.float32) for f in fs], 0)
base8 = avg(S(ALL)); o8 = cvoof(base8); rep("기존 8보기", o8)
o16 = cvoof(avg(S(ALL, (0, 1)))); rep("기존 16보기 (참고)", o16, o8)
X = {}
for st in STEMS:
    X[st] = L8(st); o = cvoof(X[st]); rep(f"{st[:34]} 8보기", o, o8); sys.stdout.flush()
    np.save(f"analysis/stem_oof_{st[:20]}.npy", o)
if len(STEMS) > 1:
    oa = cvoof(np.mean([X[s] for s in STEMS], 0)); rep(f"위 {len(STEMS)}개 특징 평균 ({8*len(STEMS)}보기)", oa, o16)
    np.save("analysis/stem_oof_avg.npy", oa)
F_ = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[ids]
last = oa if len(STEMS) > 1 else o
ref = o16 if len(STEMS) > 1 else o8
print("원본 노이즈 구간별 RMSE (기준 → 새 입력):", " | ".join(f"{lo:g}~{hi:g}: {np.sqrt(((ref-y)[(sig>=lo)&(sig<hi)]**2).mean()):.2f}→{np.sqrt(((last-y)[(sig>=lo)&(sig<hi)]**2).mean()):.2f}" for lo, hi in ((0, 5.5), (5.5, 7), (7, 8.5), (8.5, 10), (10, 14), (14, 30))))
