"""D4 보기 묶음의 위약 시험: 0/90/180/270° 묶음은 통계적으로 동등 → OOF 산포 = 보기 선택 잡음. 8방향 전체 평균의 효과.
또한 보기 잡음 크기를 재서(0° vs 180° 묶음은 서로 겹치지 않는 4보기) 특징별 신뢰도(ICC)를 계산."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from rot_eval2 import design, cvoof, L, NL, NB, C  # rot_eval2는 import 시 표를 한 번 출력함
print("\n================ D4 위약 시험 ================")
A = {a: L(a) for a in (0, 90, 180, 270)}
O = {a: cvoof(A[a]) for a in A}
for a in A: rep(f"{a}° 묶음 (4보기)", O[a], O[0])
full = (A[0] + A[180]) / 2; full2 = (A[90] + A[270]) / 2
print("8보기 평균 두 가지 계산의 차이(검산) RMS: %.2e" % np.sqrt(((full - full2) ** 2).mean()))
o8 = cvoof(full); rep("8보기 전체 평균 (특징)", o8, O[0])
rep("4묶음 예측 평균", np.mean([O[a] for a in A], 0), O[0])
# 겹치지 않는 묶음(0° vs 180°)으로 특징 신뢰도: ICC = 1 - Var(차)/2 / Var(평균+...)
d = A[0] - A[180]; s = (A[0] + A[180]) / 2
noise = (d ** 2).mean(0) / 2          # 4보기 평균 특징의 보기 잡음 분산
tot = A[0].var(0)                     # 4보기 평균 특징의 전체 분산 (신호+잡음)
icc = 1 - noise / np.maximum(tot, 1e-12)
icc_b = icc.reshape(NL, NB, C)
print("\n블록별 4보기 평균 특징의 신뢰도(ICC: 1=보기와 무관, 0=전부 보기 잡음) — 층 위치 1~4 평균")
names = ["평균", "std", "het2", "het4", "het8", "het4내부std", "2상평균", "2상std", "기지평균", "기지std", "기여량"]
print("  " + " | ".join(f"{n} {icc_b[1:5, b].mean():.2f}" for b, n in enumerate(names)))
print("  층별(9,11,13,15,17,19,21) 평균 ICC: " + " ".join(f"{icc_b[l].mean():.2f}" for l in range(NL)))
np.save("analysis/icc_ff48.npy", icc.astype(np.float32))
# 예측 수준의 보기 잡음: 같은 모델(0° 묶음으로 학습)에 180° 묶음 특징을 넣었을 때 예측 변화
p_sw = cvoof(A[0], A[180])
print("\n0° 묶음으로 학습한 모델: 0° 특징 예측 vs 180° 특징 예측의 RMS 차이 %.2f (예측 표준편차 %.2f)" % (np.sqrt(((p_sw - O[0]) ** 2).mean()), O[0].std()))
rep("0° 학습 → 180° 보기로 평가", p_sw, O[0])
rep("0° 학습 → 두 보기 예측 평균", (p_sw + O[0]) / 2, O[0])
np.savez("analysis/d4_placebo.npz", **{str(a): O[a] for a in O}, full=o8)
