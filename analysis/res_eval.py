"""입력 해상도별 8보기(D4) 특징의 정직한 비교 + 해상도 간 특징 평균 / 예측 평균."""
import sys, os; sys.path.insert(0, "analysis")
from view_eval import *
def L8(size):
    sfx = "" if size == 448 else f"_r{size}"
    return np.mean([np.load(D + f"train_s0_k{k}{sfx}.npy").astype(np.float32) for k in range(8)], 0)
sizes = [s for s in (392, 420, 448, 476, 518) if os.path.exists(D + f"train_s0_k7{'' if s == 448 else f'_r{s}'}.npy")]
X = {s: L8(s) for s in sizes}; O = {s: cvoof(X[s]) for s in sizes}
b = O[448]
for s in sizes: rep(f"{s} 단독 (D4 8보기)", O[s], b)
ref16 = cvoof(avg(S(ALL, (0, 1)))); rep("448 16보기 (D4×이동2) — 보기 수 대조군", ref16, b)
for combo in ((420, 448, 476), (448, 476), (420, 448), (392, 448, 518), tuple(sizes)):
    if all(c in X for c in combo):
        rep("특징 평균 " + "+".join(map(str, combo)), cvoof(np.mean([X[c] for c in combo], 0)), ref16)
        rep("예측 평균 " + "+".join(map(str, combo)), np.mean([O[c] for c in combo], 0), ref16)
        sys.stdout.flush()
np.savez("analysis/res_oof.npz", **{str(s): O[s] for s in sizes}, ref16=ref16)
