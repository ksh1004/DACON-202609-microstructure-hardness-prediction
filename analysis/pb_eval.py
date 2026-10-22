"""위치 편향 제거 특징 평가: 같은 보기 묶음에서 편향 제거 전/후 비교 + 신뢰도(ICC)."""
import sys; sys.path.insert(0, "analysis")
import view_eval as VE
from view_eval import *
sfx = sys.argv[1] if len(sys.argv) > 1 else "_pb"
_P = {}
def viewp(s, k):
    if (s, k) not in _P: _P[(s, k)] = np.load(D + f"train_s{s}_k{k}{sfx}.npy")
    return _P[(s, k)]
def avgp(views):
    a = np.zeros((len(y), NL * NB * C), np.float32)
    for s, k in views: a += viewp(s, k)
    return a / len(views)
names = ["평균", "std", "het2", "het4", "het8", "het4내부std", "2상평균", "2상std", "기지평균", "기지std", "기여량"]
def icc(a, b):
    noise = ((a - b) ** 2).mean(0) / 2; return (1 - noise / np.maximum(a.var(0), 1e-12)).reshape(NL, NB, C)
i0, i1 = icc(avg(S(A_)), avg(S(B_))), icc(avgp(S(A_)), avgp(S(B_)))
print("블록별 4보기 평균 특징 신뢰도(ICC): 기존 → 편향 제거")
print("  " + " | ".join(f"{n} {i0[:, b].mean():.2f}→{i1[:, b].mean():.2f}" for b, n in enumerate(names)))
s0, s1 = icc(view(0, 0), view(0, 2)), icc(viewp(0, 0), viewp(0, 2))
print("단일 보기(k=0 vs k=2, 180° 회전) 신뢰도: 기존 → 편향 제거")
print("  " + " | ".join(f"{n} {s0[:, b].mean():.2f}→{s1[:, b].mean():.2f}" for b, n in enumerate(names)))
rm = lambda o: np.sqrt(((o - y) ** 2).mean())
print("\n단일 보기 OOF (기존 → 편향 제거)")
a0, a1 = [], []
for k in range(8):
    o0, o1 = cvoof(view(0, k).astype(np.float32)), cvoof(viewp(0, k).astype(np.float32)); a0.append(rm(o0)); a1.append(rm(o1))
    print(f"  k={k}: {a0[-1]:.3f} → {a1[-1]:.3f}", flush=True)
print(f"  평균 {np.mean(a0):.3f} → {np.mean(a1):.3f}")
shifts_have = [s for s in (0, 1, 2, 3) if __import__('os').path.exists(D + f"train_s{s}_k0{sfx}.npy")]
for nm, views in (("A 4보기", S(A_)), ("B 4보기", S(B_)), ("D4 8보기", S(ALL)), ("D4 8 × 이동2 = 16보기", S(ALL, (0, 1))), ("D4 8 × 이동4 = 32보기", S(ALL, (0, 1, 2, 3)))):
    if any(s not in shifts_have for s, _ in views): continue
    o0, o1 = cvoof(avg(views)), cvoof(avgp(views))
    rep(f"{nm} 기존", o0); rep(f"{nm} 편향 제거", o1, o0); sys.stdout.flush()
    np.save(f"analysis/pb_oof_{len(views)}{sfx}.npy", o1)
