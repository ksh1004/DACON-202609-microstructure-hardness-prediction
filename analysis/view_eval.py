"""보기 부분집합 평가: (1) 단일 보기 8개, (2) 기존 묶음 A·여집합 B·8보기, (3) 무작위 4보기 묶음의 OOF 분포(=위약 분포),
(4) 격자 이동을 더한 보기 수 4→8→16→32 의 효과, (5) 이동된 A·B 묶음(조정에 안 쓰인 A 보기)."""
import sys, time; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
stem = sys.argv[1] if len(sys.argv) > 1 else "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
D = f"outputs/probe2_views/{stem}/"
NL, NB, C = 4, 11, 1024
GR0 = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
V = {}
def view(s, k):
    if (s, k) not in V: V[(s, k)] = np.load(D + f"train_s{s}_k{k}.npy")
    return V[(s, k)]
def avg(views):
    a = np.zeros((len(y), NL * NB * C), np.float32)
    for s, k in views: a += view(s, k)
    return a / len(views)
def design(A, t, GR=GR0):
    A = A.reshape(len(A), NL, NB, C)
    outs, ref = [], None
    for blocks, g in GR:
        Z = A[:, :, list(blocks)].reshape(len(A), -1).astype(np.float64)
        mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10
        Z = np.clip((Z[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(Z * s)
    return np.hstack(outs)
def cvoof(A, GR=GR0):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k
        Z = design(A, t, GR); o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]).predict(Z[v])
    return o
rm = lambda o: np.sqrt(((o - y) ** 2).mean())
A_, B_, ALL = [0, 1, 4, 5], [2, 3, 6, 7], list(range(8))
S = lambda ks, ss=(0,): [(s, k) for s in ss for k in ks]
R = {}
def run(name, views, base=None):
    t0 = time.time(); R[name] = cvoof(avg(views)); rep(f"{name} [{len(views)}보기]", R[name], R.get(base)); sys.stdout.flush()
if __name__ == "__main__":
    run("A 기존 묶음", S(A_)); run("B 여집합", S(B_), "A 기존 묶음"); run("D4 8보기", S(ALL), "B 여집합")
    print("\n-- 단일 보기 (k: 4비트=전치, 하위 2비트=90° 회전 횟수) --")
    for k in range(8): run(f"단일 k={k}", S([k]))
    print("단일 보기 OOF 평균 %.3f" % np.mean([rm(R[f'단일 k={k}']) for k in range(8)]))
    print("\n-- 무작위 4보기 묶음 (이동 없음) --")
    rng = np.random.default_rng(0); seen = {tuple(A_), tuple(B_)}; vals = []
    while len(vals) < 12:
        ks = tuple(sorted(rng.choice(8, 4, replace=False).tolist()))
        if ks in seen: continue
        seen.add(ks); o = cvoof(avg(S(list(ks)))); vals.append(rm(o)); print(f"  {ks}: {vals[-1]:.3f}", flush=True)
    print("무작위 4보기 OOF: 평균 %.3f, 표준편차 %.3f, 최소 %.3f, 최대 %.3f | A %.3f, B %.3f" % (np.mean(vals), np.std(vals), min(vals), max(vals), rm(R["A 기존 묶음"]), rm(R["B 여집합"])))
    print("\n-- 이동된 묶음 (4보기) --")
    for s in (1, 2, 3):
        run(f"A 이동{s}", S(A_, (s,))); run(f"B 이동{s}", S(B_, (s,)))
    print("\n-- 보기 수 늘리기 --")
    run("B × 이동 2", S(B_, (0, 1)), "B 여집합"); run("B × 이동 4", S(B_, (0, 1, 2, 3)), "B 여집합")
    run("A × 이동 4", S(A_, (0, 1, 2, 3)), "B 여집합")
    run("D4 8 × 이동 2", S(ALL, (0, 1)), "D4 8보기"); run("D4 8 × 이동 4", S(ALL, (0, 1, 2, 3)), "D4 8보기")
    np.savez("analysis/view_oof.npz", **{k.replace(" ", "_"): v for k, v in R.items()})
