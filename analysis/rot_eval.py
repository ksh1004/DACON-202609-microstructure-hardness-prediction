"""회전 각도별 임베딩(outputs/probe2_rot/<stem>/a*_train.npy)의 각도 집합별 평균을 multi_ridge와 같은 방식으로 평가.
묶음: hetero3 블록 0-5 (G=1) + 상 조건부 6-9 (G=0.25) + 혼합 기여량 10 (G=0.05), 층 위치 1-4, 폴드별 표준화·±6 절단, RidgeCV."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
stem = sys.argv[1] if len(sys.argv) > 1 else "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
D = f"outputs/probe2_rot/{stem}/"
NL, NB, C = 7, 11, 1024
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def load(angs, sp="train"):
    return np.mean([np.load(D + f"a{a:g}_{sp}.npy").astype(np.float64) for a in angs], 0)
def design(A, t, extra=None):
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
def cvoof(A):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k
        Z = design(A, t)
        o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]).predict(Z[v])
    return o
import os
have = sorted(float(f[1:].split("_")[0]) for f in os.listdir(D) if f.endswith("_train.npy"))
print("각도:", have)
base = cvoof(load([0])); rep("0° (기존과 동일해야 함)", base)
ref = oof(E + "dnppm0.05_ff48") if "fflowpass48" in stem else None
if ref is not None: print("   기존 dnppm0.05_ff48 대비 OOF 예측 RMS 차이 %.4f" % np.sqrt(((base - ref) ** 2).mean()))
res = {"0": base}
sets = [[45], [22.5], [0, 45], [0, 22.5, 45, 67.5], [22.5, 67.5], [45, 22.5, 67.5]]
for s in sets:
    if all(a in have for a in s):
        o = cvoof(load(s)); res["+".join(f"{a:g}" for a in s)] = o
        rep("특징 평균 " + "+".join(f"{a:g}" for a in s), o, base)
# 예측 평균(각도별 따로 적합 후 예측 평균)과 비교
if all(a in have for a in (0, 45)):
    rep("예측 평균 0 & 45", (res["0"] + res["45"]) / 2, base)
if all(k in res for k in ("0", "45", "22.5")) and 67.5 in have:
    o67 = cvoof(load([67.5])); rep("특징 평균 67.5", o67, base)
    rep("예측 평균 4각도", (res["0"] + res["45"] + res["22.5"] + o67) / 4, base)
np.savez("analysis/rot_oof_" + stem[-18:] + ".npz", **res)
# 정렬도 구간별 비교
F = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[ids]; coh = F.an_local_coherence.values
best = res.get("0+22.5+45+67.5", res.get("0+45"))
if best is not None:
    qs = np.quantile(coh, [0, .2, .4, .6, .8, 1])
    print("정렬도 5분위별 RMSE (0° → 회전 평균):", " | ".join(f"{np.sqrt(((base-y)[(coh>=qs[i])&(coh<=qs[i+1])]**2).mean()):.2f}→{np.sqrt(((best-y)[(coh>=qs[i])&(coh<=qs[i+1])]**2).mean()):.2f}" for i in range(5)))
