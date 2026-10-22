"""고정 CNN 계열: 모델별 Ridge 예측을 평균하는 대신 여러 CNN의 특징을 한 Ridge(커널 합)로 결합하면 얼마나 좋아지는가.
스캔 때 저장한 train 특징(σ48, D4 8보기, outputs/probe2_views/.../cnnscan_*_512_train.npz)의 stride-8 단계를 사용.
사용: python cnn_joint.py"""
import sys, glob, os; sys.path.insert(0, "analysis")
from blend_eval import *
import joint_kernel as JK
from joint_kernel import krr, GR, NB
D = "outputs/probe2_views/unet_cm7_am22_l1_i10000_boot_fflowpass48_low/"
rm = lambda o: np.sqrt(((o - y) ** 2).mean())
def kern(A):
    C = A.shape[1] // NB; A = A.astype(np.float64).reshape(len(A), 1, NB, C); Ks = []
    for k in range(5):
        t = fold != k; K = np.zeros((len(y), len(y)))
        for blocks, g in GR:
            Z = A[:, :, list(blocks)].reshape(len(A), -1); mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10
            Z = np.clip((Z[:, keep] - mu[keep]) / sd[keep], -6, 6); K += g * (Z @ Z.T) / keep.sum()
        Ks.append(K)
    return np.array(Ks)
cache = "analysis/cj_cache.npz"; KS = dict(np.load(cache)) if os.path.exists(cache) else {}
for f in sorted(glob.glob(D + "cnnscan_*_512_train.npz")):
    nm = os.path.basename(f)[8:-14]
    if "adabn" in nm or "rand" in nm or nm in KS: continue
    Z = np.load(f, allow_pickle=True); info = [s.split("|") for s in Z["info"]]; j8 = [j for j, i in enumerate(info) if i[2] == "8"]
    if not j8: continue
    KS[nm] = kern(Z[f"f{j8[-1]}"]); print("커널", nm, info[j8[-1]], flush=True)
np.savez(cache, **KS)
J = dict(np.load("analysis/jk_cache.npz")); oD, oT = krr(J["D16"]), krr(J["T16"])
solo = {n: krr(K) for n, K in KS.items()}; order = sorted(solo, key=lambda n: rm(solo[n]))
print("단독(stride 8):", ", ".join(f"{n} {rm(solo[n]):.3f}" for n in order))
bl = lambda oF: .45 * oD + .25 * oT + .30 * oF
base3 = ("densenet121-tv_in1k", "resnet50-a1_in1k", "resnet50-a2_in1k")
avg3 = np.mean([solo[n] for n in base3], 0); j3 = krr(sum(KS[n] for n in base3) / 3)
print(f"현재 3개: 예측 평균 {rm(avg3):.3f} → 블렌드 {rm(bl(avg3)):.3f} | 결합 커널 {rm(j3):.3f} → 블렌드 {rm(bl(j3)):.3f}")
def show(nm, names):
    a = np.mean([solo[n] for n in names], 0); j = krr(sum(KS[n] for n in names) / len(names)); d = (bl(j) - y) ** 2 - (bl(avg3) - y) ** 2
    print(f"  {nm:28s} [{len(names):2d}] 예측 평균 {rm(a):.3f} / 결합 {rm(j):.3f} | 블렌드: 평균 {rm(bl(a)):.3f} / 결합 {rm(bl(j)):.3f} (현재 대비 Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f}, 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5) | D와 잔차상관 {np.corrcoef(j - y, oD - y)[0, 1]:.3f}")
    return j
print("— 단독 순위 상위 k개 (순위는 OOF로 정했으므로 약간 낙관) —")
for k in (2, 3, 4, 5, 6, 8, 10, 12, 16, len(order)): show(f"상위 {k}", order[:k])
print("— 사전 정의 묶음 (선택 없음) —")
show("DenseNet 계열 전부", [n for n in order if n.startswith("densenet")]); show("ResNet-50 변형 전부", [n for n in order if n.startswith("resnet50-") or n.startswith("resnet50d")])
show("DenseNet + ResNet-50 변형", [n for n in order if n.startswith("densenet") or n.startswith("resnet50")]); jall = show("전부", order)
print("— 폴드 밖 선택: 폴드마다 나머지 4폴드의 내부 LOO 대신 단독 OOF(해당 폴드 제외)로 상위 k를 뽑아 결합 —")
for k in (4, 6, 8):
    o = np.zeros(len(y)); picks = []
    for f_ in range(5):
        v = fold == f_; top = sorted(solo, key=lambda n: ((solo[n] - y)[~v] ** 2).mean())[:k]; picks.append(top)
        o[v] = krr(sum(KS[n] for n in top) / k)[v]
    d = (bl(o) - y) ** 2 - (bl(avg3) - y) ** 2
    print(f"  상위 {k} (폴드 밖 선택): 결합 {rm(o):.3f} | 블렌드 {rm(bl(o)):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f}, 폴드 {sum(d[fold == q].mean() < 0 for q in range(5))}/5) | 폴드별 선택 일치 {len(set.intersection(*map(set, picks)))}/{k}")
