"""고정 CNN 위원회: 스캔한 모든 CNN의 stride-8 단계(규칙으로 고정) 모델을 평균. 선택 편향 없이 구성원 그룹의 가치를 평가.
단계별 OOF를 다시 계산해 저장(analysis/cnn_stage_oof.npz) → 위원회·부분집합·한 개 빼기 안정성, DINOv2-L·TIPSv2와의 블렌드."""
import sys, glob, os; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def design(A, t):
    C = A.shape[1] // 11; A3 = A.reshape(len(A), 11, C); outs, ref = [], None
    for blocks, g in GR:
        B = A3[:, list(blocks)].reshape(len(A), -1).astype(np.float64); mu, sd = B[t].mean(0), B[t].std(0); keep = sd > 1e-10
        B = np.clip((B[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(B * s)
    D_ = np.hstack(outs); return D_ / np.sqrt(D_.shape[1])
def cv(Zs):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k; D_ = np.hstack([design(Z, t) for Z in Zs])
        o[v] = RidgeCV(alphas=np.logspace(-6, 3, 60)).fit(D_[t], y[t]).predict(D_[v])
    return o
cache = "analysis/cnn_stage_oof.npz"; O = dict(np.load(cache)) if os.path.exists(cache) else {}
for f in sorted(glob.glob(f"outputs/probe2_views/{S48}/cnnscan_*_512_train.npz")):
    tag = os.path.basename(f)[8:-10]; Z = np.load(f, allow_pickle=True); info = [s.split("|") for s in Z["info"]]
    for j, (nm, c, r) in enumerate(info):
        key = f"{tag}|/{r}"
        if r in ("4", "8") and key not in O:
            O[key] = cv([Z[f"f{j}"]]); print(f"  {key:52s} {np.sqrt(((O[key] - y) ** 2).mean()):.3f}", flush=True)
np.savez(cache, **O)
L = np.load("analysis/view_oof.npz")["D4_8보기"]; T = np.load("analysis/model_oof_mso400m_patch14_reg1_tipsv2-webli.npy")
rm = lambda o: np.sqrt(((o - y) ** 2).mean())
s8 = {k: v for k, v in O.items() if k.endswith("|/8")}; s4 = {k: v for k, v in O.items() if k.endswith("|/4")}
print(f"\nstride 8 모델 {len(s8)}개: 단독 평균 {np.mean([rm(v) for v in s8.values()]):.3f} (최소 {min(rm(v) for v in s8.values()):.3f}, 최대 {max(rm(v) for v in s8.values()):.3f})")
print(f"stride 4 모델 {len(s4)}개: 단독 평균 {np.mean([rm(v) for v in s4.values()]):.3f}")
C8 = np.mean(list(s8.values()), 0); C4 = np.mean(list(s4.values()), 0); C48 = (C8 + C4) / 2
rep("DINOv2-L 8보기", L); rep("위원회 C8 (stride 8 전부 평균)", C8, L); rep("위원회 C4 (stride 4 전부 평균)", C4, L); rep("위원회 C48 (둘의 평균)", C48, L)
c = lambda a, b: np.corrcoef(a - y, b - y)[0, 1]
print("잔차상관: L–C8 %.3f, L–C4 %.3f, C8–C4 %.3f, T–C8 %.3f" % (c(L, C8), c(L, C4), c(C8, C4), c(T, C8)))
for nm, Cm in (("C8", C8), ("C4", C4), ("C48", C48)):
    for w in (0.2, 0.3, 0.4): rep(f"L {1-w:.1f} + {nm} {w:.1f}", (1 - w) * L + w * Cm, L)
print("\n-- DINOv2-L + TIPSv2 + 위원회 --")
rep("L .7 + T .3", .7 * L + .3 * T, L)
for w in ((.45, .25, .30), (.4, .2, .4), (.4, .25, .35), (.35, .25, .40), (.5, .2, .3)):
    rep("L %.2f + T %.2f + C8 %.2f" % w, w[0] * L + w[1] * T + w[2] * C8, L); rep("L %.2f + T %.2f + C48 %.2f" % w, w[0] * L + w[1] * T + w[2] * C48, L)
# 안정성: 무작위 절반 위원회, 한 개 빼기
rng = np.random.default_rng(0); ks = list(s8); vals = []
for _ in range(20):
    sub = rng.choice(len(ks), len(ks) // 2, replace=False); Cs = np.mean([s8[ks[i]] for i in sub], 0); vals.append(rm(.45 * L + .25 * T + .30 * Cs))
print("무작위 절반(%d개) 위원회로 L.45+T.25+C.30: 평균 %.3f, 범위 %.3f~%.3f" % (len(ks) // 2, np.mean(vals), min(vals), max(vals)))
vals = {k: rm(.45 * L + .25 * T + .30 * np.mean([s8[j] for j in ks if j != k], 0)) for k in ks}
print("한 개 빼기: 범위 %.3f~%.3f (가장 영향 큰 모델: %s)" % (min(vals.values()), max(vals.values()), max(vals, key=vals.get)))
for n in (3, 5, 8):
    vv = [rm(.45 * L + .25 * T + .30 * np.mean([s8[ks[i]] for i in rng.choice(len(ks), n, replace=False)], 0)) for _ in range(30)]
    print(f"무작위 {n}개 위원회: 평균 {np.mean(vv):.3f}, 범위 {min(vv):.3f}~{max(vv):.3f}")
np.save("analysis/cnn_C8_oof.npy", C8)
