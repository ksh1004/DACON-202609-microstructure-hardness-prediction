"""디노이저 변형의 정직한 비교 (16보기 = D4 8 × 이동 2, σ48 저품질 조명보정, v15b 블록 구성)."""
import sys, os; sys.path.insert(0, "analysis")
from view_eval import design, cvoof, rep, y, sig, fold, np, NL, NB, C
P = "outputs/probe2_views/"
def L16(stem):
    d = P + stem + "_fflowpass48_low/"
    fs = [d + f"train_s{s}_k{k}.npy" for s in (0, 1) for k in range(8)]
    if not all(os.path.exists(f) for f in fs): return None
    return np.mean([np.load(f).astype(np.float32) for f in fs], 0)
names = {"unet_cm7_am22_l1_i10000_boot": "기준 (부트스트랩)", "unet_boot2_i10000": "2차 부트스트랩", "unet_cm7_am22_l1_i10000_boot_c256": "크롭 256 학습",
         "unet_cm7_am22_l1_i10000_boot_nb": "비블라인드(σ 채널)", "unet_naf_boot_i10000": "NAFNet", "unet_kb_boot_i10000": "블러 보존", "unet_real_boot_i10000": "블러·대비 복원(현실 열화)"}
X, O = {}, {}
for st, nm in names.items():
    a = L16(st)
    if a is None: print("(없음)", nm); continue
    X[st] = a; O[st] = cvoof(a); rep(nm, O[st], O.get("unet_cm7_am22_l1_i10000_boot")); sys.stdout.flush()
b = O["unet_cm7_am22_l1_i10000_boot"]
same = [s for s in ("unet_cm7_am22_l1_i10000_boot", "unet_boot2_i10000", "unet_cm7_am22_l1_i10000_boot_c256", "unet_cm7_am22_l1_i10000_boot_nb") if s in X]
if len(same) > 1:
    rep(f"특징 평균: 같은 계열 {len(same)}종", cvoof(np.mean([X[s] for s in same], 0)), b); rep(f"예측 평균: 같은 계열 {len(same)}종", np.mean([O[s] for s in same], 0), b)
allv = list(X)
rep(f"예측 평균: 전체 {len(allv)}종", np.mean([O[s] for s in allv], 0), b)
if "unet_real_boot_i10000" in X:
    lq = sig >= 10
    mix = X["unet_cm7_am22_l1_i10000_boot"].copy(); mix[lq] = X["unet_real_boot_i10000"][lq]
    rep("고품질=기준, 저품질=블러 복원 (특징 교체)", cvoof(mix), b)
    rep("예측 평균: 기준 + 블러 복원", (b + O["unet_real_boot_i10000"]) / 2, b)
    print("잔차상관(기준 vs 블러 복원): 전체 %.3f, 저품질 %.3f" % (np.corrcoef(b - y, O["unet_real_boot_i10000"] - y)[0, 1], np.corrcoef((b - y)[lq], (O["unet_real_boot_i10000"] - y)[lq])[0, 1]))
np.savez("analysis/dn_oof.npz", **{k: v for k, v in O.items()})
