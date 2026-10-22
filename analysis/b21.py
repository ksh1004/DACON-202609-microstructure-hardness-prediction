"""성분 6개(C 미세조정 CNN, P 측정 블록, A DINOv2 4보기, D DINOv2 16보기, T TIPSv2, F 고정 CNN)의 임의 가중치 조합에 대한 Public 예상.
채점된 제출 6개(v15a, v16_noP, v18a, v19a, v19b, v19c)가 가중치 공간을 생성 → 블렌드 항등식
  MSE(Σλ_i p_i) = Σλ_i M_i − ½ ΣΣ λ_i λ_j D_ij   (Σλ=1, D_ij = Public에서의 mean (p_i−p_j)² ≈ 0.93 × test 전체)
사용: python b21.py <v19a 점수> <v19b 점수> <v19c 점수> [--save 이름 wC wP wA wD wT wF]"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
args = [a for a in sys.argv[1:] if not a.startswith("--")]
co, ct = grp(CNN); po, pt = oof(E + "mr_v14_pp_sc"), tp(E + "mr_v14_pp_sc")
ao, at = grp([E + f"dnpp0.25_ff{s}" for s in FF]); do, dt = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF]); to, tt = grp([E + f"tips16_dnppm0.05_ff{s}" for s in FF])
fo, ft = grp([E + f"cnnf_{n}_ff{s}" for n in ("dn121", "r50a1", "r50a2") for s in FF])
CO, CT = np.column_stack([co, po, ao, do, to, fo]), np.column_stack([ct, pt, at, dt, tt, ft]); names = ["C", "P", "A", "D", "T", "F"]
dm = lambda r, a, b, c: [r * a * .275 / .70, r * a * .425 / .70, r * b, r * c]     # 특징 몫 r을 (DINOv2 혼합 a, TIPS b, 고정CNN c)로
SUB = {"v15a": ([.15, .30, .55, 0, 0, 0], 10.2913472134), "v16_noP": ([.15, 0, 0, .85, 0, 0], 10.3171457020), "v18a": ([.15, .15] + dm(.70, .7, .3, 0), 10.1232275920),
       "v19a": ([.15, .15] + dm(.70, .45, .25, .30), None), "v19b": ([0, .15] + dm(.85, .45, .25, .30), None), "v19c": ([0, 0] + dm(1.0, .45, .25, .30), None)}
for k, v in zip(("v19a", "v19b", "v19c"), args[:3]): SUB[k] = (SUB[k][0], float(v))
files = {"v15a": "sub_v15a_dinopp.csv", "v16_noP": "sub_v16_noP.csv", "v18a": "sub_v18a_tips.csv", "v19a": "sub_v19a_cnnf.csv", "v19b": "sub_v19b_cnnf_noC.csv", "v19c": "sub_v19c_cnnf_noC_noP.csv"}
for k, (w, s) in SUB.items():
    f = pd.read_csv(f"submissions/{files[k]}").set_index("ID").loc[te, "hardness"].values
    assert np.sqrt(((CT @ np.array(w) - f) ** 2).mean()) < 1e-6, k          # 성분 분해가 제출 파일과 일치하는지 검산
if any(s is None for _, s in SUB.values()):
    print("사용: python b21.py <v19a> <v19b> <v19c>  — 아직 점수가 없는 제출이 있습니다."); sys.exit()
keys = list(SUB); Wm = np.array([SUB[k][0] for k in keys]).T; Ms = np.array([SUB[k][1] ** 2 for k in keys])
print("가중치 행렬 계수(6이어야 전체 공간 생성):", np.linalg.matrix_rank(Wm), "| 조건수 %.0f" % np.linalg.cond(Wm))
Tm = CT @ Wm; Dij = ((Tm[:, :, None] - Tm[:, None, :]) ** 2).mean(0) * 0.931
def pub(w):
    lam = np.linalg.solve(Wm, np.asarray(w, float)); return np.sqrt(max(lam @ Ms - 0.5 * lam @ Dij @ lam, 0)), lam
oofr = lambda w: np.sqrt(((CO @ np.asarray(w) - y) ** 2).mean())
pw = lambda w: np.sqrt(np.average((CO @ np.asarray(w) - y) ** 2, weights=w13))
print("\n검산(채점 제출의 재계산):", {k: round(pub(SUB[k][0])[0], 4) for k in keys})
print("성분별 단독 Public 외삽(λ 절대값 합이 크면 신뢰 낮음):")
for i, n in enumerate(names):
    v, lam = pub(np.eye(6)[i]); print(f"  {n}: {v:.3f}  (|λ| 합 {np.abs(lam).sum():.1f})")
# 격자 탐색: C, P 비중 × 특징 몫 배분
rows = []
for wc in (0, .05, .1, .15, .2):
    for wp in (0, .05, .1, .15, .2, .3):
        r = 1 - wc - wp
        for a in (.3, .4, .45, .5, .6):
            for b in (.15, .25, .35):
                c = 1 - a - b
                if c < 0.1: continue
                for split in (.275 / .70, .5, .2):          # DINOv2 혼합 안에서 4보기 몫
                    w = [wc, wp, r * a * split, r * a * (1 - split), r * b, r * c]; v, lam = pub(w)
                    rows.append((v, np.abs(lam).sum(), oofr(w), pw(w), wc, wp, a, b, c, split))
R = pd.DataFrame(rows, columns=["public", "lam_abs", "oof", "pw13", "C", "P", "dino", "tips", "cnnf", "A_in_dino"])
print("\nPublic 예상 상위 12 (외삽이 과하지 않은 것만: |λ| 합 ≤ 3):")
print(R[R.lam_abs <= 3].sort_values("public").head(12).round(3).to_string(index=False))
print("\nC·P 비중별 최선 Public (다른 배분은 최적):")
print(R[R.lam_abs <= 3].groupby(["C", "P"]).public.min().unstack().round(3).to_string())
if "--save" in sys.argv:
    i = sys.argv.index("--save"); name = sys.argv[i + 1]; w = [float(x) for x in sys.argv[i + 2:i + 8]]; assert abs(sum(w) - 1) < 1e-6
    pd.DataFrame({"ID": te, "hardness": CT @ np.array(w)}).to_csv(f"submissions/{name}", index=False)
    print(f"\n→ submissions/{name}: {dict(zip(names, np.round(w, 4)))} | Public 예상 {pub(w)[0]:.4f}, OOF {oofr(w):.3f}, pw13 {pw(w):.3f}")
