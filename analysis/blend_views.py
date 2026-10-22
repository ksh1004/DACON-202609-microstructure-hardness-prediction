"""블렌드 수준의 보기 잡음 효과: 무작위 4보기 묶음으로 D(σ48 dnppm)·P(emb+측정 csv)를 만들어 0.15C+0.55D+0.30P 의 OOF 분포를 구하고
16·32보기와 비교. (D 그룹은 σ48 블록 하나로 근사 — 4σ 평균의 보기 잡음은 고품질에서 동일하므로 근사가 타당)"""
import sys, re; sys.path.insert(0, "analysis")
from view_eval import *
S48 = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"
def csv_group(pre, rx):
    A = pd.read_csv(f"{pre}_train.csv").set_index("ID").loc[ids]
    cols = [c for c in A.columns if re.match(rx, c)]
    return A[cols].replace([np.inf, -np.inf], np.nan).values.astype(np.float64)
CSV = [(csv_group(f"outputs/features_cache/phase_{S48}", "pf_"), 0.381), (csv_group(f"outputs/features_cache/{S48}", "ph_"), 0.019),
       (csv_group(f"outputs/features_cache/{S48}", "(pore|gr|sp)_"), 0.15), (csv_group(f"outputs/features_cache/smcdf3_{S48}", "sc"), 0.2)]
def designP(A, t):
    A4 = A.reshape(len(A), NL, NB, C); outs, ref = [], None
    groups = [(A4[:, :, list(range(6))].reshape(len(A), -1).astype(np.float64), 1.0), (A4[:, :, [6, 7, 8, 9]].reshape(len(A), -1).astype(np.float64), 0.25)] + CSV
    for Z, g in groups:
        med = np.nanmedian(Z[t], 0); Z = np.where(np.isnan(Z), med, Z)
        mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10
        Z = np.clip((Z[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(Z * s)
    return np.hstack(outs)
def cvP(A):
    o = np.zeros(len(y))
    for k in range(5):
        t, v = fold != k, fold == k
        Z = designP(A, t); o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]).predict(Z[v])
    return o
co, _ = grp(CNN)
def blend(views):
    X = avg(views); d, p = cvoof(X), cvP(X); return d, p, .15 * co + .55 * d + .3 * p
rmse = lambda o, m=None: np.sqrt((((o - y) if m is None else (o - y)[m]) ** 2).mean())
hq = sig < 10
print(f"{'보기 묶음':28s} {'D':>7s} {'P':>7s} {'블렌드':>7s} {'고품질':>7s} {'저품질':>7s}")
def show(nm, views):
    d, p, b = blend(views); print(f"{nm:28s} {rmse(d):7.3f} {rmse(p):7.3f} {rmse(b):7.3f} {rmse(b, hq):7.3f} {rmse(b, ~hq):7.3f}", flush=True); return rmse(d), rmse(p), rmse(b), b
rA = show("A (기존 4보기)", S(A_))
rng = np.random.default_rng(1); R = []; seen = {tuple(A_)}
while len(R) < 10:
    ks = tuple(sorted(rng.choice(8, 4, replace=False).tolist()))
    if ks in seen: continue
    seen.add(ks); R.append(show(f"무작위 4보기 {ks}", S(list(ks))))
for s in (1, 2, 3):
    R.append(show(f"A 이동{s} (4보기)", S(A_, (s,))))
R = np.array([r[:3] for r in R])
print(f"{'무작위·이동 4보기 평균 ± 표준편차':28s} {R[:,0].mean():7.3f} {R[:,1].mean():7.3f} {R[:,2].mean():7.3f}   (± {R[:,0].std():.3f} / {R[:,1].std():.3f} / {R[:,2].std():.3f})")
r16 = show("16보기 (D4 × 이동 0,1)", S(ALL, (0, 1))); r16b = show("16보기 (D4 × 이동 2,3)", S(ALL, (2, 3))); r32 = show("32보기", S(ALL, (0, 1, 2, 3)))
print("\n블렌드 오차² : 4보기 평균 %.2f → 16보기 %.2f / %.2f, 32보기 %.2f  (A 묶음 %.2f)" % ((R[:, 2] ** 2).mean(), r16[2] ** 2, r16b[2] ** 2, r32[2] ** 2, rA[2] ** 2))
print("같은 모델군의 OOF 예측 RMS 차: A 블렌드 vs 16보기 블렌드 %.3f" % np.sqrt(((rA[3] - r16[3]) ** 2).mean()))
