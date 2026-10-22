"""저품질 특징 확대(g) 스캔 — 계열별 σ48 블록과 3계열 블렌드(.45/.25/.30). 저품질 행의 표준화 특징에 g를 곱한 뒤 Ridge(전체 평균 기준 = 단순 곱).
폴드별 OOF를 저장: analysis/lqgain_oof.npz (키 '<계열>_g<값>')"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
P2 = "outputs/probe2/"; S = "_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass48_low"
SPEC = {"D16": (P2 + "vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21" + S + "_sh2", [1, 2, 3, 4], 1024),
        "TIPS": (P2 + "vit_so400m_patch14_reg1_tipsv2.webli_448_L8-10-12-14" + S + "_sh2", [0, 1, 2, 3], 1152), "DenseNet": (P2 + "cnn_densenet121.tv_in1k_s2_512" + S, [0], 512)}
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05)); Lq = sig >= 10; G = (1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.5)
def design(A4, t):
    outs, ref = [], None
    for blocks, g in GR:
        B = A4[:, :, list(blocks)].reshape(len(A4), -1); mu, sd = B[t].mean(0), B[t].std(0); keep = sd > 1e-10
        B = np.clip((B[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(B * s)
    return np.hstack(outs)
def sl(p, m): cv_ = np.cov(y[m], p[m]); return cv_[0, 1] / cv_[1, 1]
rm = lambda p, m=None, w=None: np.sqrt(np.average(((p - y) ** 2) if m is None else ((p - y)[m] ** 2), weights=w))
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w18 = wg(18.0)
R = {}
for name, (path, layers, C) in SPEC.items():
    A = np.load(path + "/emb.npz")["train"].astype(np.float64); nl = A.shape[1] // (C * 11); A4 = A.reshape(len(A), nl, 11, C)[:, layers]
    Zs = [design(A4, fold != k) for k in range(5)]; al = {}
    for g in G:
        o = np.zeros(len(y)); a_ = []
        for k in range(5):
            t, v = fold != k, fold == k; Z = Zs[k].copy(); Z[Lq] *= g
            r = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]); o[v] = r.predict(Z[v]); a_.append(r.alpha_)
        R[f"{name}_g{g}"] = o; al[g] = np.exp(np.mean(np.log(a_)))
    print(f"\n== {name} σ48 ==")
    for g in G:
        o, b = R[f"{name}_g{g}"], R[f"{name}_g1.0"]; d = (o - y) ** 2 - (b - y) ** 2
        print(f"  g={g:.1f}: OOF {rm(o):.3f} (고 {rm(o, ~Lq):.3f} / 저 {rm(o, Lq):.3f}) | pw13 {rm(o, w=w13):.3f} std18 {rm(o, w=w18):.3f} | 기울기 고 {sl(o, ~Lq):.3f} 저 {sl(o, Lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f} 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | α {al[g]:.0f}", flush=True)
print("\n== 3계열 σ48 블렌드 (D16 .45 + TIPS .25 + DenseNet .30) ==")
for g in G:
    o = .45 * R[f"D16_g{g}"] + .25 * R[f"TIPS_g{g}"] + .30 * R[f"DenseNet_g{g}"]; b = .45 * R["D16_g1.0"] + .25 * R["TIPS_g1.0"] + .30 * R["DenseNet_g1.0"]; d = (o - y) ** 2 - (b - y) ** 2
    print(f"  g={g:.1f}: OOF {rm(o):.3f} (고 {rm(o, ~Lq):.3f} / 저 {rm(o, Lq):.3f}) | pw13 {rm(o, w=w13):.3f} std18 {rm(o, w=w18):.3f} | 기울기 전체 {sl(o, np.ones(len(y), bool)):.3f} 고 {sl(o, ~Lq):.3f} 저 {sl(o, Lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f} 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5")
np.savez("analysis/lqgain_oof.npz", **R)
