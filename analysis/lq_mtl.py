"""저품질 배율 g에 더해 모드 전용 성분(다중 과제 Ridge: w_m = w_공유 + v_m)을 주면 더 나아지는지. 특징 증강 [Z, √ρ·Z·1(저품질), √ρ·Z·1(고품질)].
사용: python lq_mtl.py <D16|TIPS|DenseNet>"""
import sys; NAME = sys.argv[1]; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
P2 = "outputs/probe2/"; S = "_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass48_low"
SPEC = {"D16": (P2 + "vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21" + S + "_sh2", [1, 2, 3, 4], 1024),
        "TIPS": (P2 + "vit_so400m_patch14_reg1_tipsv2.webli_448_L8-10-12-14" + S + "_sh2", [0, 1, 2, 3], 1152), "DenseNet": (P2 + "cnn_densenet121.tv_in1k_s2_512" + S, [0], 512)}
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05)); Lq = sig >= 10
def design(A4, t):
    outs, ref = [], None
    for blocks, g in GR:
        B = A4[:, :, list(blocks)].reshape(len(A4), -1); mu, sd = B[t].mean(0), B[t].std(0); keep = sd > 1e-10
        B = np.clip((B[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(B * s)
    return np.hstack(outs)
path, layers, C = SPEC[NAME]; A = np.load(path + "/emb.npz")["train"].astype(np.float64); nl = A.shape[1] // (C * 11); A4 = A.reshape(len(A), nl, 11, C)[:, layers]
Zs = [design(A4, fold != k) for k in range(5)]; rm = lambda p, m=None: np.sqrt(((p - y) ** 2).mean()) if m is None else np.sqrt(((p - y)[m] ** 2).mean())
print(f"== {NAME} σ48: 배율 g × 모드 전용 성분 ρ ==")
for g in (1.0, 1.5):
    for rho in (0, 0.1, 0.3, 1.0, 3.0):
        o = np.zeros(len(y))
        for k in range(5):
            t, v = fold != k, fold == k; Z = Zs[k].copy(); Z[Lq] *= g
            if rho > 0: Z = np.hstack([Z, np.sqrt(rho) * Z * Lq[:, None], np.sqrt(rho) * Z * (~Lq)[:, None]])
            o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]).predict(Z[v])
        print(f"  g={g} ρ={rho:<4}: OOF {rm(o):.3f} (고 {rm(o, ~Lq):.3f} / 저 {rm(o, Lq):.3f})", flush=True)
