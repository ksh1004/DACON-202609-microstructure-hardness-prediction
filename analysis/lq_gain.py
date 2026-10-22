"""저품질 예측이 눌리는 문제를 특징 단계에서 고치기: 저품질 행의 표준화 특징을 (저품질 train 평균 또는 전체 평균 기준) g배 확대한 뒤 Ridge 학습.
사후 배율(예측만 확대)과 달리 학습에도 반영되어 공유 가중치가 덜 왜곡될 수 있음. D16 σ48, DenseNet σ48, TIPS σ48 블록. CPU."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
P2 = "outputs/probe2/"; S = "_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass48_low"
SPEC = {"D16": (P2 + "vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21" + S + "_sh2", [1, 2, 3, 4], 1024),
        "DenseNet": (P2 + "cnn_densenet121.tv_in1k_s2_512" + S, [0], 512), "TIPS": (P2 + "vit_so400m_patch14_reg1_tipsv2.webli_448_L8-10-12-14" + S + "_sh2", [0, 1, 2, 3], 1152)}
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05)); Lq = sig >= 10; c = y.mean()
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
rm = lambda p, m=None: np.sqrt(((p - y) ** 2).mean()) if m is None else np.sqrt(((p - y)[m] ** 2).mean())
for name, (path, layers, C) in SPEC.items():
    A = np.load(path + "/emb.npz")["train"].astype(np.float64); nl = A.shape[1] // (C * 11); A4 = A.reshape(len(A), nl, 11, C)[:, layers]
    print(f"\n== {name} σ48 ==")
    base = None
    for mode, g in (("기준", 1.0), ("저품질 평균 기준", 1.1), ("저품질 평균 기준", 1.2), ("저품질 평균 기준", 1.3), ("전체 평균 기준", 1.1), ("전체 평균 기준", 1.2), ("모드 평균 제거", 1.0), ("모드 평균 제거", 1.15)):
        o = np.zeros(len(y))
        for k in range(5):
            t, v = fold != k, fold == k; Z = design(A4, t)
            if mode == "저품질 평균 기준": mL = Z[t & Lq].mean(0); Z[Lq] = mL + g * (Z[Lq] - mL)
            elif mode == "전체 평균 기준": Z[Lq] = g * Z[Lq]
            elif mode == "모드 평균 제거": mL, mH = Z[t & Lq].mean(0), Z[t & ~Lq].mean(0); Z[Lq] = g * (Z[Lq] - mL); Z[~Lq] = Z[~Lq] - mH
            if mode == "모드 평균 제거": Z = np.hstack([Z, Lq[:, None] * np.sqrt(Z.shape[1]) * 0.3])
            o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]).predict(Z[v])
        if base is None: base = o
        d = (o - y) ** 2 - (base - y) ** 2
        print(f"  {mode:12s} g={g:.2f}: OOF {rm(o):.3f} (고품질 {rm(o, ~Lq):.3f}, 저품질 {rm(o, Lq):.3f}) | 기울기 고 {sl(o, ~Lq):.3f} 저 {sl(o, Lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f}", flush=True)
    q = c + np.where(Lq, 1.10, 1.0) * (base - c); d = (q - y) ** 2 - (base - y) ** 2
    print(f"  (비교) 사후 배율 저품질 ×1.10: OOF {rm(q):.3f} (저품질 {rm(q, Lq):.3f}) | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f}")
