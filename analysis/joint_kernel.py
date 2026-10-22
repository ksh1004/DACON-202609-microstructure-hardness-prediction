"""계열별 Ridge 예측을 평균하는 것(현재 방식) vs 계열 커널을 더해 한 번에 적합하는 것(결합 커널 Ridge)의 비교. σ48, train 500장.
계열 커널 K_m = Σ_묶음 G_b · Z_b Z_bᵀ / (묶음 특징 수)  (묶음: hetero3 1.0, 상 조건부 0.25, 기여량 0.05 — multi_ridge와 같은 상대 분산)
폴드별 커널을 analysis/jk_cache.npz에 저장해 재사용한다. 사용: python joint_kernel.py"""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
S = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low"; P = "outputs/probe2/"
FAM = {"D16": (P + f"vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21_Pphase2m_T8_S{S}_sh2", [1, 2, 3, 4], 1024),
       "T16": (P + f"vit_so400m_patch14_reg1_tipsv2.webli_448_L8-10-12-14_Pphase2m_T8_S{S}_sh2", [0, 1, 2, 3], 1152),
       "dn121": (P + f"cnn_densenet121.tv_in1k_s2_512_Pphase2m_T8_S{S}", [0], 512),
       "r50a1": (P + f"cnn_resnet50.a1_in1k_s2_512_Pphase2m_T8_S{S}", [0], 512),
       "r50a2": (P + f"cnn_resnet50.a2_in1k_s2_512_Pphase2m_T8_S{S}", [0], 512)}
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05)); NB = 11
LAM = np.logspace(-6, 3, 60)
def fam_kernels(path, layers, C):
    A = np.load(path + "/emb.npz")["train"].astype(np.float64); nl = A.shape[1] // (C * NB); A = A.reshape(len(A), nl, NB, C)[:, layers]
    Ks = []
    for k in range(5):
        t = fold != k; K = np.zeros((len(y), len(y)))
        for blocks, g in GR:
            Z = A[:, :, list(blocks)].reshape(len(A), -1); mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10
            Z = np.clip((Z[:, keep] - mu[keep]) / sd[keep], -6, 6); K += g * (Z @ Z.T) / keep.sum()
        Ks.append(K)
    return np.array(Ks)
def krr(Ks):
    """폴드별 커널(5,n,n) → OOF. λ는 train 내부 LOO로 선택(RidgeCV와 같은 기준)."""
    o = np.zeros(len(y)); lams = []
    for k in range(5):
        t, v = fold != k, fold == k; K = Ks[k].astype(np.float64); yc = y[t] - y[t].mean()
        cm = K[:, t].mean(1, keepdims=True); K = K - cm - cm.T + K[np.ix_(t, t)].mean()        # train 평균으로 특징 중심화(RidgeCV의 절편과 같음)
        ev, Q = np.linalg.eigh(K[np.ix_(t, t)]); ev = np.clip(ev, 0, None); Qy = Q.T @ yc; best = None
        for lam in LAM * np.trace(K[np.ix_(t, t)]) / t.sum():
            a = Q @ (Qy / (ev + lam)); d = (Q ** 2) @ (1 / (ev + lam)) - 1 / (lam * t.sum()); e = ((a / d) ** 2).mean()   # 절편의 지렛대 보정
            if best is None or e < best[0]: best = (e, a, lam)
        o[v] = K[np.ix_(v, t)] @ best[1] + y[t].mean(); lams.append(best[2])
    return o
rm = lambda o: np.sqrt(((o - y) ** 2).mean())
if __name__ == "__main__":
    import os
    cache = "analysis/jk_cache.npz"
    KS = dict(np.load(cache)) if os.path.exists(cache) else {}
    for n, (p, l, c) in FAM.items():
        if n not in KS: KS[n] = fam_kernels(p, l, c); print("커널", n, flush=True)
    np.savez(cache, **KS)
    O = {n: krr(KS[n]) for n in FAM}
    print("계열 단독 OOF:", {n: round(rm(o), 3) for n, o in O.items()}, "| 저장된 Ridge:", {n: round(rm(oof(E + d)), 3) for n, d in (("D16", "mv16_dnppm0.05_ff48"), ("T16", "tips16_dnppm0.05_ff48"), ("dn121", "cnnf_dn121_ff48"))})
    F3 = ("dn121", "r50a1", "r50a2")
    for nm, w in (("D .45 T .25 F .30", dict(D16=.45, T16=.25, dn121=.10, r50a1=.10, r50a2=.10)), ("D .40 T .30 F .30", dict(D16=.40, T16=.30, dn121=.10, r50a1=.10, r50a2=.10)),
                  ("D T F 동일", dict(D16=1 / 3, T16=1 / 3, dn121=1 / 9, r50a1=1 / 9, r50a2=1 / 9)), ("D .6 T .4", dict(D16=.6, T16=.4)), ("F 셋만", dict(dn121=1 / 3, r50a1=1 / 3, r50a2=1 / 3))):
        avg = sum(v * O[n] for n, v in w.items()); jo = krr(sum(v * KS[n] for n, v in w.items()))
        d = (jo - y) ** 2 - (avg - y) ** 2
        print(f"  {nm:20s} 예측 평균 {rm(avg):.3f} | 결합 커널 {rm(jo):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f} 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | 둘의 평균 {rm((avg + jo) / 2):.3f} | 잔차상관 {np.corrcoef(avg - y, jo - y)[0, 1]:.3f}")
