"""저품질 배율 g를 '예측 대상 모드'별로 달리 쓰기: 고품질 이미지는 g_H 모델, 저품질 이미지는 g_L 모델의 예측을 사용.
(g가 커지면 저품질 예측은 좋아지고 고품질 예측은 조금 나빠짐 → 모드별로 다른 g.)  g<1 추가 스캔, 중첩 선택으로 정직한 이득 추정,
그리고 블록별 분산비(비지도)·노이즈 구간별 배율 점검. σ48 블록 3계열. CPU."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from sklearn.linear_model import RidgeCV
P2 = "outputs/probe2/"; S = "_Pphase2m_T8_Sunet_cm7_am22_l1_i10000_boot_fflowpass48_low"
SPEC = {"D16": (P2 + "vit_large_patch14_dinov2.lvd142m_448_L9-11-13-15-17-19-21" + S + "_sh2", [1, 2, 3, 4], 1024),
        "TIPS": (P2 + "vit_so400m_patch14_reg1_tipsv2.webli_448_L8-10-12-14" + S + "_sh2", [0, 1, 2, 3], 1152), "DenseNet": (P2 + "cnn_densenet121.tv_in1k_s2_512" + S, [0], 512)}
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05)); Lq = sig >= 10; NEW = (0.5, 0.7, 0.85, 1.1)
R = dict(np.load("analysis/lqgain_oof.npz"))
def design(A4, t):
    outs, ref = [], None
    for blocks, g in GR:
        B = A4[:, :, list(blocks)].reshape(len(A4), -1); mu, sd = B[t].mean(0), B[t].std(0); keep = sd > 1e-10
        B = np.clip((B[:, keep] - mu[keep]) / sd[keep], -6, 6)
        if ref is None: ref, s = keep.sum(), 1.0
        else: s = np.sqrt(g * ref / keep.sum())
        outs.append(B * s)
    return np.hstack(outs)
for name, (path, layers, C) in SPEC.items():
    A = np.load(path + "/emb.npz")["train"].astype(np.float64); nl = A.shape[1] // (C * 11); A4 = A.reshape(len(A), nl, 11, C)[:, layers]
    if name == "D16":   # 블록별(층 평균) 고품질/저품질 편차 크기 비 — 라벨을 쓰지 않는 비지도 추정
        Z = (A4 - A4.mean(0)) / (A4.std(0) + 1e-12); rH = (Z[~Lq] - Z[~Lq].mean(0)).std(0); rL = (Z[Lq] - Z[Lq].mean(0)).std(0)
        print("D16 블록별 고품질/저품질 표준편차 비(채널 중앙값): " + " ".join(f"b{b}:{np.median(rH[:, b] / rL[:, b]):.2f}" for b in range(11)))
    if all(f"{name}_g{g}" in R for g in NEW): continue
    Zs = [design(A4, fold != k) for k in range(5)]
    for g in NEW:
        o = np.zeros(len(y))
        for k in range(5):
            t, v = fold != k, fold == k; Z = Zs[k].copy(); Z[Lq] *= g; o[v] = RidgeCV(alphas=np.logspace(-1, 7, 60)).fit(Z[t], y[t]).predict(Z[v])
        R[f"{name}_g{g}"] = o
    np.savez("analysis/lqgain_oof.npz", **R); print(f"{name} 추가 스캔 완료", flush=True)
G = (0.5, 0.7, 0.85, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0)
B3 = {g: .45 * R[f"D16_g{g}"] + .25 * R[f"TIPS_g{g}"] + .30 * R[f"DenseNet_g{g}"] for g in G}
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w18, w166 = wg(18.0), wg(16.6)
rm = lambda p, m=None, w=None: np.sqrt(np.average(((p - y) ** 2) if m is None else ((p - y)[m] ** 2), weights=w if m is None or w is None else w[m]))
print("\n3계열 σ48 블렌드 — g별 모드 오차 (고품질 / 저품질), 재가중 지표별:")
for g in G: print(f"  g={g:<4}: OOF 고 {rm(B3[g], ~Lq):.3f} 저 {rm(B3[g], Lq):.3f} | pw13 고 {rm(B3[g], ~Lq, w13):.3f} 저 {rm(B3[g], Lq, w13):.3f} | std16.6 고 {rm(B3[g], ~Lq, w166):.3f} 저 {rm(B3[g], Lq, w166):.3f} | std18 고 {rm(B3[g], ~Lq, w18):.3f} 저 {rm(B3[g], Lq, w18):.3f}")
base = B3[1.0]; mix = lambda gh, gl: np.where(Lq, B3[gl], B3[gh])
print("\n모드별 g 조합 (고품질←g_H 모델, 저품질←g_L 모델):")
for gh, gl in ((1.0, 1.0), (1.3, 1.3), (1.4, 1.4), (1.0, 1.3), (1.0, 1.4), (1.0, 1.5), (1.0, 1.6), (1.0, 1.8), (0.85, 1.5), (0.7, 1.5), (0.7, 1.6), (1.1, 1.5)):
    o = mix(gh, gl); d = (o - y) ** 2 - (base - y) ** 2
    print(f"  g_H={gh:<4} g_L={gl:<4}: OOF {rm(o):.3f} | pw13 {rm(o, w=w13):.3f} std16.6 {rm(o, w=w166):.3f} std18 {rm(o, w=w18):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f} 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5")
# 중첩 선택: 바깥 폴드를 뺀 나머지 OOF에서 모드별 최선 g를 고르고 바깥 폴드에 적용
o = np.zeros(len(y)); pick = []
for k in range(5):
    t, v = fold != k, fold == k
    gh = min(G, key=lambda g: ((B3[g] - y)[t & ~Lq] ** 2).mean()); gl = min(G, key=lambda g: ((B3[g] - y)[t & Lq] ** 2).mean()); pick.append((gh, gl)); o[v] = np.where(Lq[v], B3[gl][v], B3[gh][v])
d = (o - y) ** 2 - (base - y) ** 2
print(f"\n중첩 선택(폴드별 고른 g: {pick}): OOF {rm(o):.3f} | pw13 {rm(o, w=w13):.3f} std18 {rm(o, w=w18):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f} 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5")
o1 = np.zeros(len(y)); pk1 = []
for k in range(5):
    t, v = fold != k, fold == k; g1 = min(G, key=lambda g: ((B3[g] - y)[t] ** 2).mean()); pk1.append(g1); o1[v] = B3[g1][v]
d = (o1 - y) ** 2 - (base - y) ** 2
print(f"중첩 선택(g 하나, {pk1}): OOF {rm(o1):.3f} | pw13 {rm(o1, w=w13):.3f} std18 {rm(o1, w=w18):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(len(d)):.2f}")
# 노이즈 구간별: 저품질을 σ10~14 / 14+ 로 나눠 각 구간의 g별 오차
for nm, m in (("σ10~14", (sig >= 10) & (sig < 14)), ("σ14+", sig >= 14)):
    print(f"  {nm} ({m.sum()}장) g별 오차: " + " ".join(f"{g}:{rm(B3[g], m):.2f}" for g in G[3:]))
