"""약한 정규화(고정 λ)에서 추가 구성원이 도움이 되는지 재평가 — LOO λ에서는 ViT 추가가 모두 악화였다. σ48, 8보기 스캔 특징, 3계열 기준(D .45 T .25 F .30)."""
import sys, os, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()): import lam_sweep as L
from lam_sweep import *
LAMF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
Dv = "outputs/probe2_views/unet_cm7_am22_l1_i10000_boot_fflowpass48_low/"; GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05))
def kern_views(tag, C, layers):
    X = np.mean([np.load(Dv + f"train_s0_k{k}{tag}.npy").astype(np.float32) for k in range(8)], 0); NL = X.shape[1] // (11 * C); A = X.reshape(len(X), NL, 11, C)[:, layers].astype(np.float64); Ks = []
    for k in range(5):
        t = fold != k; K = np.zeros((len(y), len(y)))
        for blocks, g in GR:
            Z = A[:, :, list(blocks)].reshape(len(A), -1); mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10; Z = np.clip((Z[:, keep] - mu[keep]) / sd[keep], -6, 6); K += g * (Z @ Z.T) / keep.sum()
        Ks.append(K)
    return np.array(Ks)
cache = "analysis/ml_cache.npz"; KX = dict(np.load(cache)) if os.path.exists(cache) else {}
SPEC = {"reg4(7~13)": ("_mlarge_patch14_reg4_dinov2-lvd142m", 1024, [0, 1, 2, 3]), "DINOv2-g(16~25)": ("_mgiant_patch14_dinov2-lvd142m", 1536, [0, 1, 2, 3]),
        "PE-spatial-L(7~13)": ("_mpe_spatial_large_patch14_448-fb", 1024, [0, 1, 2, 3]), "DINOv2-B": ("_mbasep14", 768, [0, 1, 2, 3, 4]), "CAPI(7~13)": ("_mcapi_vitl14_lvd", 1024, [0, 1, 2, 3]),
        "TIPS-L": ("_mlarge_patch14_reg1_tipsv2-webli", 1024, None), "TIPS-g": ("_mgiant_patch14_reg1_tipsv2-webli", 1536, None), "EVA02-MIM(7~13)": ("_meva02_large_patch14_224-mim_m38m", 1024, [0, 1, 2, 3])}
for nm, (tag, C, ly) in SPEC.items():
    if nm in KX: continue
    try:
        if ly is None:
            n = np.load(Dv + f"train_s0_k0{tag}.npy").shape[1] // (11 * C); ly = list(range(n))[:4]
        KX[nm] = kern_views(tag, C, ly); print("커널", nm, flush=True)
    except FileNotFoundError: print("없음", nm)
np.savez(cache, **KX)
C3 = dict(np.load("analysis/cj2_cache.npz")); KX["DPN92+98(/8+/16)"] = (C3["dpn92-mx_in1k@8"] + C3["dpn92-mx_in1k@16"] + C3["dpn98-mx_in1k@8"] + C3["dpn98-mx_in1k@16"]) / 2
KX["Fj4+DPN 결합"] = KS["F"] + KX["DPN92+98(/8+/16)"] / 2
for LAM_ in (None, LAMF):
    f = lambda K: fit(prep(K), LAM_)[0]; B = {k: f(K) for k, K in KS.items()}; ob = bl(B)
    print(f"\n=== λ = {'LOO 선택' if LAM_ is None else LAM_} === 기준 블렌드 {rm(ob):.3f} pw13 {rw(ob, w13):.3f}")
    for nm, K in KX.items():
        o = f(K); line = f"  {nm:20s} 단독 {rm(o):.3f} 잔차상관(블렌드) {np.corrcoef(o - y, ob - y)[0, 1]:.3f} |"
        if nm == "Fj4+DPN 결합":
            b = .45 * B["D"] + .25 * B["T"] + .30 * o; d = (b - y) ** 2 - (ob - y) ** 2; line += f" F 대체: {rm(b):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5, pw13 {rw(b, w13) - rw(ob, w13):+.3f})"
        else:
            for w in (.10, .20):
                b = (1 - w) * ob + w * o; d = (b - y) ** 2 - (ob - y) ** 2; line += f" +{w:.2f}: {rm(b) - rm(ob):+.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5, pw13 {rw(b, w13) - rw(ob, w13):+.3f})"
        print(line, flush=True)
