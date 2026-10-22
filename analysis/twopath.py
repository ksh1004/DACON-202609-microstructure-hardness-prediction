"""두 디노이즈 경로(자체 U-Net 부트스트랩 / 공개 DRUNet) 특징을 계열마다 한 Ridge에 결합했을 때 (σ48, train 8보기, 3계열 블렌드 D .45 T .25 F .30)."""
import sys, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()): import lam_sweep as L
from lam_sweep import *
GR = (((0, 1, 2, 3, 4, 5), 1.0), ((6, 7, 8, 9), 0.25), ((10,), 0.05)); BT, DR = "unet_cm7_am22_l1_i10000_boot_fflowpass48_low", "drunet_fflowpass48_low"
def kmat(X, NL, C):
    X = X.astype(np.float64).reshape(500, NL, 11, C); Ks = []
    for k in range(5):
        t = fold != k; K = np.zeros((500, 500))
        for blocks, g in GR:
            Z = X[:, :, list(blocks)].reshape(500, -1); mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10; Z = np.clip((Z[:, keep] - mu[keep]) / sd[keep], -6, 6); K += g * (Z @ Z.T) / keep.sum()
        Ks.append(K)
    return np.array(Ks)
kv = lambda stem, tag, NL, C: kmat(np.mean([np.load(f"outputs/probe2_views/{stem}/train_s0_k{k}{tag}.npy").astype(np.float32) for k in range(8)], 0), NL, C)
ke = lambda path: kmat(np.load(path + "/emb.npz")["train"], 1, 512)
K = {"D_u": kv(BT, "", 4, 1024), "D_d": kv(DR, "", 4, 1024), "T_u": dict(np.load("analysis/jk_cache.npz"))["T16"], "T_d": kv(DR, "_mso400m_patch14_reg1_tipsv2-webli", 4, 1152)}
C2 = dict(np.load("analysis/cj_cache.npz")); four = ("densenet121-tv_in1k", "resnet50-a1_in1k", "resnet50-a2_in1k", "densenetblur121d-ra_in1k")
K["F_u"] = sum(C2[n] for n in four) / 4
K["F_d"] = sum(ke(f"outputs/probe2/cnn_{m}_s2_512_Pphase2m_T8_S{DR}") for m in ("densenet121.tv_in1k", "resnet50.a1_in1k", "resnet50.a2_in1k", "densenetblur121d.ra_in1k")) / 4
np.savez("analysis/twopath_cache.npz", **K)
lq = sig >= 10
for lam in (None, 0.01):
    f = lambda Km: fit(prep(Km), lam)[0]; O = {}
    print(f"\n=== λ {'LOO' if lam is None else lam} ===")
    for fam in "DTF":
        O[fam + "_u"], O[fam + "_d"] = f(K[fam + "_u"]), f(K[fam + "_d"]); O[fam + "_j"] = f(K[fam + "_u"] + K[fam + "_d"]); O[fam + "_a"] = (O[fam + "_u"] + O[fam + "_d"]) / 2
        print(f"  {fam}: U-Net {rm(O[fam + '_u']):.3f} (고 {rm(O[fam + '_u'], ~lq):.2f} 저 {rm(O[fam + '_u'], lq):.2f}) | DRUNet {rm(O[fam + '_d']):.3f} (고 {rm(O[fam + '_d'], ~lq):.2f} 저 {rm(O[fam + '_d'], lq):.2f}) | 잔차상관 {np.corrcoef(O[fam + '_u'] - y, O[fam + '_d'] - y)[0, 1]:.3f} | 예측 평균 {rm(O[fam + '_a']):.3f} | 결합 {rm(O[fam + '_j']):.3f} (고 {rm(O[fam + '_j'], ~lq):.2f} 저 {rm(O[fam + '_j'], lq):.2f})")
    bl = lambda s: .45 * O["D_" + s[0]] + .25 * O["T_" + s[1]] + .30 * O["F_" + s[2]]; base = bl("uuu")
    for s, nm in (("uuu", "모두 U-Net 경로 (기준)"), ("juu", "D만 결합"), ("uju", "T만 결합"), ("uuj", "F만 결합"), ("jjj", "세 계열 모두 결합"), ("aaa", "세 계열 모두 예측 평균"), ("ddd", "모두 DRUNet 경로")):
        o = bl(s); d = (o - y) ** 2 - (base - y) ** 2; cv = np.cov(y, o)
        print(f"    {nm:22s} 블렌드 {rm(o):.3f} pw13 {rw(o, w13):.3f} Public꼴 {rw(o, w166):.3f} | 고 {rm(o, ~lq):.3f} 저 {rm(o, lq):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | 가운데 {d[mid].sum() / 500:+.2f} 양끝 {d[~mid].sum() / 500:+.2f} | 기울기 {cv[0, 1] / cv[1, 1]:.3f}")
