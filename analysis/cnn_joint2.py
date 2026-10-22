"""고정 CNN 결합 Ridge에 DPN 계열(stride 8 + stride 16 두 단계)을 넣었을 때 (train σ48, 8보기 스캔 특징, 3계열 블렌드 D .45 T .25 F .30)."""
import sys, os; sys.path.insert(0, "analysis")
from blend_eval import *
from joint_kernel import krr, GR, NB
D = "outputs/probe2_views/unet_cm7_am22_l1_i10000_boot_fflowpass48_low/"
rm = lambda o: np.sqrt(((o - y) ** 2).mean())
def kern(A):
    C = A.shape[1] // NB; A = A.astype(np.float64).reshape(len(A), 1, NB, C); Ks = []
    for k in range(5):
        t = fold != k; K = np.zeros((len(y), len(y)))
        for blocks, g in GR:
            Z = A[:, :, list(blocks)].reshape(len(A), -1); mu, sd = Z[t].mean(0), Z[t].std(0); keep = sd > 1e-10
            Z = np.clip((Z[:, keep] - mu[keep]) / sd[keep], -6, 6); K += g * (Z @ Z.T) / keep.sum()
        Ks.append(K)
    return np.array(Ks)
cache = "analysis/cj2_cache.npz"; KS = dict(np.load(cache)) if os.path.exists(cache) else {}
for nm in ("densenet121-tv_in1k", "densenetblur121d-ra_in1k", "resnet50-a1_in1k", "resnet50-a2_in1k", "dpn92-mx_in1k", "dpn98-mx_in1k", "dpn107-mx_in1k", "dpn131-mx_in1k", "dpn68b-ra_in1k"):
    Z = np.load(D + f"cnnscan_{nm}_512_train.npz", allow_pickle=True); info = [s.split("|") for s in Z["info"]]
    for r in ("8", "16"):
        js = [j for j, i in enumerate(info) if i[2] == r]
        if js and f"{nm}@{r}" not in KS: KS[f"{nm}@{r}"] = kern(Z[f"f{js[-1]}"])
np.savez(cache, **KS)
J = dict(np.load("analysis/jk_cache.npz")); oD, oT = krr(J["D16"]), krr(J["T16"]); bl = lambda oF: .45 * oD + .25 * oT + .30 * oF
k8 = lambda n: KS[n + "@8"]; k816 = lambda n: KS[n + "@8"] + (KS[n + "@16"] if n + "@16" in KS else 0)
dn, db, a1, a2, d92, d98, d107, d131 = "densenet121-tv_in1k", "densenetblur121d-ra_in1k", "resnet50-a1_in1k", "resnet50-a2_in1k", "dpn92-mx_in1k", "dpn98-mx_in1k", "dpn107-mx_in1k", "dpn131-mx_in1k"
fj4 = krr(sum(k8(n) for n in (dn, db, a1, a2)) / 4); base = bl(fj4)
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166 = wg(16.6)
rw = lambda o, w: np.sqrt(np.average((o - y) ** 2, weights=w)); q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1]); Lq = sig >= 10
print("단독 (/8 | /8+/16): " + ", ".join(f"{n.split('-')[0]} {rm(krr(k8(n))):.3f} | {rm(krr(k816(n))):.3f}" for n in (dn, db, a1, a2, d92, d98, d107, d131)))
def show(nm, K):
    j = krr(K); o = bl(j); d = (o - y) ** 2 - (base - y) ** 2
    print(f"  {nm:46s} F {rm(j):.3f} | 블렌드 {rm(o):.3f} pw13 {rw(o, w13):.3f} Public꼴 {rw(o, w166):.3f} 고 {np.sqrt(((o - y)[~Lq] ** 2).mean()):.3f} 저 {np.sqrt(((o - y)[Lq] ** 2).mean()):.3f} | Fj4 대비 Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | 가운데 {d[mid].sum() / 500:+.2f} 양끝 {d[~mid].sum() / 500:+.2f} | D와 잔차상관 {np.corrcoef(j - y, oD - y)[0, 1]:.3f}")
    return j
show("Fj4 (기준: dn121·blur121d·r50a1·r50a2 /8)", sum(k8(n) for n in (dn, db, a1, a2)))
show("dpn92 (/8+/16) 단독", k816(d92)); show("dpn98 (/8+/16) 단독", k816(d98))
show("dpn92 + dpn98 (/8+/16)", k816(d92) + k816(d98))
show("dpn92 + dpn98 + dpn107 (/8+/16)", k816(d92) + k816(d98) + k816(d107))
show("DPN 4종 (92·98·107·131, /8+/16)", sum(k816(n) for n in (d92, d98, d107, d131)))
show("Fj4 + dpn92 + dpn98 (/8)", sum(k8(n) for n in (dn, db, a1, a2, d92, d98)))
show("Fj4 + dpn92 + dpn98 (DPN은 /8+/16)", sum(k8(n) for n in (dn, db, a1, a2)) + k816(d92) + k816(d98))
show("Fj4(가중 1) + DPN 2종(/8+/16, 가중 2)", sum(k8(n) for n in (dn, db, a1, a2)) / 4 + (k816(d92) + k816(d98)) / 2)
show("dn121 + blur121d + dpn92 + dpn98 (DPN /8+/16)", k8(dn) + k8(db) + k816(d92) + k816(d98))
show("Fj4 전부 /8+/16 + DPN 2종 /8+/16", sum(k816(n) for n in (dn, db, a1, a2, d92, d98)))
jd = krr(k816(d92) + k816(d98)); print("예측 수준: Fj4와 DPN 결합의 평균을 F로 → 블렌드 %.3f | F를 Fj4 .15 + DPN .15로 → %.3f | 잔차상관(Fj4, DPN) %.3f" % (rm(bl((fj4 + jd) / 2)), rm(.45 * oD + .25 * oT + .15 * fj4 + .15 * jd), np.corrcoef(fj4 - y, jd - y)[0, 1]))
