"""변동 계수(varying-coefficient) Ridge: 특징→경도 관계가 미세구조의 조대함·촬영 모드에 따라 달라지게 허용.
K_vc(i,j) = K_feat(i,j) × (1 + ρ·k_z(z_i, z_j)),  k_z = 조대함 z(q_blur_dn)의 RBF 또는 같은 그룹 여부.
근거: 조대한 이미지 구간에서 실제~예측 기울기가 1.4~1.6(예측이 덜 반응) — 사후 보정은 폴드 밖에서 불안정했으므로 모델 안에서 풀어 본다. σ48, 3계열."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from joint_kernel import krr
J = dict(np.load("analysis/jk_cache.npz")); C2 = dict(np.load("analysis/cj_cache.npz"))
KS = {"D": J["D16"], "T": J["T16"], "F": sum(C2[n] for n in ("densenet121-tv_in1k", "resnet50-a1_in1k", "resnet50-a2_in1k", "densenetblur121d-ra_in1k")) / 4}
F = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[ids]; z = F.q_blur_dn.values; lq = sig >= 10
rm = lambda o, m=slice(None): np.sqrt(((o - y)[m] ** 2).mean()); bl = lambda O: .45 * O["D"] + .25 * O["T"] + .30 * O["F"]
base = {k: krr(K) for k, K in KS.items()}; ob = bl(base)
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166 = wg(16.6)
rw = lambda o, w: np.sqrt(np.average((o - y) ** 2, weights=w)); q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1]); co = z >= .5
print(f"기준: " + " ".join(f"{k} {rm(v):.3f}" for k, v in base.items()) + f" → 블렌드 {rm(ob):.3f} (고 {rm(ob, ~lq):.3f} 저 {rm(ob, lq):.3f} | 조대 {rm(ob, co):.3f} 비조대 {rm(ob, ~co):.3f})")
def show(nm, G):
    O = {k: krr(K * (1 + G)[None]) for k, K in KS.items()}; o = bl(O); d = (o - y) ** 2 - (ob - y) ** 2; cv = np.cov(y[co], o[co])
    print(f"  {nm:34s} " + " ".join(f"{k} {rm(v):.3f}" for k, v in O.items()) + f" → {rm(o):.3f} pw13 {rw(o, w13) - rw(ob, w13):+.3f} Public꼴 {rw(o, w166) - rw(ob, w166):+.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | 가운데 {d[mid].sum() / 500:+.2f} 양끝 {d[~mid].sum() / 500:+.2f} | 조대 {rm(o, co):.3f} 비조대 {rm(o, ~co):.3f} 조대 기울기 {cv[0, 1] / cv[1, 1]:.2f}")
    return o
zs = (z - z.mean()) / z.std()
for h in (0.5, 1.0, 2.0):
    Kz = np.exp(-(zs[:, None] - zs[None, :]) ** 2 / (2 * h * h))
    for rho in (0.3, 1.0, 3.0): show(f"조대함 RBF h={h} ρ={rho}", rho * Kz)
same = lambda g: (g[:, None] == g[None, :]).astype(float)
for nm, g in (("조대 여부(z≥0.5)", co), ("모드(σ≥10)", lq), ("모드×조대 4그룹", lq * 2 + co)):
    for rho in (0.3, 1.0): show(f"같은 그룹: {nm} ρ={rho}", rho * same(g.astype(int)))
ls = (np.log(sig) - np.log(sig).mean()) / np.log(sig).std(); Ks = np.exp(-(ls[:, None] - ls[None, :]) ** 2 / 2)
show("노이즈 σ RBF ρ=1", 1.0 * Ks); show("조대함 RBF(h1) + 모드, ρ 1+1", np.exp(-(zs[:, None] - zs[None, :]) ** 2 / 2) + same(lq.astype(int)))
