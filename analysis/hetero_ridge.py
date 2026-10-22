"""이분산 Ridge: 표본마다 라벨 잡음 크기가 다르다고 보고 (K + λ·diag(n_i))⁻¹ 로 적합. n_i = 그룹별 잡음 분산 비.
저품질·조대 이미지의 잔차 분산이 2~3배 큰데 지금은 모든 표본을 같은 잡음으로 다룬다. σ48, 3계열 블렌드."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
J = dict(np.load("analysis/jk_cache.npz")); C2 = dict(np.load("analysis/cj_cache.npz"))
KS = {"D": J["D16"], "T": J["T16"], "F": sum(C2[n] for n in ("densenet121-tv_in1k", "resnet50-a1_in1k", "resnet50-a2_in1k", "densenetblur121d-ra_in1k")) / 4}
F_ = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[ids]; z = F_.q_blur_dn.values; lq = sig >= 10; co = z >= .5
LAM = np.logspace(-6, 3, 60)
def krr_w(Ks, nv):
    """nv: 표본별 상대 잡음 분산 (평균 1로 정규화). LOO는 가중 잔차 기준으로 λ 선택."""
    o = np.zeros(len(y)); nv = nv / nv.mean()
    for k in range(5):
        t, v = fold != k, fold == k; K = Ks[k].astype(np.float64); w = 1 / nv[t]; mu_y = np.average(y[t], weights=w)
        cm = (K[:, t] * w).sum(1, keepdims=True) / w.sum(); K = K - cm - cm.T + (K[np.ix_(t, t)] * np.outer(w, w)).sum() / w.sum() ** 2
        s = np.sqrt(w); Kt = K[np.ix_(t, t)] * np.outer(s, s); ev, Q = np.linalg.eigh(Kt); ev = np.clip(ev, 0, None); yc = (y[t] - mu_y) * s; Qy = Q.T @ yc; best = None
        for lam in LAM * np.trace(Kt) / t.sum():
            a_ = Q @ (Qy / (ev + lam)); d = (Q ** 2) @ (1 / (ev + lam)); e = ((a_ / d) ** 2).mean()
            if best is None or e < best[0]: best = (e, a_)
        o[v] = K[np.ix_(v, t)] @ (best[1] * s) + mu_y
    return o
rm = lambda o, m=slice(None): np.sqrt(((o - y)[m] ** 2).mean()); bl = lambda O: .45 * O["D"] + .25 * O["T"] + .30 * O["F"]
base = {k: krr_w(K, np.ones(len(y))) for k, K in KS.items()}; ob = bl(base)
print("기준: " + " ".join(f"{k} {rm(v):.3f}" for k, v in base.items()) + f" → 블렌드 {rm(ob):.3f} (고 {rm(ob, ~lq):.3f} 저 {rm(ob, lq):.3f})")
def show(nm, nv):
    O = {k: krr_w(K, nv.astype(float)) for k, K in KS.items()}; o = bl(O); d = (o - y) ** 2 - (ob - y) ** 2
    print(f"  {nm:40s} " + " ".join(f"{k} {rm(v):.3f}" for k, v in O.items()) + f" → {rm(o):.3f} (Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}, {sum(d[fold == k].mean() < 0 for k in range(5))}/5) 고 {rm(o, ~lq):.3f} 저 {rm(o, lq):.3f}")
for r in (0.5, 1.5, 2.0, 3.0, 5.0): show(f"저품질 잡음 ×{r}", np.where(lq, r, 1.0))
for r in (2.0, 3.0): show(f"조대(z≥0.5) 잡음 ×{r}", np.where(co, r, 1.0))
show("저품질 ×2, 조대 ×2 (곱)", np.where(lq, 2.0, 1.0) * np.where(co, 2.0, 1.0))
show("고품질·조대만 ×3", np.where(~lq & co, 3.0, 1.0))
