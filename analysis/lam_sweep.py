"""Ridge 정규화 세기 λ를 구성원별 LOO 최적(지금 방식) 대신 고정값으로 두었을 때 구성원·블렌드 OOF. σ48, 3계열 (D .45 T .25 F .30).
동기: 개별 구성원에 최적인 λ가 블렌드에도 최적이라는 보장이 없다 — 평균은 분산을 줄여 주므로 덜 정규화한 구성원이 유리할 수 있다."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
J = dict(np.load("analysis/jk_cache.npz")); C2 = dict(np.load("analysis/cj_cache.npz"))
KS = {"D": J["D16"], "T": J["T16"], "F": sum(C2[n] for n in ("densenet121-tv_in1k", "resnet50-a1_in1k", "resnet50-a2_in1k", "densenetblur121d-ra_in1k")) / 4}
lq = sig >= 10; LAM = np.logspace(-6, 3, 60)
def prep(Ks):
    out = []
    for k in range(5):
        t, v = fold != k, fold == k; K = Ks[k].astype(np.float64); cm = K[:, t].mean(1, keepdims=True); K = K - cm - cm.T + K[np.ix_(t, t)].mean()
        ev, Q = np.linalg.eigh(K[np.ix_(t, t)]); ev = np.clip(ev, 0, None); out.append((t, v, K[np.ix_(v, t)], ev, Q, Q.T @ (y[t] - y[t].mean()), np.trace(K[np.ix_(t, t)]) / t.sum()))
    return out
PR = {k: prep(K) for k, K in KS.items()}
def fit(pr, lam_rel=None):
    """lam_rel=None이면 LOO 선택(절편 보정 포함, RidgeCV와 같음). 반환: OOF, 폴드별 선택 λ"""
    o = np.zeros(len(y)); ls = []
    for t, v, Kvt, ev, Q, Qy, sc in pr:
        if lam_rel is None:
            best = None
            for l in LAM:
                lam = l * sc; a_ = Q @ (Qy / (ev + lam)); d = (Q ** 2) @ (1 / (ev + lam)) - 1 / (lam * t.sum()); e = ((a_ / d) ** 2).mean()
                if best is None or e < best[0]: best = (e, l)
            l = best[1]
        else: l = lam_rel
        ls.append(l); o[v] = Kvt @ (Q @ (Qy / (ev + l * sc))) + y[t].mean()
    return o, ls
rm = lambda o, m=slice(None): np.sqrt(((o - y)[m] ** 2).mean()); bl = lambda O: .45 * O["D"] + .25 * O["T"] + .30 * O["F"]
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166 = wg(16.6); rw = lambda o, w: np.sqrt(np.average((o - y) ** 2, weights=w))
q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
B = {}; print("LOO 선택 λ(대각 평균 대비, 폴드별):")
for k in PR: B[k], ls = fit(PR[k]); print(f"  {k}: {np.round(ls, 4)} → OOF {rm(B[k]):.3f}")
ob = bl(B); print(f"  블렌드 {rm(ob):.3f} pw13 {rw(ob, w13):.3f} | 잔차상관 D-T {np.corrcoef(B['D'] - y, B['T'] - y)[0, 1]:.3f} D-F {np.corrcoef(B['D'] - y, B['F'] - y)[0, 1]:.3f} T-F {np.corrcoef(B['T'] - y, B['F'] - y)[0, 1]:.3f}")
print("고정 λ (모든 구성원 같은 값):")
for l in (1e-6, 1e-4, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 0.1, 0.2, 0.5, 1.0):
    O = {k: fit(PR[k], l)[0] for k in PR}; o = bl(O); d = (o - y) ** 2 - (ob - y) ** 2; cv = np.cov(y, o)
    print(f"  λ {l:<7g}: D {rm(O['D']):.3f} T {rm(O['T']):.3f} F {rm(O['F']):.3f} → 블렌드 {rm(o):.3f} pw13 {rw(o, w13):.3f} Public꼴 {rw(o, w166):.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | 가운데 {d[mid].sum() / 500:+.2f} 양끝 {d[~mid].sum() / 500:+.2f} | 고 {rm(o, ~lq):.3f} 저 {rm(o, lq):.3f} | 기울기 {cv[0, 1] / cv[1, 1]:.3f} 예측 std {o.std():.2f} | 잔차상관 D-F {np.corrcoef(O['D'] - y, O['F'] - y)[0, 1]:.3f}")
