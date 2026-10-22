"""예측 압축이 미세구조의 조대함(q_blur_dn = 디노이즈 이미지의 매끄러움)에 따라 달라지는지: 조건부 보정을 폴드 밖에서 맞춰 평가.
보정식은 폴드 train 쪽 OOF 예측으로 맞추고 폴드 밖에 적용 (5폴드). 기준 = v21a 블렌드 OOF."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
G = dict(C=grp(CNN), P=(oof(E + "mr_v14_pp_sc"), None), A=grp([E + f"dnpp0.25_ff{f}" for f in FF]), D=grp([E + f"mv16_dnppm0.05_ff{f}" for f in FF]),
         T=grp([E + f"tips16_dnppm0.05_ff{f}" for f in FF]), J=grp([E + f"cnnfj4_ff{f}" for f in FF]))
W = dict(C=.15, P=.15, A=.12375, D=.19125, T=.175, J=.21); o = sum(v * G[k][0] for k, v in W.items()); c = y.mean()
F = pd.read_csv("outputs/features/train_features.csv").set_index("ID").loc[ids]; z = F.q_blur_dn.values; lq = (sig >= 10).astype(float)
wg = lambda s: np.exp(-0.5 * ((y - y.mean()) / s) ** 2) / np.exp(-0.5 * ((y - y.mean()) / y.std()) ** 2); w166 = wg(16.6)
rm = lambda p, w=None: np.sqrt(np.average((p - y) ** 2, weights=w)); q = np.quantile(y, [.1, .9]); mid = (y >= q[0]) & (y < q[1])
def cvfit(design, ridge=0.0):
    """design(idx) → 회귀 행렬. 폴드 밖 예측 반환"""
    p = np.zeros(len(y)); X = design()
    for k in range(5):
        t, v = fold != k, fold == k; A = X[t]; R = ridge * np.eye(A.shape[1]); R[0, 0] = 0
        b = np.linalg.solve(A.T @ A + R, A.T @ y[t]); p[v] = X[v] @ b
    return p, np.linalg.lstsq(X, y, rcond=None)[0]
def show(nm, p, coef=None):
    d = (p - y) ** 2 - (o - y) ** 2
    print(f"{nm:46s} OOF {rm(p):.3f} ({rm(p) - rm(o):+.3f}) pw13 {rm(p, w13) - rm(o, w13):+.3f} Public꼴 {rm(p, w166) - rm(o, w166):+.3f} | Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f} 폴드 {sum(d[fold == k].mean() < 0 for k in range(5))}/5 | 가운데 {d[mid].sum() / 500:+.2f} 양끝 {d[~mid].sum() / 500:+.2f}" + (f" | 계수 {np.round(coef, 3)}" if coef is not None else ""))
e = o - c; zs = (z - z.mean()) / z.std(); one = np.ones(len(y))
print(f"기준 v21a OOF {rm(o):.3f} | q_blur_dn과 경도 상관 {np.corrcoef(z, y)[0, 1]:.3f}, 예측과 {np.corrcoef(z, o)[0, 1]:.3f}")
show("전체 한 기울기 (절편+기울기)", *cvfit(lambda: np.column_stack([one, e])))
show("예측의 2차항", *cvfit(lambda: np.column_stack([one, e, e ** 2])))
show("기울기가 조대함에 선형 (e, z, e·z)", *cvfit(lambda: np.column_stack([one, e, zs, e * zs])))
show("  〃 주효과 없이 (e, e·z)", *cvfit(lambda: np.column_stack([one, e, e * zs])))
show("기울기가 모드에 (e, e·저품질)", *cvfit(lambda: np.column_stack([one, e, e * lq])))
show("조대함 + 모드 (e, e·z, e·저품질)", *cvfit(lambda: np.column_stack([one, e, e * zs, e * lq])))
show("조대함 + 2차 (e, e², e·z)", *cvfit(lambda: np.column_stack([one, e, e ** 2, e * zs])))
for th in (0.45, 0.48, 0.50, 0.52):
    hi = (z >= th).astype(float); show(f"조대 여부 2구간 (q_blur_dn ≥ {th}: {int(hi.sum())}장)", *cvfit(lambda: np.column_stack([one, e, e * hi])))
zc = np.clip(zs, -1.5, 2.5); show("e·z (z 절단 −1.5~2.5)", *cvfit(lambda: np.column_stack([one, e, e * zc])))
print("\n다른 조대함 지표로 (e, e·z) — 수작업 특징 중 기울기 상호작용이 큰 것 상위:")
res = []
for col in F.columns:
    v = F[col].values.astype(float)
    if not np.isfinite(v).all() or v.std() < 1e-9: continue
    vs = np.clip((v - v.mean()) / v.std(), -3, 3); p, cf = cvfit(lambda: np.column_stack([one, e, e * vs])); d = (p - y) ** 2 - (o - y) ** 2
    res.append((d.mean(), d.std() / np.sqrt(500), col, cf[2], np.corrcoef(v, z)[0, 1]))
for dm, se, col, b, rz in sorted(res)[:14]: print(f"  {col:22s} Δ² {dm:+.2f}±{se:.2f} | 상호작용 계수 {b:+.3f} | q_blur_dn과 상관 {rz:+.2f}")
