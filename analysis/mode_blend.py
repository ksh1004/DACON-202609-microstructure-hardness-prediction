"""모드별 블렌드 가중치: 구성원 OOF(전체 파이프라인 블록)로 고품질/저품질 각각의 비음수 가중치를 폴드 밖에서 추정(NNLS, 합 1로 정규화) → 폴드 안 평가."""
import sys; sys.path.insert(0, "analysis")
from blend_eval import *
from scipy.optimize import nnls
co, _ = grp(CNN); po = oof(E + "mr_v14_pp_sc")
ao, _ = grp([E + f"dnpp0.25_ff{s}" for s in FF]); do, _ = grp([E + f"mv16_dnppm0.05_ff{s}" for s in FF]); to, _ = grp([E + f"tips16_dnppm0.05_ff{s}" for s in FF])
fo, _ = grp([E + f"cnnf_{n}_ff{s}" for n in ("dn121", "r50a1", "r50a2") for s in FF])
M = {"C": co, "P": po, "A": ao, "D": do, "T": to, "F": fo}; names = list(M); X = np.column_stack([M[k] for k in names])
hq = sig < 10
v19a = .15 * co + .15 * po + .70 * (.45 * (.275 * ao + .425 * do) / .70 + .25 * to + .30 * fo)
rep("v19a (고정 가중치)", v19a)
def fitw(Xt, yt, shrink=0.0, w0=None):
    mu = yt.mean(); w, _ = nnls(Xt - mu, yt - mu); w = w / max(w.sum(), 1e-9)
    return w if w0 is None else (1 - shrink) * w + shrink * w0
w0 = np.array([.15, .15, .70 * .45 * .275 / .70, .70 * .45 * .425 / .70, .175, .21])
def nested(by_mode, shrink):
    o = np.zeros(len(y)); W = []
    for k in range(5):
        t, v = fold != k, fold == k
        if by_mode:
            for m in (hq, ~hq):
                w = fitw(X[t & m], y[t & m], shrink, w0); o[v & m] = X[v & m] @ w; W.append(w)
        else:
            w = fitw(X[t], y[t], shrink, w0); o[v] = X[v] @ w; W.append(w)
    return o, np.array(W)
for nm, bm, sh in (("전역 NNLS (폴드 밖)", False, 0.0), ("전역 NNLS, 기존 가중치 쪽으로 50% 축소", False, 0.5), ("모드별 NNLS (폴드 밖)", True, 0.0), ("모드별 NNLS, 50% 축소", True, 0.5), ("모드별 NNLS, 75% 축소", True, 0.75)):
    o, W = nested(bm, sh); rep(nm, o, v19a)
    if bm: print(f"{'':10s} 평균 가중치 {names}: 고품질 {W[0::2].mean(0).round(2)} | 저품질 {W[1::2].mean(0).round(2)}")
    else: print(f"{'':10s} 평균 가중치 {names}: {W.mean(0).round(2)}")
# 전체 데이터로 모드별 최적(낙관) — 방향 확인용
for nm, m in (("고품질", hq), ("저품질", ~hq)):
    w = fitw(X[m], y[m]); print(f"[{nm}] 전체 적합 가중치 {dict(zip(names, w.round(2)))} → RMSE {np.sqrt(((X[m] @ w - y[m]) ** 2).mean()):.3f} (v19a {np.sqrt(((v19a - y)[m] ** 2).mean()):.3f})")
