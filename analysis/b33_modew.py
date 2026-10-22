"""모드별(고품질 σ<10 / 저품질 σ≥10) 블렌드 가중치 — 중첩 교차검증으로 공정 비교.
바깥 폴드 k: 나머지 4폴드 OOF로 가중치(합 1, 비음수) 적합 → 폴드 k에 적용. 전역 가중 / 모드별 가중 / v26b 고정 가중 비교."""
import sys, io, contextlib; sys.path.insert(0, "analysis"); sys.argv = ["b32.py"]
with contextlib.redirect_stdout(io.StringIO()):
    exec(open("analysis/b32.py", encoding="utf-8").read())
from scipy.optimize import nnls
W = PRE["D→Fo, T→세 경로 평균"]; names = list(W); k = 1.03
X = np.stack([G[n][0] for n in names], 1)
def fitw(Xs, ys, ridge=0.0):
    # 합 1 제약: 마지막 열 기준 차분으로 nnls 대신 간단히 nnls + 정규화
    w, _ = nnls(Xs - c, ys - c); return w / w.sum() if w.sum() > 0 else np.ones(len(names)) / len(names)
def nested(mode):
    o = np.zeros(len(y))
    for f in range(5):
        t, v = fold != f, fold == f
        if mode == "global":
            w = fitw(X[t], y[t]); o[v] = c + k * (X[v] @ w - c)
        else:
            for m in (Lq, ~Lq):
                w = fitw(X[t & m], y[t & m]); o[v & m] = c + k * (X[v & m] @ w - c)
    return o
fixed = c + k * (X @ np.array([W[n] for n in names]) - c)
for nm, o in (("v26b 고정 가중", fixed), ("전역 가중 (중첩)", nested("global")), ("모드별 가중 (중첩)", nested("mode"))):
    print(f"{nm:20s} OOF {rm(o):.3f} | 고 {rm(o, ~Lq):.3f} 저 {rm(o, Lq):.3f} | pw13 {rm(o, w=w13):.3f}")
for m, lab in ((~Lq, "고품질"), (Lq, "저품질")):
    w = fitw(X[m], y[m]); print(lab, "전체 적합 가중:", dict(zip(names, np.round(w, 3))))

print("\n— P 비중 고정(0.1765), 나머지 4개만 적합 —")
iP = names.index("P"); oth = [i for i in range(len(names)) if i != iP]; wP = W["P"]
def fitw_fixP(Xs, ys):
    r = (ys - c) - wP * (Xs[:, iP] - c); w, _ = nnls(Xs[:, oth] - c, r)
    w = w / w.sum() * (1 - wP) if w.sum() > 0 else np.full(len(oth), (1 - wP) / len(oth)); full = np.zeros(len(names)); full[iP] = wP; full[oth] = w; return full
o = np.zeros(len(y)); ws = []
for f in range(5):
    t, v = fold != f, fold == f; w = fitw_fixP(X[t], y[t]); ws.append(w); o[v] = c + k * (X[v] @ w - c)
print(f"P 고정 중첩            OOF {rm(o):.3f} | 고 {rm(o, ~Lq):.3f} 저 {rm(o, Lq):.3f} | pw13 {rm(o, w=w13):.3f}")
print("폴드별 가중:"); [print("  ", dict(zip(names, np.round(w_, 3)))) for w_ in ws]
wf = fitw_fixP(X, y); print("전체 적합:", dict(zip(names, np.round(wf, 3))))
show("P 고정 전체 적합 (낙관적)", dict(zip(names, wf)))
