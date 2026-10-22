"""모드별 블렌드 가중치 (중첩 CV: 4폴드로 비음수 가중 적합 → 남은 폴드 평가). 전역 가중 적합과 비교."""
import sys, io, contextlib; sys.path.insert(0, "analysis")
with contextlib.redirect_stdout(io.StringIO()):
    from b30 import *
from scipy.optimize import nnls
names = list(W25); M = np.stack([G[n][0] for n in names], 1); k = 1.03
def fitw(idx):
    w, _ = nnls(M[idx] - c, (y[idx] - c) / k); return w
og, om, of = np.zeros(500), np.zeros(500), np.zeros(500)
for f in range(5):
    t, v = fold != f, fold == f
    og[v] = c + k * ((M[v] - c) @ fitw(t))
    for m in (Lq, ~Lq): om[v & m] = c + k * ((M[v & m] - c) @ fitw(t & m))
    of[v] = c + k * ((M[v] - c) @ np.array([W25[n] for n in names]))
for nm, o in (("고정 W25", of), ("전역 적합", og), ("모드별 적합", om)):
    d = (o - y) ** 2 - (of - y) ** 2
    print(f"{nm:10s} OOF {rm(o):.3f} 고 {rm(o, ~Lq):.3f} 저 {rm(o, Lq):.3f} pw13 {rm(o, w=w13):.3f} Δ² {d.mean():+.2f}±{d.std() / np.sqrt(500):.2f}")
print("전체 적합 가중:", dict(zip(names, fitw(np.ones(500, bool)).round(3))))
print("고품질 가중:", dict(zip(names, fitw(~Lq).round(3)))); print("저품질 가중:", dict(zip(names, fitw(Lq).round(3))))
